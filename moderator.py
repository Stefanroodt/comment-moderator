"""
AI moderation logic.

All prompt construction and Claude API calls live here. Keeping this
separate from the route layer means the prompts can be iterated without
touching HTTP concerns, and each function is independently testable.

Uses AsyncAnthropic so Claude API calls are non-blocking — the FastAPI
event loop stays free to handle other requests while waiting for the AI.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

import anthropic

from models import FinalDecision, ModerationDecision, RejectionCategory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared async Claude client (reads ANTHROPIC_API_KEY from environment)
# Lazy-initialised so tests can import this module without a live API key.
# ---------------------------------------------------------------------------

_client: anthropic.AsyncAnthropic | None = None
MODEL = "claude-haiku-4-5-20251001"


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic()
    return _client


# ---------------------------------------------------------------------------
# Forum context (used in every prompt)
# ---------------------------------------------------------------------------

FORUM_CONTEXT = """
You are the AI content moderator for PropertyTribes (propertytribes.com), the UK's #1
forum for private landlords and property investors.

COMMUNITY PROFILE:
The forum serves private landlords, buy-to-let investors, letting agents, property
developers, and some tenants. It is organised into 40+ specialist "Tribes" (sub-forums)
including: Buy-to-Let, HMOs, Mortgages & Finance, Tax, Refurb/Develop, Property Management,
Problem Tenants, Tenant Referencing, Holiday Lets, Short Term Rentals, Rent-to-Rent,
Leasehold Property, Commercial Property, Auction Tribe, New Landlords, Scottish PRS,
Welsh PRS, Expat Investors, Wanted & Recommendations, Products & Services, Property Seminars,
Investors in Distress, and more.

CURRENT HOT TOPICS (approve discussion of these):
- The Renters' Rights Act and its implications for landlords
- EPC upgrade requirements and energy efficiency obligations
- Section 21 abolition and no-fault evictions
- Mortgage rate changes and buy-to-let finance
- Problem tenants, rent arrears, and eviction processes
- HMO licensing requirements (Article 4 directions, mandatory/additional licensing)
- Capital gains tax, stamp duty, and landlord taxation

APPROVE comments that:
- Ask genuine questions about any property investment or landlord/tenant topic
- Share personal experiences as a landlord or investor, even if negative or frustrated
- Complain about problem tenants or difficult situations — this is what "Problem Tenants" tribe is for
- Request recommendations for tradespeople, solicitors, letting agents, or services
  (the "Wanted & Recommendations" tribe exists specifically for this)
- Discuss UK property market trends, yields, regional analysis, or investment strategy
- Offer professional advice or share credentials when relevant
- Promote property-related products or services IF posted in "Products & Services" or
  "Property Seminars" tribes, or if the user discloses their affiliation clearly
- Respectfully challenge government policy, tenant advocacy positions, or forum opinions

REJECT comments that:
- Contain hate speech, threats, or personal abuse targeting individuals or groups
- Promote "no money down", "get rich quick", or misleading property investment schemes
  (PropertyTribes explicitly flags these as marketing hype)
- Are spam or undisclosed self-promotion — e.g., linking to a commercial service without
  disclosing affiliation, especially using pressure tactics ("limited time", "DM me")
- Spread false information about property law, tenancy rights, or tax regulations
  where the error could cause real financial or legal harm
- Are entirely off-topic (e.g., unrelated to property, landlording, or UK real estate)
- Contain targeted harassment of named individuals

FLAG FOR REVIEW (human moderator should decide):
- Borderline self-promotion where affiliation is unclear
- Claims about property law or tax that may be inaccurate but are not clearly false
- Content that is negative/critical of tenants in a way that could be legitimate venting
  OR could be discriminatory — context matters
- Posts promoting property education seminars or courses (legitimate if transparent,
  problematic if misleading about returns or using high-pressure tactics)
- Rent-to-rent or "creative finance" strategies that are legal but sometimes mis-sold
""".strip()


# ---------------------------------------------------------------------------
# Helper: parse Claude's JSON response safely
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from Claude's response text.
    Claude may wrap JSON in markdown code fences — this handles both cases.
    """
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        raw = json_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not brace_match:
            raise ValueError(f"No JSON object found in AI response: {text!r}")
        raw = brace_match.group(0)

    return json.loads(raw)


def _safe_decision(value: str, fallback: ModerationDecision) -> ModerationDecision:
    try:
        return ModerationDecision(value.lower())
    except ValueError:
        logger.warning("Unexpected decision value %r — falling back to %s", value, fallback)
        return fallback


def _safe_category(value: str) -> RejectionCategory:
    try:
        return RejectionCategory(value.lower())
    except ValueError:
        return RejectionCategory.NONE


def _clamp_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def moderate_comment(comment: str) -> Dict[str, Any]:
    """
    Async: send a comment to Claude for moderation.

    Returns a dict with keys:
        decision, confidence, reasoning, rejection_category
    """
    prompt = f"""
{FORUM_CONTEXT}

---

A user has submitted the following comment to the PropertyTribes forum.
Evaluate it and respond with ONLY a JSON object — no preamble, no explanation outside
the JSON.

Comment to evaluate:
<comment>
{comment}
</comment>

Respond with exactly this JSON structure:
{{
  "decision": "<approved | rejected | flagged_for_review>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<1-3 sentence explanation visible to moderators>",
  "rejection_category": "<spam | hate_speech | misinformation | off_topic | abusive | promotional | none>"
}}

Notes:
- rejection_category must be "none" if decision is "approved" or "flagged_for_review"
- confidence should reflect how clear-cut the decision is (1.0 = certain, 0.5 = borderline)
- reasoning should be concise but specific enough to justify a moderation action
""".strip()

    logger.info("Sending comment to Claude for moderation (length=%d)", len(comment))

    message = await _get_client().messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = message.content[0].text
    logger.debug("Claude raw response: %s", raw_text)

    parsed = _extract_json(raw_text)

    return {
        "decision": _safe_decision(parsed.get("decision", "flagged_for_review"), ModerationDecision.FLAGGED_FOR_REVIEW),
        "confidence": _clamp_confidence(parsed.get("confidence", 0.5)),
        "reasoning": str(parsed.get("reasoning", "No reasoning provided.")),
        "rejection_category": _safe_category(parsed.get("rejection_category", "none")),
    }


async def moderate_appeal(original_comment: str, appeal_context: str) -> Dict[str, Any]:
    """
    Async: re-evaluate a rejected comment in light of the user's appeal context.

    The prompt explicitly instructs Claude to genuinely reconsider — not just
    rubber-stamp the original decision. Returns a dict with:
        appeal_decision, reasoning
    """
    prompt = f"""
{FORUM_CONTEXT}

---

A user is appealing a moderation decision on PropertyTribes. Your job is to conduct a
GENUINE re-evaluation. Do not simply repeat the original rejection. Carefully read the
user's context and determine whether it changes your assessment.

Original comment (previously rejected):
<comment>
{original_comment}
</comment>

User's appeal explanation:
<appeal_context>
{appeal_context}
</appeal_context>

Consider:
1. Does the appeal context clarify the intent of the comment?
2. Does it provide professional credentials or evidence that change the risk assessment?
3. Would a reasonable PropertyTribes moderator approve this with the additional context?

This is a FINAL decision — there are no further appeals. Respond with ONLY a JSON object:
{{
  "appeal_decision": "<approved | rejected>",
  "reasoning": "<2-4 sentence explanation that acknowledges the appeal context>"
}}
""".strip()

    logger.info("Sending appeal to Claude for re-evaluation")

    message = await _get_client().messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = message.content[0].text
    logger.debug("Claude appeal raw response: %s", raw_text)

    parsed = _extract_json(raw_text)

    raw_decision = parsed.get("appeal_decision", "rejected")
    try:
        appeal_decision = FinalDecision(raw_decision.lower())
    except ValueError:
        logger.warning("Unexpected appeal_decision %r — defaulting to rejected", raw_decision)
        appeal_decision = FinalDecision.REJECTED

    return {
        "appeal_decision": appeal_decision,
        "reasoning": str(parsed.get("reasoning", "No reasoning provided.")),
    }
