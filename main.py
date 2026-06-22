"""
AI Comment Moderator API
========================
Endpoints:
  POST  /moderate         — submit a comment for AI moderation
  POST  /appeal           — appeal a rejected decision
  GET   /log              — retrieve the full moderation log
  GET   /stats            — aggregate moderation statistics
  PATCH /log/{comment_id} — admin: override a decision
  GET   /health           — health check

Run with:
  uvicorn main:app --reload
"""

import logging
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
from uuid import uuid4

load_dotenv()

import anthropic
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from models import (
    AdminOverrideRequest,
    AppealRequest,
    AppealResponse,
    CommentRequest,
    FinalDecision,
    LogEntry,
    ModerationDecision,
    ModerationResponse,
    ModerationStats,
)
from moderator import moderate_comment, moderate_appeal
from rate_limiter import moderate_limiter, appeal_limiter
from storage import store
from webhook import send_flagged_webhook

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Webhook config
# ---------------------------------------------------------------------------

WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # Optional[str]

# ---------------------------------------------------------------------------
# IP-based rate limiting (fallback / per-IP protection)
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Comment Moderator",
    description=(
        "Automatic comment moderation for PropertyTribes using Claude. "
        "Supports an appeal flow for rejected comments."
    ),
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Error handling helpers
# ---------------------------------------------------------------------------

def _ai_error_response(exc: Exception) -> JSONResponse:
    logger.exception("AI service error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": "AI service unavailable. Please try again later."},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/moderate",
    response_model=ModerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a comment for moderation",
)
@limiter.limit("60/minute")
async def moderate(request: Request, body: CommentRequest) -> ModerationResponse:
    """
    Submit a comment for AI moderation.

    Returns a decision (`approved`, `rejected`, or `flagged_for_review`),
    a confidence score, a brief reasoning, and a rejection category if applicable.

    Rate limited to 30 requests per user per minute (per user_id).
    A secondary IP-based limit of 60/minute acts as a hard ceiling.
    """
    # Per-user rate limit (keyed on user_id from body)
    if not moderate_limiter.is_allowed(body.user_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for user '{body.user_id}'. Max 30 requests per minute.",
        )

    try:
        result = await moderate_comment(body.comment, tribe=body.tribe)
    except anthropic.APIError as exc:
        return _ai_error_response(exc)
    except Exception as exc:
        return _ai_error_response(exc)

    comment_id = uuid4()
    now = datetime.now(timezone.utc)

    entry = LogEntry(
        comment_id=comment_id,
        user_id=body.user_id,
        comment=body.comment,
        tribe=body.tribe,
        decision=result["decision"],
        confidence=result["confidence"],
        reasoning=result["reasoning"],
        rejection_category=result["rejection_category"],
        timestamp=now,
    )
    store.add(entry)

    logger.info(
        "Moderated comment %s for user %s → %s (confidence=%.2f)",
        comment_id,
        body.user_id,
        result["decision"].value,
        result["confidence"],
    )

    # Fire webhook if content is flagged for human review
    if result["decision"] == ModerationDecision.FLAGGED_FOR_REVIEW and WEBHOOK_URL:
        try:
            send_flagged_webhook(
                webhook_url=WEBHOOK_URL,
                comment_id=comment_id,
                user_id=body.user_id,
                comment=body.comment,
                confidence=result["confidence"],
                reasoning=result["reasoning"],
                timestamp=now,
            )
        except Exception as exc:
            logger.warning("Webhook call failed unexpectedly: %s", exc)

    return ModerationResponse(
        comment_id=comment_id,
        decision=result["decision"],
        confidence=result["confidence"],
        reasoning=result["reasoning"],
        rejection_category=result["rejection_category"],
        timestamp=now,
    )


@app.post(
    "/appeal",
    response_model=AppealResponse,
    status_code=status.HTTP_200_OK,
    summary="Appeal a rejected moderation decision",
)
@limiter.limit("20/minute")
async def appeal(request: Request, body: AppealRequest) -> AppealResponse:
    """
    Submit an appeal for a comment that was previously rejected.

    The AI re-evaluates the **original comment alongside the appeal context**.
    This is a final decision — `approved` or `rejected`; no further appeals allowed.

    Rate limited to 5 appeals per user per 10 minutes (per user_id).
    """
    entry = store.get(body.comment_id)

    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No moderation record found for comment_id={body.comment_id}.",
        )

    if entry.decision != ModerationDecision.REJECTED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Appeals are only allowed for rejected comments. "
                f"This comment was '{entry.decision.value}'."
            ),
        )

    if entry.appealed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This comment has already been appealed. No further appeals are allowed.",
        )

    # Per-user rate limit on appeals (keyed on user_id from the original entry)
    if not appeal_limiter.is_allowed(entry.user_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Appeal rate limit exceeded for user '{entry.user_id}'. Max 5 appeals per 10 minutes.",
        )

    try:
        result = await moderate_appeal(
            entry.comment,
            body.appeal_context,
            original_decision=entry.decision.value,
            rejection_category=entry.rejection_category.value,
            original_reasoning=entry.reasoning,
            tribe=entry.tribe,
        )
    except anthropic.APIError as exc:
        return _ai_error_response(exc)
    except Exception as exc:
        return _ai_error_response(exc)

    now = datetime.now(timezone.utc)

    store.record_appeal(
        comment_id=body.comment_id,
        appeal_context=body.appeal_context,
        appeal_decision=result["appeal_decision"],
        appeal_reasoning=result["reasoning"],
    )

    logger.info(
        "Appeal for comment %s → %s",
        body.comment_id,
        result["appeal_decision"].value,
    )

    return AppealResponse(
        comment_id=body.comment_id,
        original_decision=entry.decision,
        appeal_decision=result["appeal_decision"],
        reasoning=result["reasoning"],
        timestamp=now,
    )


@app.get(
    "/log",
    response_model=List[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="Retrieve the full moderation log",
)
async def get_log(
    page: int = Query(1, ge=1, description="Page number (1-indexed)."),
    limit: int = Query(20, ge=1, le=100, description="Results per page (max 100)."),
) -> List[Dict[str, Any]]:
    """
    Returns moderation decisions in reverse-chronological order.

    Supports pagination via `page` and `limit` query parameters.
    Example: GET /log?page=2&limit=10
    """
    entries = store.all()
    entries.sort(key=lambda e: e.timestamp, reverse=True)

    start = (page - 1) * limit
    end = start + limit
    return [e.model_dump(mode="json") for e in entries[start:end]]


# ---------------------------------------------------------------------------
# Admin override endpoint
# ---------------------------------------------------------------------------

@app.patch(
    "/log/{comment_id}",
    status_code=status.HTTP_200_OK,
    summary="Admin: override a moderation decision",
)
async def admin_override(comment_id: str, body: AdminOverrideRequest) -> Dict[str, Any]:
    """
    Allow a human moderator to override any AI moderation decision.

    Typical use case: a comment was `flagged_for_review` and a moderator
    has reviewed it and wants to set a final `approved` or `rejected` decision.

    The original AI decision and reasoning are preserved in the log alongside
    the override, creating a full audit trail.
    """
    from uuid import UUID as _UUID
    try:
        uid = _UUID(comment_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid comment_id format: {comment_id!r}",
        )

    updated = store.record_admin_override(
        comment_id=uid,
        decision=body.decision,
        note=body.note,
    )

    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No moderation record found for comment_id={comment_id}.",
        )

    logger.info(
        "Admin override for comment %s → %s (note: %s)",
        comment_id,
        body.decision.value,
        body.note or "none",
    )

    return updated.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@app.get(
    "/stats",
    response_model=ModerationStats,
    status_code=status.HTTP_200_OK,
    summary="Moderation statistics",
)
async def get_stats(
    since: Optional[datetime] = None,
) -> ModerationStats:
    """
    Returns aggregate statistics across all moderation decisions.

    - **Decision breakdown** — counts and percentages for approved / rejected / flagged
    - **Average confidence** — mean AI confidence score (`null` when no data)
    - **Top rejection categories** — top 5 ranked by frequency (spam, promotional, etc.)
    - **Appeal stats** — total appeals, overturn rate, upheld rate
    - **Admin overrides** — number of human moderator interventions

    Use the optional `since` query parameter (ISO 8601 datetime) to scope stats
    to a time window, e.g. `GET /stats?since=2024-11-15T00:00:00Z` for today's activity.
    """
    return ModerationStats(**store.stats(since=since))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", include_in_schema=False)
async def health() -> Dict[str, Any]:
    return {"status": "ok", "log_entries": store.count()}
