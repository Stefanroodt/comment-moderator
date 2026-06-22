"""
Unit tests for the AI Comment Moderator API.

Strategy: mock the Claude API at the moderator-function level so tests
run instantly, deterministically, and without real API keys. We test that
the routes handle inputs, edge cases, and error conditions correctly.

Run with:
  pytest tests/ -v
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import app
from models import FinalDecision, ModerationDecision, RejectionCategory
from storage import store
from rate_limiter import moderate_limiter, appeal_limiter

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

APPROVED_RESULT = {
    "decision": ModerationDecision.APPROVED,
    "confidence": 0.95,
    "reasoning": "Genuine property question — approved.",
    "rejection_category": RejectionCategory.NONE,
}

REJECTED_RESULT = {
    "decision": ModerationDecision.REJECTED,
    "confidence": 0.92,
    "reasoning": "Spam / promotional content.",
    "rejection_category": RejectionCategory.SPAM,
}

FLAGGED_RESULT = {
    "decision": ModerationDecision.FLAGGED_FOR_REVIEW,
    "confidence": 0.55,
    "reasoning": "Borderline self-promotion — needs human review.",
    "rejection_category": RejectionCategory.NONE,
}

APPEAL_APPROVED = {
    "appeal_decision": FinalDecision.APPROVED,
    "reasoning": "Appeal context clarifies this is legitimate advice.",
}

APPEAL_REJECTED = {
    "appeal_decision": FinalDecision.REJECTED,
    "reasoning": "Appeal does not change the assessment.",
}


@pytest.fixture(autouse=True)
def clear_store():
    """Reset the in-memory store and rate limiters before each test."""
    store._entries.clear()
    moderate_limiter._windows.clear()
    appeal_limiter._windows.clear()
    yield
    store._entries.clear()
    moderate_limiter._windows.clear()
    appeal_limiter._windows.clear()


# ---------------------------------------------------------------------------
# POST /moderate — happy paths
# ---------------------------------------------------------------------------

class TestModerateHappyPaths:

    def test_approved_comment(self):
        with patch("main.moderate_comment", return_value=APPROVED_RESULT):
            resp = client.post("/moderate", json={
                "user_id": "user_1",
                "comment": "Has anyone used a letting agent in Bristol recently?",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "approved"
        assert 0.0 <= data["confidence"] <= 1.0
        assert "comment_id" in data
        assert "reasoning" in data
        assert "timestamp" in data

    def test_rejected_comment(self):
        with patch("main.moderate_comment", return_value=REJECTED_RESULT):
            resp = client.post("/moderate", json={
                "user_id": "user_2",
                "comment": "Buy my property leads — best in the UK! DM me.",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "rejected"
        assert data["rejection_category"] == "spam"

    def test_flagged_comment(self):
        with patch("main.moderate_comment", return_value=FLAGGED_RESULT):
            resp = client.post("/moderate", json={
                "user_id": "user_3",
                "comment": "I wrote a blog post about this topic, might be useful.",
            })
        assert resp.status_code == 200
        assert resp.json()["decision"] == "flagged_for_review"

    def test_decision_is_logged(self):
        with patch("main.moderate_comment", return_value=APPROVED_RESULT):
            resp = client.post("/moderate", json={
                "user_id": "user_1",
                "comment": "Question about HMO licensing.",
            })
        assert store.count() == 1


# ---------------------------------------------------------------------------
# POST /moderate — edge cases
# ---------------------------------------------------------------------------

class TestModerateEdgeCases:

    def test_empty_comment_rejected_by_validation(self):
        resp = client.post("/moderate", json={"user_id": "u1", "comment": ""})
        assert resp.status_code == 422

    def test_whitespace_only_comment_rejected(self):
        resp = client.post("/moderate", json={"user_id": "u1", "comment": "   "})
        assert resp.status_code == 422

    def test_missing_user_id(self):
        resp = client.post("/moderate", json={"comment": "Hello"})
        assert resp.status_code == 422

    def test_comment_at_max_length(self):
        long_comment = "A" * 5000
        with patch("main.moderate_comment", return_value=APPROVED_RESULT):
            resp = client.post("/moderate", json={"user_id": "u1", "comment": long_comment})
        assert resp.status_code == 200

    def test_comment_exceeds_max_length(self):
        too_long = "A" * 5001
        resp = client.post("/moderate", json={"user_id": "u1", "comment": too_long})
        assert resp.status_code == 422

    def test_ai_error_returns_502(self):
        import anthropic as _anthropic
        with patch("main.moderate_comment", side_effect=_anthropic.APIError("fail", request=None, body=None)):
            resp = client.post("/moderate", json={"user_id": "u1", "comment": "test"})
        assert resp.status_code == 502

    def test_internal_error_returns_500(self):
        """A bug in our own code (e.g. KeyError) returns 500, not 502."""
        with patch("main.moderate_comment", side_effect=KeyError("unexpected_key")):
            resp = client.post("/moderate", json={"user_id": "u1", "comment": "test"})
        assert resp.status_code == 500

    def test_unparseable_ai_response_fails_to_flagged(self):
        """When Claude returns prose with no valid JSON, the API fails closed to flagged_for_review — not 502."""
        from unittest.mock import AsyncMock, MagicMock
        mock_msg = MagicMock()
        mock_msg.stop_reason = "end_turn"
        mock_msg.content = [MagicMock(text="I think this comment is fine, probably {just approve} it.")]
        with patch("moderator._get_client") as mock_client:
            mock_client.return_value.messages.create = AsyncMock(return_value=mock_msg)
            resp = client.post("/moderate", json={"user_id": "u1", "comment": "test comment"})
        assert resp.status_code == 200
        assert resp.json()["decision"] == "flagged_for_review"

    def test_max_tokens_response_fails_to_flagged(self):
        """A truncated response (stop_reason=max_tokens) fails closed to flagged_for_review."""
        from unittest.mock import AsyncMock, MagicMock
        mock_msg = MagicMock()
        mock_msg.stop_reason = "max_tokens"
        mock_msg.content = [MagicMock(text='{"decision": "approved", "confidence": 0.9, "reason')]
        with patch("moderator._get_client") as mock_client:
            mock_client.return_value.messages.create = AsyncMock(return_value=mock_msg)
            resp = client.post("/moderate", json={"user_id": "u1", "comment": "test comment"})
        assert resp.status_code == 200
        assert resp.json()["decision"] == "flagged_for_review"


# ---------------------------------------------------------------------------
# POST /appeal — happy paths
# ---------------------------------------------------------------------------

class TestAppealHappyPaths:

    def _submit_and_reject(self) -> str:
        """Helper: submit a comment, patch it as rejected, return comment_id."""
        with patch("main.moderate_comment", return_value=REJECTED_RESULT):
            resp = client.post("/moderate", json={
                "user_id": "u1",
                "comment": "Visit my property investment site.",
            })
        return resp.json()["comment_id"]

    def test_successful_appeal_approved(self):
        comment_id = self._submit_and_reject()
        with patch("main.moderate_appeal", return_value=APPEAL_APPROVED):
            resp = client.post("/appeal", json={
                "comment_id": comment_id,
                "appeal_context": "I am a RICS-qualified surveyor sharing professional advice.",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["appeal_decision"] == "approved"
        assert data["original_decision"] == "rejected"

    def test_successful_appeal_rejected(self):
        comment_id = self._submit_and_reject()
        with patch("main.moderate_appeal", return_value=APPEAL_REJECTED):
            resp = client.post("/appeal", json={
                "comment_id": comment_id,
                "appeal_context": "I just wanted to share a link.",
            })
        assert resp.status_code == 200
        assert resp.json()["appeal_decision"] == "rejected"

    def test_appeal_updates_log(self):
        comment_id = self._submit_and_reject()
        with patch("main.moderate_appeal", return_value=APPEAL_APPROVED):
            client.post("/appeal", json={
                "comment_id": comment_id,
                "appeal_context": "Professional context here.",
            })
        from uuid import UUID
        entry = store.get(UUID(comment_id))
        assert entry is not None
        assert entry.appealed is True
        assert entry.appeal_decision == FinalDecision.APPROVED


# ---------------------------------------------------------------------------
# POST /appeal — error cases
# ---------------------------------------------------------------------------

class TestAppealErrorCases:

    def test_appeal_unknown_comment_id(self):
        resp = client.post("/appeal", json={
            "comment_id": str(uuid4()),
            "appeal_context": "I want to appeal.",
        })
        assert resp.status_code == 404

    def test_cannot_appeal_approved_comment(self):
        with patch("main.moderate_comment", return_value=APPROVED_RESULT):
            resp = client.post("/moderate", json={"user_id": "u1", "comment": "Good question"})
        comment_id = resp.json()["comment_id"]

        resp2 = client.post("/appeal", json={
            "comment_id": comment_id,
            "appeal_context": "Why would I appeal an approved comment?",
        })
        assert resp2.status_code == 400

    def test_cannot_appeal_twice(self):
        with patch("main.moderate_comment", return_value=REJECTED_RESULT):
            resp = client.post("/moderate", json={"user_id": "u1", "comment": "spam"})
        comment_id = resp.json()["comment_id"]

        with patch("main.moderate_appeal", return_value=APPEAL_REJECTED):
            client.post("/appeal", json={
                "comment_id": comment_id,
                "appeal_context": "First appeal.",
            })
            resp2 = client.post("/appeal", json={
                "comment_id": comment_id,
                "appeal_context": "Second attempt.",
            })
        assert resp2.status_code == 409

    def test_appeal_context_too_short(self):
        resp = client.post("/appeal", json={
            "comment_id": str(uuid4()),
            "appeal_context": "short",
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /log
# ---------------------------------------------------------------------------

class TestLog:

    def test_empty_log(self):
        resp = client.get("/log")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_log_contains_entries(self):
        with patch("main.moderate_comment", return_value=APPROVED_RESULT):
            client.post("/moderate", json={"user_id": "u1", "comment": "Comment 1"})
            client.post("/moderate", json={"user_id": "u2", "comment": "Comment 2"})
        resp = client.get("/log")
        assert len(resp.json()) == 2

    def test_log_entries_have_required_fields(self):
        with patch("main.moderate_comment", return_value=APPROVED_RESULT):
            client.post("/moderate", json={"user_id": "u1", "comment": "HMO question"})
        entry = client.get("/log").json()[0]
        for field in ("comment_id", "user_id", "comment", "decision",
                      "confidence", "reasoning", "timestamp", "appealed"):
            assert field in entry, f"Missing field: {field}"

    def test_log_is_reverse_chronological(self):
        with patch("main.moderate_comment", return_value=APPROVED_RESULT):
            client.post("/moderate", json={"user_id": "u1", "comment": "First"})
            client.post("/moderate", json={"user_id": "u2", "comment": "Second"})
        entries = client.get("/log").json()
        assert entries[0]["timestamp"] >= entries[1]["timestamp"]

    def test_log_pagination(self):
        with patch("main.moderate_comment", return_value=APPROVED_RESULT):
            for i in range(5):
                client.post("/moderate", json={"user_id": f"u{i}", "comment": f"Comment {i}"})
        page1 = client.get("/log?page=1&limit=3").json()
        page2 = client.get("/log?page=2&limit=3").json()
        assert len(page1) == 3
        assert len(page2) == 2
        ids1 = {e["comment_id"] for e in page1}
        ids2 = {e["comment_id"] for e in page2}
        assert ids1.isdisjoint(ids2)

    def test_page_zero_returns_422(self):
        """page=0 is invalid — pages are 1-indexed."""
        resp = client.get("/log?page=0")
        assert resp.status_code == 422

    def test_limit_too_large_returns_422(self):
        """limit above 100 is rejected."""
        resp = client.get("/log?limit=101")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Rate limiting (per user_id)
# ---------------------------------------------------------------------------

class TestRateLimiting:

    def test_moderate_rate_limit_blocks_after_threshold(self):
        """User exceeding 30 requests/minute gets a 429."""
        moderate_limiter.max_requests = 3
        with patch("main.moderate_comment", return_value=APPROVED_RESULT):
            for _ in range(3):
                resp = client.post("/moderate", json={"user_id": "heavy_user", "comment": "test"})
                assert resp.status_code == 200
            resp = client.post("/moderate", json={"user_id": "heavy_user", "comment": "test"})
            assert resp.status_code == 429
        moderate_limiter.max_requests = 30

    def test_rate_limit_is_per_user(self):
        """Different users have independent rate limit windows."""
        moderate_limiter.max_requests = 1
        with patch("main.moderate_comment", return_value=APPROVED_RESULT):
            resp1 = client.post("/moderate", json={"user_id": "user_a", "comment": "test"})
            resp2 = client.post("/moderate", json={"user_id": "user_b", "comment": "test"})
            assert resp1.status_code == 200
            assert resp2.status_code == 200
        moderate_limiter.max_requests = 30


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------

class TestWebhook:

    def test_webhook_fires_on_flagged_content(self):
        """A webhook POST is attempted when content is flagged_for_review."""
        with patch("main.moderate_comment", return_value=FLAGGED_RESULT), \
             patch("main.WEBHOOK_URL", "https://webhook.site/test"), \
             patch("main.send_flagged_webhook") as mock_webhook:
            client.post("/moderate", json={"user_id": "u1", "comment": "borderline post"})
            mock_webhook.assert_called_once()
            call_kwargs = mock_webhook.call_args.kwargs
            assert call_kwargs["webhook_url"] == "https://webhook.site/test"
            assert call_kwargs["user_id"] == "u1"

    def test_webhook_not_fired_on_approved(self):
        """No webhook for approved content."""
        with patch("main.moderate_comment", return_value=APPROVED_RESULT), \
             patch("main.WEBHOOK_URL", "https://webhook.site/test"), \
             patch("main.send_flagged_webhook") as mock_webhook:
            client.post("/moderate", json={"user_id": "u1", "comment": "good post"})
            mock_webhook.assert_not_called()

    def test_webhook_not_fired_when_url_not_configured(self):
        """No webhook attempt if WEBHOOK_URL is not set."""
        with patch("main.moderate_comment", return_value=FLAGGED_RESULT), \
             patch("main.WEBHOOK_URL", None), \
             patch("main.send_flagged_webhook") as mock_webhook:
            client.post("/moderate", json={"user_id": "u1", "comment": "borderline"})
            mock_webhook.assert_not_called()

    def test_webhook_failure_does_not_affect_response(self):
        """A webhook that raises an exception must not break the API response."""
        with patch("main.moderate_comment", return_value=FLAGGED_RESULT), \
             patch("main.WEBHOOK_URL", "https://webhook.site/test"), \
             patch("main.send_flagged_webhook", side_effect=Exception("network error")):
            resp = client.post("/moderate", json={"user_id": "u1", "comment": "borderline"})
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /stats
# ---------------------------------------------------------------------------

class TestStats:

    def test_empty_stats(self):
        """Stats on an empty store returns zeros and null confidence."""
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_comments"] == 0
        assert data["decisions"] == {"approved": 0, "rejected": 0, "flagged_for_review": 0}
        # avg_confidence must be null (not 0.0) so callers can tell "no data" from "zero confidence"
        assert data["avg_confidence"] is None
        assert data["top_rejection_categories"] == {}
        assert data["appeals"]["total"] == 0
        assert data["appeals"]["overturn_rate"] == 0.0
        assert data["admin_overrides"] == 0

    def test_decision_counts(self):
        """Decision breakdown reflects what was moderated."""
        with patch("main.moderate_comment", return_value=APPROVED_RESULT):
            client.post("/moderate", json={"user_id": "u1", "comment": "good question"})
            client.post("/moderate", json={"user_id": "u2", "comment": "another good one"})
        with patch("main.moderate_comment", return_value=REJECTED_RESULT):
            client.post("/moderate", json={"user_id": "u3", "comment": "spam comment"})

        resp = client.get("/stats")
        data = resp.json()
        assert data["total_comments"] == 3
        assert data["decisions"]["approved"] == 2
        assert data["decisions"]["rejected"] == 1
        assert data["decisions"]["flagged_for_review"] == 0

    def test_decision_percentages(self):
        """Percentages sum to 100 and reflect correct proportions."""
        with patch("main.moderate_comment", return_value=APPROVED_RESULT):
            client.post("/moderate", json={"user_id": "u1", "comment": "post one"})
        with patch("main.moderate_comment", return_value=REJECTED_RESULT):
            client.post("/moderate", json={"user_id": "u2", "comment": "spam post"})
            client.post("/moderate", json={"user_id": "u3", "comment": "more spam"})

        data = client.get("/stats").json()
        pcts = data["decision_percentages"]
        assert abs(pcts["approved"] - 33.3) < 0.2
        assert abs(pcts["rejected"] - 66.7) < 0.2
        total_pct = pcts["approved"] + pcts["rejected"] + pcts["flagged_for_review"]
        assert abs(total_pct - 100.0) < 0.5

    def test_avg_confidence(self):
        """Average confidence is the mean of all entries."""
        result_95 = {**APPROVED_RESULT, "confidence": 0.95}
        result_85 = {**REJECTED_RESULT, "confidence": 0.85}
        with patch("main.moderate_comment", return_value=result_95):
            client.post("/moderate", json={"user_id": "u1", "comment": "good"})
        with patch("main.moderate_comment", return_value=result_85):
            client.post("/moderate", json={"user_id": "u2", "comment": "spam"})

        data = client.get("/stats").json()
        assert abs(data["avg_confidence"] - 0.9) < 0.01

    def test_top_rejection_categories(self):
        """Rejection categories are ranked by frequency, excluding 'none'."""
        spam_result = {**REJECTED_RESULT, "rejection_category": RejectionCategory.SPAM}
        promo_result = {**REJECTED_RESULT, "rejection_category": RejectionCategory.PROMOTIONAL}

        with patch("main.moderate_comment", return_value=spam_result):
            client.post("/moderate", json={"user_id": "u1", "comment": "spam 1"})
            client.post("/moderate", json={"user_id": "u2", "comment": "spam 2"})
        with patch("main.moderate_comment", return_value=promo_result):
            client.post("/moderate", json={"user_id": "u3", "comment": "promo"})
        with patch("main.moderate_comment", return_value=APPROVED_RESULT):
            client.post("/moderate", json={"user_id": "u4", "comment": "good"})

        data = client.get("/stats").json()
        cats = data["top_rejection_categories"]
        assert cats["spam"] == 2
        assert cats["promotional"] == 1
        assert "none" not in cats
        # spam should come first (highest count)
        assert list(cats.keys())[0] == "spam"

    def test_appeal_overturn_rate(self):
        """Appeal stats reflect overturned and upheld counts correctly."""
        with patch("main.moderate_comment", return_value=REJECTED_RESULT):
            resp = client.post("/moderate", json={"user_id": "u1", "comment": "rejected"})
        comment_id = resp.json()["comment_id"]

        with patch("main.moderate_appeal", return_value=APPEAL_APPROVED):
            client.post("/appeal", json={
                "comment_id": comment_id,
                "appeal_context": "I am a professional with 10 years experience.",
            })

        data = client.get("/stats").json()
        assert data["appeals"]["total"] == 1
        assert data["appeals"]["overturned"] == 1
        assert data["appeals"]["upheld"] == 0
        assert data["appeals"]["overturn_rate"] == 1.0

    def test_effective_decision_in_stats(self):
        """Stats use effective_decision: a flagged comment that an admin approves counts as approved."""
        with patch("main.moderate_comment", return_value=FLAGGED_RESULT):
            resp = client.post("/moderate", json={"user_id": "u1", "comment": "borderline"})
        comment_id = resp.json()["comment_id"]

        client.patch(f"/log/{comment_id}", json={"decision": "approved", "note": "Verified."})

        data = client.get("/stats").json()
        assert data["decisions"]["approved"] == 1
        assert data["decisions"]["flagged_for_review"] == 0

    def test_admin_override_count(self):
        """Admin overrides are counted in stats."""
        with patch("main.moderate_comment", return_value=FLAGGED_RESULT):
            resp = client.post("/moderate", json={"user_id": "u1", "comment": "borderline"})
        comment_id = resp.json()["comment_id"]

        client.patch(f"/log/{comment_id}", json={"decision": "approved", "note": "Verified."})

        data = client.get("/stats").json()
        assert data["admin_overrides"] == 1

    def test_top_rejection_categories_capped_at_five(self):
        """top_rejection_categories never returns more than 5 entries."""
        from models import RejectionCategory
        categories = [
            RejectionCategory.SPAM,
            RejectionCategory.PROMOTIONAL,
            RejectionCategory.HATE_SPEECH,
            RejectionCategory.MISINFORMATION,
            RejectionCategory.OFF_TOPIC,
            RejectionCategory.ABUSIVE,
        ]
        for i, cat in enumerate(categories):
            result = {**REJECTED_RESULT, "rejection_category": cat}
            with patch("main.moderate_comment", return_value=result):
                client.post("/moderate", json={"user_id": f"u{i}", "comment": f"bad comment {i}"})

        data = client.get("/stats").json()
        assert len(data["top_rejection_categories"]) <= 5

    def test_since_filter_excludes_old_entries(self):
        """The since parameter scopes stats to entries on or after the given datetime."""
        from datetime import timezone
        from storage import store
        from models import LogEntry, ModerationDecision, RejectionCategory
        from uuid import uuid4

        # Inject an old entry directly into the store
        old_entry = LogEntry(
            comment_id=uuid4(),
            user_id="old_user",
            comment="old comment",
            decision=ModerationDecision.REJECTED,
            confidence=0.9,
            reasoning="spam",
            rejection_category=RejectionCategory.SPAM,
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        store.add(old_entry)

        # Add a recent entry via the API
        with patch("main.moderate_comment", return_value=APPROVED_RESULT):
            client.post("/moderate", json={"user_id": "new_user", "comment": "new good post"})

        # Stats without filter → both entries
        all_data = client.get("/stats").json()
        assert all_data["total_comments"] == 2

        # Stats with since=2024-01-01 → only the recent entry
        recent_data = client.get("/stats?since=2024-01-01T00:00:00Z").json()
        assert recent_data["total_comments"] == 1
        assert recent_data["decisions"]["approved"] == 1
        assert recent_data["decisions"]["rejected"] == 0

    def test_since_filter_empty_window_returns_null_confidence(self):
        """When since filters out all entries, avg_confidence is null not 0.0."""
        with patch("main.moderate_comment", return_value=APPROVED_RESULT):
            client.post("/moderate", json={"user_id": "u1", "comment": "old post"})

        # Far-future since → no entries match
        data = client.get("/stats?since=2099-01-01T00:00:00Z").json()
        assert data["total_comments"] == 0
        assert data["avg_confidence"] is None


# ---------------------------------------------------------------------------
# Tribe-aware moderation
# ---------------------------------------------------------------------------

class TestTribeAwareness:

    def test_no_tribe_still_works(self):
        """Omitting tribe is fine — backward compatible."""
        with patch("main.moderate_comment", return_value=APPROVED_RESULT) as mock:
            resp = client.post("/moderate", json={
                "user_id": "u1",
                "comment": "Good property question.",
            })
        assert resp.status_code == 200
        mock.assert_called_once_with("Good property question.", tribe=None)

    def test_known_tribe_is_passed_to_moderator(self):
        """A known tribe name is forwarded to moderate_comment."""
        with patch("main.moderate_comment", return_value=APPROVED_RESULT) as mock:
            resp = client.post("/moderate", json={
                "user_id": "u1",
                "comment": "Looking for a good letting agent in Leeds.",
                "tribe": "Wanted & Recommendations",
            })
        assert resp.status_code == 200
        mock.assert_called_once_with(
            "Looking for a good letting agent in Leeds.",
            tribe="Wanted & Recommendations",
        )

    def test_unknown_tribe_falls_back_gracefully(self):
        """An unrecognised tribe name is accepted — falls back to generic rules."""
        with patch("main.moderate_comment", return_value=APPROVED_RESULT) as mock:
            resp = client.post("/moderate", json={
                "user_id": "u1",
                "comment": "Interesting post about property.",
                "tribe": "Some Future Tribe Not In Our List",
            })
        assert resp.status_code == 200
        mock.assert_called_once_with(
            "Interesting post about property.",
            tribe="Some Future Tribe Not In Our List",
        )

    def test_tribe_is_stored_in_log(self):
        """Tribe is persisted in the log entry and visible in GET /log."""
        with patch("main.moderate_comment", return_value=REJECTED_RESULT):
            client.post("/moderate", json={
                "user_id": "u1",
                "comment": "Buy my course!",
                "tribe": "HMO Landlords",
            })
        log = client.get("/log").json()
        assert log[0]["tribe"] == "HMO Landlords"

    def test_tribe_field_too_long_rejected(self):
        """Tribe names over 100 characters fail validation."""
        resp = client.post("/moderate", json={
            "user_id": "u1",
            "comment": "Good question.",
            "tribe": "T" * 101,
        })
        assert resp.status_code == 422

    def test_tribe_specific_prompt_content(self):
        """TRIBE_GUIDANCE contains entries for key PropertyTribes tribes."""
        from moderator import TRIBE_GUIDANCE
        assert "Wanted & Recommendations" in TRIBE_GUIDANCE
        assert "No Money Down (NMD)" in TRIBE_GUIDANCE
        assert "Problem Tenants" in TRIBE_GUIDANCE
        assert "HMO Landlords" in TRIBE_GUIDANCE
        # NMD tribe should flag higher scrutiny
        assert "HIGHER SCRUTINY" in TRIBE_GUIDANCE["No Money Down (NMD)"]
        # Wanted & Recommendations should allow self-promotion
        assert "approve" in TRIBE_GUIDANCE["Wanted & Recommendations"].lower()
