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
