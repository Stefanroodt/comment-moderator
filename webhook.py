"""
Webhook delivery for flagged content.

When a comment is flagged for human review, this module fires a POST request
to a configurable WEBHOOK_URL. Delivery is best-effort — a failed webhook
never affects the moderation response returned to the user.

Configure via .env:
    WEBHOOK_URL=https://hooks.example.com/moderation
    (or use https://webhook.site for testing)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

WEBHOOK_TIMEOUT = 5.0  # seconds


def send_flagged_webhook(
    webhook_url: str,
    comment_id: UUID,
    user_id: str,
    comment: str,
    confidence: float,
    reasoning: str,
    timestamp: datetime,
) -> None:
    """
    POST a notification payload to webhook_url.

    Called synchronously but wrapped in try/except so any network failure
    is logged and silently swallowed — it must never break the API response.
    """
    payload: Dict[str, Any] = {
        "event": "content_flagged_for_review",
        "comment_id": str(comment_id),
        "user_id": user_id,
        "comment": comment,
        "confidence": confidence,
        "reasoning": reasoning,
        "timestamp": timestamp.isoformat(),
        "action_required": "Human review needed at GET /log",
    }

    try:
        with httpx.Client(timeout=WEBHOOK_TIMEOUT) as client:
            response = client.post(webhook_url, json=payload)
            response.raise_for_status()
            logger.info(
                "Webhook delivered for comment %s → %s (HTTP %d)",
                comment_id,
                webhook_url,
                response.status_code,
            )
    except httpx.TimeoutException:
        logger.warning("Webhook timed out for comment %s (url=%s)", comment_id, webhook_url)
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Webhook HTTP error for comment %s: %s",
            comment_id,
            exc.response.status_code,
        )
    except Exception as exc:
        logger.warning("Webhook delivery failed for comment %s: %s", comment_id, exc)
