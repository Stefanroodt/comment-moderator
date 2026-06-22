"""
In-memory moderation log.

Thread-safe via a threading.Lock so the store is safe under FastAPI's
default multi-threaded Uvicorn workers. No persistence across restarts —
as per the spec, a database is not required.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from uuid import UUID

from models import AdminOverrideRequest, FinalDecision, LogEntry, ModerationDecision


class ModerationStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[UUID, LogEntry] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, entry: LogEntry) -> None:
        with self._lock:
            self._entries[entry.comment_id] = entry

    def record_appeal(
        self,
        comment_id: UUID,
        appeal_context: str,
        appeal_decision: FinalDecision,
        appeal_reasoning: str,
    ) -> Optional[LogEntry]:
        """Mutate the existing log entry in place with appeal outcome."""
        with self._lock:
            entry = self._entries.get(comment_id)
            if entry is None:
                return None
            updated = entry.model_copy(
                update={
                    "appealed": True,
                    "appeal_context": appeal_context,
                    "appeal_decision": appeal_decision,
                    "appeal_reasoning": appeal_reasoning,
                    "appeal_timestamp": datetime.now(timezone.utc),
                }
            )
            self._entries[comment_id] = updated
            return updated

    def record_admin_override(
        self,
        comment_id: UUID,
        decision: ModerationDecision,
        note: Optional[str],
    ) -> Optional[LogEntry]:
        """Apply a human moderator override to an existing log entry."""
        with self._lock:
            entry = self._entries.get(comment_id)
            if entry is None:
                return None
            updated = entry.model_copy(
                update={
                    "admin_overridden": True,
                    "admin_decision": decision,
                    "admin_note": note,
                    "admin_timestamp": datetime.now(timezone.utc),
                }
            )
            self._entries[comment_id] = updated
            return updated

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, comment_id: UUID) -> Optional[LogEntry]:
        with self._lock:
            return self._entries.get(comment_id)

    def all(self) -> List[LogEntry]:
        with self._lock:
            return list(self._entries.values())

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def stats(self, since: Optional[datetime] = None) -> dict:
        """Compute aggregate moderation statistics over all stored entries.

        Args:
            since: If provided, only include entries on or after this datetime.
        """
        with self._lock:
            entries = list(self._entries.values())

        if since is not None:
            # Normalise to UTC-aware for safe comparison
            since_utc = since.replace(tzinfo=timezone.utc) if since.tzinfo is None else since
            entries = [e for e in entries if e.timestamp >= since_utc]

        total = len(entries)
        if total == 0:
            return {
                "total_comments": 0,
                "decisions": {"approved": 0, "rejected": 0, "flagged_for_review": 0},
                "decision_percentages": {"approved": 0.0, "rejected": 0.0, "flagged_for_review": 0.0},
                "avg_confidence": None,  # null — no data, not zero confidence
                "top_rejection_categories": {},
                "appeals": {"total": 0, "overturned": 0, "upheld": 0, "overturn_rate": 0.0},
                "admin_overrides": 0,
            }

        decisions: dict = {"approved": 0, "rejected": 0, "flagged_for_review": 0}
        categories: dict = {}
        confidences: list = []
        appeals_total = 0
        appeals_overturned = 0
        admin_overrides = 0

        for entry in entries:
            decisions[entry.decision.value] += 1
            confidences.append(entry.confidence)

            if entry.rejection_category.value != "none":
                cat = entry.rejection_category.value
                categories[cat] = categories.get(cat, 0) + 1

            if entry.appealed:
                appeals_total += 1
                if entry.appeal_decision and entry.appeal_decision.value == "approved":
                    appeals_overturned += 1

            if entry.admin_overridden:
                admin_overrides += 1

        overturn_rate = round(appeals_overturned / appeals_total, 3) if appeals_total > 0 else 0.0

        # Cap at top 5 — beyond that the signal-to-noise ratio drops significantly
        top_categories = dict(
            sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
        )

        return {
            "total_comments": total,
            "decisions": decisions,
            "decision_percentages": {k: round(v / total * 100, 1) for k, v in decisions.items()},
            "avg_confidence": round(sum(confidences) / len(confidences), 3),
            "top_rejection_categories": top_categories,
            "appeals": {
                "total": appeals_total,
                "overturned": appeals_overturned,
                "upheld": appeals_total - appeals_overturned,
                "overturn_rate": overturn_rate,
            },
            "admin_overrides": admin_overrides,
        }


# Singleton — imported and used directly by main.py and routes.
store = ModerationStore()
