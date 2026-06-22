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


# Singleton — imported and used directly by main.py and routes.
store = ModerationStore()
