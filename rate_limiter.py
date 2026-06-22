"""
Per-user rate limiting using a sliding window algorithm.

Keyed on user_id (from the request body) rather than IP address, which is
more accurate for APIs where multiple users share an IP (e.g. corporate NAT).

Usage:
    from rate_limiter import moderate_limiter, appeal_limiter

    if not moderate_limiter.is_allowed(user_id):
        raise HTTPException(429, "Rate limit exceeded")
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List


class SlidingWindowRateLimiter:
    """
    Thread-safe sliding window rate limiter.

    Tracks per-user request timestamps within a rolling time window.
    Automatically evicts stale entries to keep memory bounded.
    """

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        # user_id -> list of request timestamps (as floats)
        self._windows: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, user_id: str) -> bool:
        """
        Returns True if the user is within their rate limit, False if exceeded.
        Records the request if allowed.
        """
        now = datetime.now(timezone.utc).timestamp()
        cutoff = now - self.window_seconds

        with self._lock:
            # Drop timestamps outside the current window
            self._windows[user_id] = [
                t for t in self._windows[user_id] if t > cutoff
            ]

            if len(self._windows[user_id]) >= self.max_requests:
                return False

            self._windows[user_id].append(now)
            return True

    def remaining(self, user_id: str) -> int:
        """How many requests the user has left in the current window."""
        now = datetime.now(timezone.utc).timestamp()
        cutoff = now - self.window_seconds
        with self._lock:
            active = [t for t in self._windows[user_id] if t > cutoff]
            return max(0, self.max_requests - len(active))

    def reset(self, user_id: str) -> None:
        """Clear rate limit state for a user (useful in tests)."""
        with self._lock:
            self._windows.pop(user_id, None)


# ---------------------------------------------------------------------------
# Shared limiter instances
# ---------------------------------------------------------------------------

# 30 moderation requests per user per minute
moderate_limiter = SlidingWindowRateLimiter(max_requests=30, window_seconds=60)

# 5 appeal requests per user per 10 minutes (appeals should be rare)
appeal_limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=600)
