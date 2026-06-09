"""Shared FMP API rate limiter.

FMP has a 3000 req/min cap across the whole account. Realtime ingestion
and NLP article fetching run in separate processes, so a single in-memory
limiter cannot enforce the global budget across them. Instead each
process holds its own :class:`FMPRateLimiter` with a static cap that
together stays under the account limit:

* ``FMPRateLimiter.for_realtime()`` -> 1000 req/min cap
* ``FMPRateLimiter.for_nlp()``      -> 2000 req/min cap

Realtime ingestion currently uses ~1 req/min (single batch quote call),
so the 1000 cap is generous headroom. NLP gets the bulk of the budget
because polling ~1900 tickers every 5 minutes is the dominant consumer.

The limiter is thread-safe within a process. Callers acquire a slot via
:meth:`acquire`; on overflow the slot's timestamp is reserved up front so
concurrent threads see the budget consumed while one of them sleeps.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


REALTIME_CAP_PER_MIN = 1000
NLP_CAP_PER_MIN = 2000
WINDOW_SECONDS = 60


class FMPRateLimiter:
    """Sliding-window rate limiter for FMP HTTP calls.

    Use :meth:`for_realtime` / :meth:`for_nlp` instead of instantiating
    directly so all callers in a process share the same instance.
    """

    _instances: dict[str, "FMPRateLimiter"] = {}
    _instances_lock = threading.Lock()

    def __init__(self,
        max_requests_per_min: int,
        label: str = "fmp"
    ):
        if max_requests_per_min <= 0:
            raise ValueError("max_requests_per_min must be > 0")
        self.max_requests_per_min: int = max_requests_per_min
        self.window_seconds: int = WINDOW_SECONDS
        self.label: str = label
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    @classmethod
    def _shared(cls, key: str, cap: int) -> "FMPRateLimiter":
        with cls._instances_lock:
            existing = cls._instances.get(key)
            if existing is None:
                existing = cls(max_requests_per_min=cap, label=key)
                cls._instances[key] = existing
            return existing

    @classmethod
    def for_realtime(cls) -> "FMPRateLimiter":
        return cls._shared("realtime", REALTIME_CAP_PER_MIN)

    @classmethod
    def for_nlp(cls) -> "FMPRateLimiter":
        return cls._shared("nlp", NLP_CAP_PER_MIN)

    def acquire(self) -> None:
        """Block until one request slot is available, then claim it.

        Sliding-window enforcement: prunes timestamps older than
        ``window_seconds``; sleeps if the live count already meets the
        cap; reserves the slot under the lock so concurrent callers see
        it consumed.
        """
        with self._lock:
            now = time.time()
            self._timestamps = [
                t for t in self._timestamps if (now - t) < self.window_seconds
            ]
            if len(self._timestamps) >= self.max_requests_per_min:
                wait_time = self.window_seconds - (now - self._timestamps[0])
            else:
                wait_time = 0.0
            self._timestamps.append(time.time())

        if wait_time > 0:
            logger.warning(
                "[%s] Hit FMP cap (%s/min). Sleeping %.2fs",
                self.label,
                self.max_requests_per_min,
                wait_time,
            )
            time.sleep(wait_time)

    def usage_in_window(self) -> int:
        """Live count of requests in the current window. Diagnostic only."""
        with self._lock:
            now = time.time()
            self._timestamps = [
                t for t in self._timestamps if (now - t) < self.window_seconds
            ]
            return len(self._timestamps)


__all__ = [
    "FMPRateLimiter",
    "NLP_CAP_PER_MIN",
    "REALTIME_CAP_PER_MIN",
    "WINDOW_SECONDS",
]
