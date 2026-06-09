"""
Minimal FMP HTTP helper for one-off batch scripts (universe + fundamentals).

Why this exists:
  src/orchestrator/marketData/fmpMarketData.py::_check_rate_limit currently has
  an unconditional `while True` loop with no `break` path, which hangs any caller
  on the first request. This helper sidesteps it with a simple sleep-based
  rate limiter, verbose per-call logging, and explicit retry/backoff.

Drop-in usage:
    client = FMPClient(logger=logger)
    data = client.get(
        "https://financialmodelingprep.com/stable/sp500-constituent",
        label="S&P 500",
    )
"""

import logging
import os
import time
from typing import Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MIN_INTERVAL_SECONDS = 0.05   # ~1200 req/min, well under the 3000 cap.
DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_MAX_BACKOFF_SECONDS = 60


class FMPClient:
    """Thin wrapper around requests.get with logging, throttling, and retries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_SECONDS,
        timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        logger: Optional[logging.Logger] = None,
    ):
        resolved_key = (api_key or os.environ.get("FMP_API_KEY") or "").strip()
        if not resolved_key:
            raise ValueError(
                "FMP_API_KEY is not set. Add it to .env or export it before running."
            )
        self.api_key = resolved_key
        self.min_interval_s = max(0.0, float(min_interval_s))
        self.timeout_s = float(timeout_s)
        self.max_retries = max(1, int(max_retries))
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._last_call_at = 0.0
        self._call_count = 0

    def _throttle(self):
        if self.min_interval_s <= 0:
            return
        elapsed = time.time() - self._last_call_at
        wait = self.min_interval_s - elapsed
        if wait > 0:
            self.logger.debug("[FMP] Throttle %.3fs", wait)
            time.sleep(wait)

    def _redact(self, params: Optional[dict]) -> dict:
        if not params:
            return {}
        return {k: ("<redacted>" if k == "apikey" else v) for k, v in params.items()}

    def get(
        self,
        url: str,
        params: Optional[dict] = None,
        *,
        label: str = "",
    ) -> Optional[Any]:
        """Execute a GET with retries. Returns parsed JSON, or None if exhausted."""
        merged = dict(params or {})
        merged.setdefault("apikey", self.api_key)
        tag = label or url

        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            self._last_call_at = time.time()
            self._call_count += 1

            self.logger.info(
                "[FMP] GET %s | params=%s | attempt=%s/%s | call#=%s",
                tag,
                self._redact(merged),
                attempt,
                self.max_retries,
                self._call_count,
            )

            try:
                resp = requests.get(url, params=merged, timeout=self.timeout_s)
            except requests.RequestException as e:
                backoff = min(2 ** attempt, DEFAULT_MAX_BACKOFF_SECONDS)
                self.logger.warning(
                    "[FMP] %s network error: %s -> sleeping %ss (attempt %s/%s)",
                    tag,
                    e,
                    backoff,
                    attempt,
                    self.max_retries,
                )
                time.sleep(backoff)
                continue

            status = resp.status_code
            if status == 200:
                try:
                    payload = resp.json()
                except ValueError as e:
                    self.logger.warning(
                        "[FMP] %s 200 but invalid JSON: %s | body[:200]=%s",
                        tag,
                        e,
                        resp.text[:200],
                    )
                    return None
                size = len(payload) if isinstance(payload, (list, dict)) else "?"
                self.logger.info("[FMP] %s 200 OK | size=%s", tag, size)
                return payload

            if status == 429:
                backoff = min(2 ** attempt, DEFAULT_MAX_BACKOFF_SECONDS)
                self.logger.warning(
                    "[FMP] %s 429 rate-limited -> sleeping %ss (attempt %s/%s)",
                    tag,
                    backoff,
                    attempt,
                    self.max_retries,
                )
                time.sleep(backoff)
                continue

            if status in (401, 403):
                self.logger.error(
                    "[FMP] %s auth failure HTTP %s body[:200]=%s",
                    tag,
                    status,
                    resp.text[:200],
                )
                return None

            backoff = min(2 ** attempt, DEFAULT_MAX_BACKOFF_SECONDS)
            self.logger.warning(
                "[FMP] %s HTTP %s body[:200]=%s -> sleeping %ss (attempt %s/%s)",
                tag,
                status,
                resp.text[:200],
                backoff,
                attempt,
                self.max_retries,
            )
            time.sleep(backoff)

        self.logger.error(
            "[FMP] %s exhausted after %s attempts.", tag, self.max_retries
        )
        return None

    def get_call_count(self) -> int:
        return self._call_count
