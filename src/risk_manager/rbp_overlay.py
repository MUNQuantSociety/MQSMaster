"""
src/risk_manager/rbp_overlay.py

RBP (Risk-Budgeted Position) conviction overlay.

Sits between portfolio strategies and trade execution. Strategies emit
(ticker, signal_type, confidence) tuples; this overlay blends in the latest
RBP forecast (y_pred) to nudge confidence up when the RBP model agrees with
the strategy's directional call, and down when it disagrees.

Design notes:
  - Stateless except for an in-memory ticker -> (y_pred, fetched_at) cache.
    Cache TTL is short (default 60s) so DB load stays low when many tickers
    trade per minute. NaN sentinels are cached for misses so we don't hammer
    the DB on tickers that have no recent forecast.
  - SAFE: never raises. Any exception (DB outage, schema drift, bad row) is
    swallowed, logged at WARNING, and the original confidence is returned.
    This is a deliberate fail-open contract -- the overlay must never block
    a trade that the underlying strategy already approved.
  - Injected as a callable into tradeExecutor:
        overlay = RBPOverlay(db, cfg)
        executor = tradeExecutor(db, rbp_overlay=overlay)
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, Optional, Tuple

from common.database.MQSDBConnector import MQSDBConnector


class RBPOverlay:
    """Conviction overlay: blends RBP forecast into trade confidence.

    Stateless except for in-memory ticker->signal cache (TTL controlled by caller).
    Safe -- never raises. On any failure, returns the original confidence unchanged.

    Used as a callable injected into tradeExecutor:
        overlay = RBPOverlay(db, cfg)
        executor = tradeExecutor(db, rbp_overlay=overlay)
    """

    # Parameterized INTERVAL: matches the existing repo style in
    # src/risk_manager/daily_allocator.py (~line 178), which uses
    # `INTERVAL '%s days'` with psycopg2 %s substitution. psycopg2 substitutes
    # the value into the string literal before the server parses it.
    LATEST_FORECAST_QUERY = """
        SELECT y_pred, asof
        FROM rbp_forecasts
        WHERE ticker = %s
          AND asof >= NOW() - INTERVAL '%s hours'
        ORDER BY asof DESC, generated_at DESC
        LIMIT 1
    """

    def __init__(self, db: MQSDBConnector, cfg: Dict[str, Any]):
        """cfg keys (with defaults):
            enabled: bool = True
            blend_weight: float = 0.10
            tanh_scale: float = 20.0
            stale_after_hours: int = 24
            disabled_portfolios: List[str] = []
            cache_ttl_seconds: int = 60   # how long to cache a (ticker, y_pred) in memory
        """
        self.db = db
        self.enabled = bool(cfg.get("enabled", True))
        self.blend_weight = float(cfg.get("blend_weight", 0.10))
        self.tanh_scale = float(cfg.get("tanh_scale", 20.0))
        self.stale_after_hours = int(cfg.get("stale_after_hours", 24))
        self.disabled_portfolios = set(
            str(p) for p in cfg.get("disabled_portfolios", [])
        )
        self.cache_ttl = int(cfg.get("cache_ttl_seconds", 60))
        # ticker -> (y_pred, fetched_at_epoch); NaN y_pred = cached miss
        self._cache: Dict[str, Tuple[float, float]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Self-disable if rbp_forecasts table doesn't exist. Prevents per-trade
        # log spam when bot runs without rbp_runner having bootstrapped schema.
        if self.enabled and not self._forecast_table_exists():
            self.logger.warning(
                "rbp_forecasts table not found in DB. Disabling RBP overlay. "
                "Bootstrap schema (SchemaDefinitions.create_all_tables) or start "
                "rbp_runner to enable."
            )
            self.enabled = False

    def _forecast_table_exists(self) -> bool:
        try:
            res = self.db.execute_query(
                "SELECT to_regclass('public.rbp_forecasts') AS t",
                fetch=True,
            )
            if res.get("status") != "success" or not res.get("data"):
                return False
            return res["data"][0].get("t") is not None
        except Exception:
            self.logger.exception("rbp_forecasts existence check failed")
            return False

    def __call__(
        self,
        portfolio_id: str,
        ticker: str,
        signal_type: str,
        confidence: float,
    ) -> float:
        """Returns blended confidence. NEVER raises."""
        try:
            if not self.enabled:
                return confidence
            if str(portfolio_id) in self.disabled_portfolios:
                return confidence
            if signal_type not in ("BUY", "SELL"):
                return confidence

            y_pred = self._get_forecast(ticker)
            if y_pred is None or math.isnan(y_pred):
                return confidence

            # Sign agreement: BUY wants y_pred > 0, SELL wants y_pred < 0
            agree = (y_pred > 0) if signal_type == "BUY" else (y_pred < 0)
            rbp_mag = abs(math.tanh(y_pred * self.tanh_scale))  # 0..1
            rbp_conf = rbp_mag if agree else 0.0

            new_conf = (
                (1.0 - self.blend_weight) * confidence
                + self.blend_weight * rbp_conf
            )
            new_conf = max(0.0, min(1.0, new_conf))

            self.logger.debug(
                "[RBP overlay] portfolio=%s ticker=%s side=%s y_pred=%.4f "
                "agree=%s rbp_conf=%.4f conf=%.3f -> %.3f",
                portfolio_id,
                ticker,
                signal_type,
                y_pred,
                agree,
                rbp_conf,
                confidence,
                new_conf,
            )
            return new_conf
        except Exception as exc:
            self.logger.warning(
                "RBP overlay failed for %s/%s: %s (pass-through)",
                portfolio_id,
                ticker,
                exc,
            )
            return confidence

    def _get_forecast(self, ticker: str) -> Optional[float]:
        """Returns latest y_pred for ticker (in-memory TTL cache, falls back to DB).

        Returns None when no row is found within the staleness window.
        Caches NaN as a sentinel for misses so repeated calls within the
        TTL don't re-hit the DB for tickers with no forecast.
        """
        now = time.time()
        cached = self._cache.get(ticker)
        if cached is not None:
            y_pred, fetched_at = cached
            if now - fetched_at < self.cache_ttl:
                # NaN sentinel means "known miss" -- return None to caller
                if math.isnan(y_pred):
                    return None
                return y_pred

        result = self.db.execute_query(
            self.LATEST_FORECAST_QUERY,
            (ticker, self.stale_after_hours),
            fetch=True,
        )
        if result.get("status") != "success" or not result.get("data"):
            self._cache[ticker] = (float("nan"), now)
            return None
        row = result["data"][0]
        y_pred = float(row["y_pred"])
        self._cache[ticker] = (y_pred, now)
        return y_pred
