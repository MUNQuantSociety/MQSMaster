"""
src/risk_manager/drawdown_circuit_breaker.py

Drawdown circuit breaker for the master allocator.

A three-state hysteresis machine that throttles or pauses the strategy sleeves
based on rolling drawdown of the master equity curve.

  - ARMED      -> leverage_multiplier = 1.0
  - THROTTLED  -> leverage_multiplier = 0.5
  - PAUSED     -> leverage_multiplier = 0.0

Hard contract (Team B B3 invariant):
  This module does NOT measure realized portfolio volatility. It does NOT
  compute sigma_target / sigma_realized. It does NOT introduce a second
  vol-target layer.

Citations:
  - CPPI cushion concept: Black & Jones (1987); Perold (1986).
  - TIPP floor ratchet: Estep & Kritzman (1988).
  - Linear-in-cushion optimality: Grossman & Zhou, Math Finance 3(3), 1993.
  - Drawdown modulation + restart: Hsieh & Barmish (IFAC 2017); Hsieh (2023).
  - DD-feedback risk aversion: Nystrup, Boyd et al., Ann. Oper. Res. 2019.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


STATE_ARMED = "ARMED"
STATE_THROTTLED = "THROTTLED"
STATE_PAUSED = "PAUSED"

_STATE_LEVERAGE: Dict[str, float] = {
    STATE_ARMED: 1.0,
    STATE_THROTTLED: 0.5,
    STATE_PAUSED: 0.0,
}


@dataclass
class CircuitBreakerConfig:
    enabled: bool = True
    trip_throttle: float = -0.15
    trip_pause: float = -0.25
    rearm_throttle: float = -0.075
    rearm_pause: float = -0.15
    trip_window_days: int = 90
    rearm_window_days: int = 30
    min_dwell_days: int = 10
    min_history_days: int = 30
    state_path: str = "src/risk_manager/.circuit_breaker_state.json"

    @classmethod
    def from_mapping(cls, m: Optional[Mapping]) -> "CircuitBreakerConfig":
        if not m:
            return cls()
        kwargs = {k: m[k] for k in m.keys() if k in cls.__dataclass_fields__}
        return cls(**kwargs)


@dataclass
class CircuitBreakerState:
    state: str = STATE_ARMED
    peak_equity: float = 0.0
    last_transition_ts: Optional[str] = None
    days_in_state: int = 0
    last_seen_ts: Optional[str] = None
    leverage_multiplier: float = 1.0
    paused: bool = False

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "CircuitBreakerState":
        kwargs = {k: d[k] for k in d.keys() if k in cls.__dataclass_fields__}
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str) -> "CircuitBreakerState":
        if not path or not os.path.exists(path):
            return cls()
        try:
            with open(path, "r") as f:
                return cls.from_dict(json.load(f))
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to load CB state from %s; using defaults.", path)
            return cls()

    def save(self, path: str) -> None:
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        except OSError:
            logger.exception("Failed to persist CB state to %s.", path)


def _coerce_equity_series(equity_curve) -> pd.Series:
    if isinstance(equity_curve, pd.Series):
        s = equity_curve.copy()
    elif isinstance(equity_curve, pd.DataFrame):
        cols = {c.lower(): c for c in equity_curve.columns}
        ts_col = cols.get("timestamp") or cols.get("date") or cols.get("ts")
        eq_col = cols.get("equity") or cols.get("notional") or cols.get("value")
        if ts_col is None or eq_col is None:
            raise ValueError(
                "DataFrame equity_curve must have timestamp & equity columns; "
                f"got {list(equity_curve.columns)}"
            )
        s = pd.Series(
            pd.to_numeric(equity_curve[eq_col], errors="coerce").to_numpy(),
            index=pd.to_datetime(equity_curve[ts_col], utc=True, errors="coerce"),
        )
    else:
        ts, eq = zip(*equity_curve)
        s = pd.Series(
            pd.to_numeric(np.asarray(eq), errors="coerce"),
            index=pd.to_datetime(np.asarray(ts), utc=True, errors="coerce"),
        )

    s = s.dropna()
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.index.tz is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    return s


def _rolling_drawdown_min(equity: pd.Series, window_days: int) -> float:
    if equity.empty:
        return 0.0
    end = equity.index[-1]
    start = end - timedelta(days=window_days)
    window = equity.loc[equity.index >= start]
    if len(window) < 2:
        return 0.0
    arr = window.to_numpy(dtype=float)
    arr = np.where(arr <= 0, 1e-9, arr)
    peak = np.maximum.accumulate(arr)
    dd = (arr / peak) - 1.0
    return float(np.min(dd))


def _hwm_drawdown(equity: pd.Series, peak: float) -> float:
    if equity.empty:
        return 0.0
    last = float(equity.iloc[-1])
    if last <= 0:
        return -1.0
    eff_peak = max(peak, last)
    return float((last / eff_peak) - 1.0)


def update_state(
    equity_curve,
    cfg=None,
    *,
    prior_state: Optional[CircuitBreakerState] = None,
    now: Optional[datetime] = None,
    persist: bool = True,
) -> Dict:
    """Advance the circuit breaker state machine by one day."""
    if isinstance(cfg, Mapping) or cfg is None:
        cfg = CircuitBreakerConfig.from_mapping(cfg)

    if not cfg.enabled:
        return {
            "state": STATE_ARMED,
            "leverage_multiplier": 1.0,
            "paused": False,
            "peak_equity": float("nan"),
            "days_in_state": 0,
            "rolling_dd_90d": 0.0,
            "rolling_dd_30d": 0.0,
            "rolling_dd_hwm": 0.0,
            "reason": "disabled",
            "prior_state_obj": prior_state or CircuitBreakerState(),
            "new_state_obj": CircuitBreakerState(),
        }

    eq = _coerce_equity_series(equity_curve)

    if len(eq) < cfg.min_history_days:
        st = prior_state or CircuitBreakerState.load(cfg.state_path)
        return {
            "state": STATE_ARMED,
            "leverage_multiplier": 1.0,
            "paused": False,
            "peak_equity": st.peak_equity,
            "days_in_state": st.days_in_state,
            "rolling_dd_90d": 0.0,
            "rolling_dd_30d": 0.0,
            "rolling_dd_hwm": 0.0,
            "reason": (
                f"insufficient history: {len(eq)} < {cfg.min_history_days}; "
                "breaker idle"
            ),
            "prior_state_obj": st,
            "new_state_obj": st,
        }

    if prior_state is None:
        prior_state = CircuitBreakerState.load(cfg.state_path)

    last_ts = eq.index[-1].to_pydatetime()
    if now is None:
        now = last_ts

    last_equity = float(eq.iloc[-1])
    new_peak = max(prior_state.peak_equity or 0.0, last_equity)

    dd_90 = _rolling_drawdown_min(eq, cfg.trip_window_days)
    dd_30 = _rolling_drawdown_min(eq, cfg.rearm_window_days)
    dd_hwm = _hwm_drawdown(eq, prior_state.peak_equity or 0.0)

    prev_state = prior_state.state if prior_state.state in _STATE_LEVERAGE else STATE_ARMED
    days_in_state = prior_state.days_in_state
    if prior_state.last_transition_ts:
        try:
            prev_ts = datetime.fromisoformat(prior_state.last_transition_ts)
            if prev_ts.tzinfo:
                prev_ts = prev_ts.astimezone(timezone.utc).replace(tzinfo=None)
            days_in_state = max(int((now - prev_ts).days), 0)
        except ValueError:
            pass

    new_state = prev_state
    reason = "no transition"

    if prev_state == STATE_ARMED:
        if dd_90 <= cfg.trip_pause:
            new_state = STATE_THROTTLED
            reason = (
                f"trip ARMED->THROTTLED: dd_90={dd_90:.4f} crossed pause level "
                f"{cfg.trip_pause:.4f}; will reassess next tick"
            )
        elif dd_90 <= cfg.trip_throttle:
            new_state = STATE_THROTTLED
            reason = f"trip ARMED->THROTTLED: dd_90={dd_90:.4f} <= {cfg.trip_throttle:.4f}"
    elif prev_state == STATE_THROTTLED:
        if dd_90 <= cfg.trip_pause:
            new_state = STATE_PAUSED
            reason = f"trip THROTTLED->PAUSED: dd_90={dd_90:.4f} <= {cfg.trip_pause:.4f}"
        elif dd_30 >= cfg.rearm_throttle:
            new_state = STATE_ARMED
            reason = f"rearm THROTTLED->ARMED: dd_30={dd_30:.4f} >= {cfg.rearm_throttle:.4f}"
    elif prev_state == STATE_PAUSED:
        if dd_30 >= cfg.rearm_pause and days_in_state >= cfg.min_dwell_days:
            new_state = STATE_THROTTLED
            reason = (
                f"rearm PAUSED->THROTTLED: dd_30={dd_30:.4f} >= "
                f"{cfg.rearm_pause:.4f} AND days_in_state={days_in_state} "
                f">= {cfg.min_dwell_days}"
            )

    transitioned = new_state != prev_state
    if transitioned:
        days_in_state = 0
        last_transition_ts = now.isoformat()
        logger.warning(
            "[circuit_breaker] %s -> %s | %s | leverage=%.2f peak=%.2f equity=%.2f",
            prev_state, new_state, reason,
            _STATE_LEVERAGE[new_state], new_peak, last_equity,
        )
    else:
        last_transition_ts = prior_state.last_transition_ts
        days_in_state = prior_state.days_in_state + 1

    new_state_obj = CircuitBreakerState(
        state=new_state,
        peak_equity=new_peak,
        last_transition_ts=last_transition_ts or now.isoformat(),
        days_in_state=days_in_state,
        last_seen_ts=now.isoformat(),
        leverage_multiplier=_STATE_LEVERAGE[new_state],
        paused=(new_state == STATE_PAUSED),
    )

    if persist:
        new_state_obj.save(cfg.state_path)

    return {
        "state": new_state,
        "leverage_multiplier": _STATE_LEVERAGE[new_state],
        "paused": new_state == STATE_PAUSED,
        "peak_equity": new_peak,
        "days_in_state": days_in_state,
        "rolling_dd_90d": dd_90,
        "rolling_dd_30d": dd_30,
        "rolling_dd_hwm": dd_hwm,
        "reason": reason,
        "prior_state_obj": prior_state,
        "new_state_obj": new_state_obj,
    }
