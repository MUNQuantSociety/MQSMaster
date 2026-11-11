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
from dataclasses import asdict
from typing import Literal

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
class CircuitState:
    def __init__(self, name: str, leverage_multiplier: float):
        self.name = name
        self.leverage_multiplier = leverage_multiplier


class States:
    ARMED = CircuitState("ARMED", 1.0)
    THROTTLED = CircuitState("ARMED", 0.5)
    PAUSED = CircuitState("ARMED", 0.0)


type ValidState = Literal["ARMED", "THROTTLED", "PAUSED"]


@dataclass
class CircuitBreakerState:
    state: CircuitState = States.ARMED
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


@dataclass
class UpdateResult:
    def __init__(
        self,
        state_name: str,
        leverage_mult: float,
        peak_equity: float = float("nan"),
        days_in_state: int = 0,
        rolling_dd_90d: float = 0.0,
        rolling_dd_30d: float = 0.0,
        rolling_dd_hwm: float = 0.0,
        reason: str = "",
        prior_state: CircuitBreakerState = CircuitBreakerState(),
        new_state: CircuitBreakerState = CircuitBreakerState(),
    ):
        self.state: str = state_name
        self.leverage_multiplier: float = leverage_mult
        self.paused: bool = state_name == States.PAUSED.name
        self.peak_equity: float = peak_equity
        self.days_in_state: int = days_in_state
        self.rolling_dd_90d: float = rolling_dd_90d
        self.rolling_dd_30d: float = rolling_dd_30d
        self.rolling_dd_hwm: float = rolling_dd_hwm
        self.reason: str = reason
        self.prior_state_obj: CircuitBreakerState = prior_state
        self.new_state_obj: CircuitBreakerState = new_state


def _change_state(
    previous_state: str,
    days_in_state: float,
    drawdown_90: float,
    drawdown_30: float,
    config: CircuitBreakerConfig,
) -> tuple[CircuitState, str]:
    """Change the state and reason. Returns (new_state, reason)."""
    new_state: CircuitState = States.ARMED
    reason: str = ""
    if previous_state == States.ARMED.name and drawdown_90 <= config.trip_pause:
        new_state = States.THROTTLED
        reason = (
            f"trip ARMED->THROTTLED: dd_90={drawdown_90:.4f} crossed pause level "
            f"{config.trip_pause:.4f}; will reassess next tick"
        )

    elif previous_state == States.ARMED.name and drawdown_90 <= config.trip_throttle:
        new_state = States.THROTTLED
        reason = (
            f"trip ARMED->THROTTLED: dd_90={drawdown_90} <= {config.trip_throttle:.4f}"
        )

    elif previous_state == States.THROTTLED.name and drawdown_90 <= config.trip_pause:
        new_state = States.PAUSED
        reason = f"trip THROTTLED->PAUSED: dd_90={drawdown_90:.4f} <= {config.trip_pause:.4f}"

    elif (
        previous_state == States.THROTTLED.name and drawdown_30 >= config.rearm_throttle
    ):
        new_state = States.ARMED.name
        reason = f"rearm THROTTLED->ARMED: dd_30={drawdown_30:.4f} >= {config.rearm_throttle:.4f}"

    elif (
        previous_state == States.PAUSED.name
        and drawdown_30 >= config.rearm_pause
        and days_in_state >= config.min_dwell_days
    ):
        new_state = States.THROTTLED.name
        reason = (
            f"rearm PAUSED->THROTTLED: dd_30={drawdown_30:.4f} >= "
            f"{config.rearm_pause:.4f} AND days_in_state={days_in_state} "
            f">= {config.min_dwell_days}"
        )

    return (new_state, reason)


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

    result = UpdateResult(
        States.ARMED.name,
        States.ARMED.leverage_multiplier,
        float("nan"),
        0,
        0.0,
        0.0,
        0.0,
        "disabled",
        prior_state or CircuitBreakerState(),
        CircuitBreakerState(),
    )

    if not cfg.enabled:
        return asdict(result)

    eq = _coerce_equity_series(equity_curve)

    if len(eq) < cfg.min_history_days:
        st = prior_state or CircuitBreakerState.load(cfg.state_path)
        result.peak_equity = st.peak_equity
        result.days_in_state = st.days_in_state
        result.reason = (
            f"insufficient history: {len(eq)} < {cfg.min_history_days}; " "breaker idle"
        )
        result.prior_state_obj = st
        result.new_state_obj = st
        return asdict(result)

    result.prior_state_obj = prior_state or CircuitBreakerState.load(cfg.state_path)
    now = now or eq.index[-1].to_pydatetime()

    last_equity = float(eq.iloc[-1])
    result.peak_equity = max(prior_state.peak_equity or 0.0, last_equity)

    result.rolling_dd_90d = _rolling_drawdown_min(eq, cfg.trip_window_days)
    result.rolling_dd_30d = _rolling_drawdown_min(eq, cfg.rearm_window_days)
    result.rolling_dd_hwm = _hwm_drawdown(eq, prior_state.peak_equity or 0.0)

    prev_state: CircuitState = (
        prior_state.state if prior_state.state.name in ValidState else States.ARMED
    )
    result.days_in_state = prior_state.days_in_state
    if prior_state.last_transition_ts:
        try:
            prev_ts = datetime.fromisoformat(prior_state.last_transition_ts)
            if prev_ts.tzinfo:
                prev_ts = prev_ts.astimezone(timezone.utc).replace(tzinfo=None)
            result.days_in_state = max(int((now - prev_ts).days), 0)
        except ValueError:
            pass

    new_state: CircuitState = prev_state

    (new_state, result.reason) = _change_state(
        prev_state,
        result.days_in_state,
        result.rolling_dd_90d,
        result.rolling_dd_30d,
        cfg,
    )

    if new_state != prev_state:
        result.days_in_state = 0
        last_transition_ts = now.isoformat()
        logger.warning(
            "[circuit_breaker] %s -> %s | %s | leverage=%.2f peak=%.2f equity=%.2f",
            prev_state,
            new_state,
            result.reason,
            new_state.leverage_multiplier,
            result.peak_equity,
            last_equity,
        )
    else:
        last_transition_ts = prior_state.last_transition_ts
        result.days_in_state = prior_state.days_in_state + 1

    result.new_state_obj = CircuitBreakerState(
        new_state,
        result.peak_equity,
        last_transition_ts or now.isoformat(),
        result.days_in_state,
        now.isoformat(),
        new_state.leverage_multiplier,
        (new_state == States.PAUSED),
    )

    if persist:
        result.new_state_obj.save(cfg.state_path)

    result.paused = result.state == States.PAUSED.name
    return {
        "state": new_state,
        "leverage_multiplier": new_state.leverage_multiplier,
        "paused": new_state == States.PAUSED,
    }
