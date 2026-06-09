# C3 — Drawdown Circuit Breaker (P6 / P7 / P8 + Master Allocator)

**Author:** Team C, Risk Layer
**Date:** 2026-05-20
**Mandate:** Read-only audit + design. Single deliverable file. No source-tree edits.
**Constraint inherited from Team B B3:** the breaker **MUST NOT** measure σ_target / σ_realized.
It can throttle leverage or pause a sleeve. It cannot add a second vol-target layer.

---

## 1. Executive summary

MQSMaster currently has **no equity-drawdown circuit breaker**. The string "circuit breaker"
in `src/live_trading/engine.py` refers to an *exception-count* breaker (5 consecutive Python
exceptions → kill thread) that has nothing to do with P&L or drawdown. `DailyAllocator` re-balances
to fixed `portfolio_weights` (currently `{"1": 0.10, "2": 0.90}`) every day with no awareness of
losses. P6 already has sleeve-level vol-targeting (`screener.vol_target_scale`) and a single
`MAX_LEVERAGE=1.5` cap; per Team B B3 those are the **only** vol-target lever in the system.

We propose a **hybrid CPPI/TIPP-style throttle** that runs on the master equity curve
(sum of `cash_equity_book` + mark-to-market positions across portfolios) once per day, **inside the
`DailyAllocator.run_allocation` loop, before the target-weight transfers fire**. The breaker reads
recent equity values, computes drawdown vs a peak that ratchets up (TIPP-style), and emits two
outputs: `leverage_multiplier ∈ [0, 1]` and `paused: bool`. `daily_allocator` then scales each
sleeve's target weight by `leverage_multiplier`; on `paused=True` the master flat-cashes the strategy
sleeves (target=0) until the recovery clause re-arms. This is *not* a second vol-target — there is
no σ_target / σ_realized term anywhere in the breaker.

**Concrete rule (cross-validated against ≥3 sources, see §4):** trip to `0.5×` at 90-day rolling
DD ≤ −15%; trip to `0.0×` (paused) at 90-day rolling DD ≤ −25%; re-arm `0.5×→1.0×` once 30-day
rolling DD ≥ −7.5%; re-arm `paused→0.5×` once 30-day rolling DD ≥ −15% AND ≥10 trading days
have elapsed since pause (hysteresis + dwell). Single feature-flag rollback.

---

## 2. Sources (≥10, cross-validated)

| # | URL | Annotation | Used for |
|---|---|---|---|
| S1 | https://en.wikipedia.org/wiki/Constant_proportion_portfolio_insurance | CPPI: E = m·(V−F); m = 1/crash; m typically 4–5 | Mechanism choice, m-derivation |
| S2 | https://alpaca.markets/learn/cppi-1 | CPPI worked example, m=3 default, GAP risk = 1/m | Validates m-to-threshold mapping (S1 cross-check) |
| S3 | https://paperswithbacktest.com/wiki/introduction-to-cppi-constant-proportion-portfolio-insurance (404 in fetch — alpaca + wiki cross-cover) | Reserved; replaced by S2 + S1 cross-validation | — |
| S3' | https://quantpedia.com/introduction-to-cppi-constant-proportion-portfolio-insurance/ | "If max DD set to 0%, method protects every new max" — TIPP ratchet | TIPP floor ratchet, monthly reb. cadence |
| S4 | https://quantpedia.com/time-invariant-portfolio-protection/ | TIPP floor ratchet F(t)=max(F(t−1), p·V(t)); multiplier 2–4 range "considerably benefits monitoring"; m>5 impaired by costs | TIPP floor design, low-m guidance |
| S5 | https://www.cis.upenn.edu/~mkearns/finread/drawdown.pdf (binary unreadable — Wiley abstract used instead) | Grossman-Zhou problem statement (replacement S5' below) | — |
| S5' | https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9965.1993.tb00044.x → https://www.maths.ox.ac.uk/node/9253 | Grossman-Zhou 1993 *Mathematical Finance* 3(3) 241–276: optimal policy = constant proportion of (W − αM), where M=running max. Foundational | Validates linear-in-cushion exposure rule |
| S6 | https://arxiv.org/abs/1710.01503 | Hsieh & Barmish 2017 IFAC: "Drawdown Modulation Lemma" — closed-form rule that guarantees `max DD ≤ d_max` w.p.1 by scaling exposure as f(current DD) | Justifies our continuous throttle + hard-pause regime |
| S7 | https://arxiv.org/abs/2303.02613 | Hsieh 2023: drawdown modulation + **restart mechanism**. Without restart, policy collapses to a permanent stop-loss when limit approached | Justifies re-arm clause (avoid permanent kill) |
| S8 | https://arxiv.org/abs/1609.00869 | Zambelli 2016 *Determining Optimal Stop-Loss Thresholds via Bayesian Analysis of Drawdown Distributions*: stops "seldom constructed systematically", advocates threshold via DD distribution | Cites threshold selection methodology |
| S9 | https://arxiv.org/html/2402.05272v2 | Shu, Yu & Mulvey 2024 (Princeton): Statistical Jump Model regime detection. **SPY 1990–2023: JM strategy max DD −26.6% vs buy-and-hold −55.2% (52% DD reduction); Sharpe 0.68 vs 0.48.** Jump-penalty λ controls regime persistence (dwell-time analogue) | Empirical evidence that regime-style throttle halves DD; supports hysteresis |
| S10 | https://volatilitybox.com/research/volatility-regimes-explained/ | 4-regime VIX framework (Low / Normal / Elevated 20–30 / Crisis >30); exposure 100/100/50–75/25–50%; **5-day confirmation rule** before regime switch | Hysteresis dwell-time precedent (5-trading-day confirmation) |
| S11 | https://qoppac.blogspot.com/2016/06/capital-correction-pysystemtrade.html | Rob Carver: pysystemtrade "capital multiplier" half-compounding rule; losses reduce notional capital but gains capped at HWM → equivalent to a hedge fund with 100% performance fee | Validates HWM-ratchet floor (= TIPP) for systematic strategies |
| S12 | https://navnoorbawa.substack.com/p/bridgewaters-pure-alpha-returned | Bridgewater Pure Alpha: 12% target vol, real-life **max DD ~20% over 34-year history** (early 2020). Firm-level target: avoid DD > 1/3 (=33%). Documented in Dalio's *Principles* | Real-world institutional threshold anchors |
| S13 | https://www.quantifiedstrategies.com/cta-trading-strategy/ (text via search snippet) + https://groww.in/blog/manage-drawdowns-like-a-hedge-fund-manager | CTA practice: "hard stop at fixed threshold (e.g. 10% total portfolio DD) and reducing position sizes or pausing trading when approaching it"; "automated stop-loss mechanisms typically triggered at 2–5% daily losses and max DD controls ranging from 8–15% depending on strategy vol target" | Industry-band cross-check for our −15% trip and −25% pause |
| S14 | https://en.wikipedia.org/wiki/Maximum_drawdown | Definition `MDD(T) = max_{τ}[max_{t≤τ} X(t) − X(τ)]` used directly in our code | Formal definition of rolling-window DD |
| S15 | https://web.stanford.edu/~boyd/papers/pdf/multiperiod_portfolio_drawdown.pdf (PDF binary — abstract via search) → Nystrup, Boyd et al., *Annals of Operations Research* 282(1), 2019 | "Adjusting risk aversion based on **realized drawdown** controls drawdowns with little or no sacrifice of mean-variance efficiency" | Modern academic blessing of DD-feedback (vs. naive constant-leverage) |

**Cross-validation map (≥2 sources per claim):**

| Claim | Sources |
|---|---|
| CPPI exposure = m·(V−F) | S1 + S2 |
| Multiplier choice m≈3–5 ↔ 1/m crash protection | S1 + S2 + S4 |
| TIPP floor ratchet (HWM-anchored) outperforms fixed floor | S4 + S11 |
| Linear throttle in cushion is **provably** optimal | S5' Grossman-Zhou + S6 Hsieh-Barmish |
| Restart mechanism needed to avoid permanent kill | S7 + S11 |
| DD-feedback risk aversion preserves mean-variance | S15 + S9 |
| −10% to −15% is the standard CTA throttle band | S13 + S12 (Bridgewater max 20%, firm-cap 33%) |
| Hysteresis / dwell-time required (avoid daily flip) | S10 (5-day VIX confirm) + S9 (jump penalty λ) + S7 (restart) |

---

## 3. Current-state analysis (file:line)

### 3.1 What exists today

| Concern | Location | Verdict |
|---|---|---|
| Exception-count breaker (named "circuit breaker") | `src/live_trading/engine.py:32, 40, 84, 117, 123–128` | Operational only — counts Python exceptions in the per-portfolio thread loop, trips at 5 consecutive failures, kills that thread. **Does not look at P&L.** |
| OMS-level monitor | `src/oms/monitor.py` | **0 bytes** (empty stub). Documented gap. Was flagged in B4 too. |
| Drawdown computation | `src/backtest/reporting.py:16–27` (`_compute_max_drawdown`) and `src/backtest/vectorized_backtest.py:152, 176, 192, 464` | Implemented **only for backtest reporting**, not consumed by live or by allocator. |
| Master allocator | `src/risk_manager/daily_allocator.py:157–210` (`run_allocation`) | Reads current cash + mark-to-market positions, multiplies by static `portfolio_weights`, executes transfers. **No DD awareness, no leverage scaling hook.** |
| Capital manager | `src/risk_manager/manage_capital.py` | CLI for ADD/WITHDRAW funding. Not a runtime control. |
| Sleeve-level vol target | `src/portfolios/portfolio_6/screener.py:115–129` (`vol_target_scale`) | Per Team B B3: **this is the sole vol-target layer** and the breaker must not duplicate it. |
| Master config | `src/portfolios/portfolio_manager_config.json` | `{"master_portfolio_id":"0","currency":"USD","portfolio_weights":{"1":0.10,"2":0.90}}` — no breaker config. |
| `live-trading-flow.md` | `docs/workflows/live-trading-flow.md:34, 93–96` | The doc's "Circuit breaker" section refers exclusively to the exception-count breaker. **No P&L circuit-breaker section exists.** ← documented gap. |
| P6 / P7 / P8 strategies | `src/portfolios/portfolio_6/strategy.py:296–342` issues per-ticker target weights via `executor.execute_trade(..., ticker_weight=...)`. P7, P8 use the same `BasePortfolio.OnData` contract (`src/portfolios/portfolio_BASE/strategy.py:336–348`). | Breaker hooks **above** these — at the master allocator, by scaling the sleeve's effective weight, not by editing strategies. |

### 3.2 Gap summary

1. **No equity-curve series** is read by anything in `risk_manager/`. The breaker has to fetch it.
2. **No leverage override** path exists in `daily_allocator.run_allocation`. Has to be added.
3. **No persisted state** for a kill-switch state machine. Has to be added (JSON sidecar by default, optional DB table).
4. The string "circuit breaker" in code and docs is overloaded — the new breaker must use a distinct name (`drawdown_circuit_breaker`) to avoid confusion with the exception breaker.

---

## 4. Mechanism choice + thresholds with citations

### 4.1 Choice of mechanism

Three families surveyed:

| Family | Pro | Con | Verdict |
|---|---|---|---|
| **CPPI / TIPP** (S1, S2, S4, S5') | Closed-form linear policy `E=m·(V−F)`; provably optimal under DD constraint (S5'); HWM ratchet (S4, S11) matches systematic-trading practice | Naive m=1/d_max gives bang-bang behavior; multipliers ≥5 known to underperform under jumps/costs (S4) | **Adopt the *cushion* concept and HWM ratchet, but use a discrete step throttle instead of continuous m·(V−F)** — discrete is more stable under daily polling. |
| **Vol-regime kill-switch** (S9, S10) | Strong empirical record (52% DD reduction on SPY 1990–2023, S9) | **Forbidden by Team B B3** — measuring σ_realized to drive exposure = second vol-target layer | **Reject as the *primary* trigger.** Optional advisory only. |
| **Threshold DD throttle** (S6, S7, S13, S15) | Simple, auditable, matches CTA industry practice (S13); supported by modern academic work (S15) | Pure threshold = bang-bang ⇒ daily flip-flop without hysteresis | **Adopt with hysteresis + dwell.** |

**Decision (D1):** Adopt **threshold-DD throttle with TIPP-style HWM ratchet and two-stage hysteresis**.
The TIPP ratchet is the "floor" concept; the threshold steps `−15% / −25%` and recovery steps
`−7.5% / −15%` give us a CPPI-style cushion mapping without a literal `m·(V−F)` exposure formula
(which would interact awkwardly with sleeve weights). Dwell-time + asymmetric trip/re-arm
thresholds = the hysteresis state machine. **No σ-ratio anywhere.**

### 4.2 Why these specific thresholds

Pick the trip thresholds from the literature band, then validate via cross-source:

- **Trip to 0.5× at DD ≤ −15%.** Bridgewater (S12) targets that the firm "did not want to have a
  drawdown greater than one-third"; their realized max in 34 years is ~20% (early 2020). CTA
  industry practice (S13) places hard caps at 8–15% depending on vol target. P6 runs vol target
  13% ann — at 13% vol a 1-sigma annual DD is ~13%, and 1.15·σ_ann is right at our −15% trip,
  i.e. we throttle when realized DD has already exceeded ~1σ. This matches Carver's capital-correction
  half-compounding (S11) qualitatively (penalize losses, no naive symmetric upside).
- **Trip to 0.0× (pause) at DD ≤ −25%.** Below the Bridgewater 1/3 limit (S12), above CTA hard
  caps (S13). Equivalent to a CPPI `m=4` ⇒ insures against a 25% gap (S1).
- **Re-arm 0.5× → 1.0× at 30-day rolling DD ≥ −7.5%.** Half of the trip threshold. The asymmetry
  (re-arm tighter than trip) is hysteresis. Borrowed from S10's 5-day confirmation rule philosophy:
  proof of regime stability *before* size-up. Halving is also the natural inverse of the 50% throttle.
- **Re-arm pause → 0.5× at 30-day rolling DD ≥ −15% AND ≥10 days dwell.** S7 (Hsieh 2023) shows
  restart mechanism beats permanent stop; S10 uses 5-day VIX confirm; S9 uses jump penalty λ
  giving <1 regime change/year at λ=100. 10-day dwell = compromise: short enough to capture rebounds,
  long enough to avoid a single up-day flipping us back into full risk.
- **Rolling-window choice — 90 day trip / 30 day re-arm.** 90 days = quarterly horizon, standard
  for institutional risk reporting (S12); long enough to be robust to single-day noise, short enough
  to react to drawdowns within a typical recession. 30 days for re-arm because asymmetry (faster
  to confirm health than to confirm distress) is the same hysteresis logic as S10's
  exposure-up-quicker / exposure-down-slower rule.

### 4.3 What the breaker does NOT do (B3 invariant guard)

- **No `sigma_target / sigma_realized` calculation.** The only ratios consumed are
  `equity / peak_equity` (drawdown) and `days_since_state_change` (dwell).
- **No vol forecasting.** No EWMA, no GARCH, no rolling σ.
- **Does not override the P6 sleeve's `vol_target_scale`.** That stays the source of truth for
  intra-sleeve risk; the breaker only scales the inter-sleeve master weight.

---

## 5. State machine

```
                        equity_curve (daily) ──┐
                                               │
                                               ▼
                                       ┌───────────────┐
                                       │  ARMED (1.0×) │ ◄────────────────┐
                                       └──────┬────────┘                  │
                                              │                           │
                                  90d DD ≤ −15%│         30d DD ≥ −7.5%   │
                                              │         AND state=THROTTLE│
                                              ▼                           │
                                       ┌──────────────────┐               │
                                       │ THROTTLED (0.5×) ├───────────────┘
                                       └──────┬───────────┘
                                              │
                                  90d DD ≤ −25%│
                                              │
                                              ▼
                                       ┌──────────────┐
                                       │ PAUSED (0.0×)│
                                       └──────┬───────┘
                                              │
                          30d DD ≥ −15% AND   │
                          days_in_state ≥ 10  │
                                              ▼
                                  (re-arm one step) → THROTTLED → ARMED
```

**Invariants:**
- Trip is **always one step down** (ARMED→THROTTLED→PAUSED) — never ARMED→PAUSED in one tick.
- Re-arm is **always one step up** (PAUSED→THROTTLED→ARMED) — never PAUSED→ARMED in one tick.
- Each state transition resets `days_in_state` to 0.
- Trip uses 90-day window; re-arm uses 30-day window (deliberate asymmetry).
- Re-arm additionally requires `days_in_state ≥ 10` to prevent same-day flip-flop on a noisy bar.

---

## 6. Full source — `src/risk_manager/drawdown_circuit_breaker.py` (apply-ready)

> **Apply path:** new file at `/Users/abhinav/Desktop/MQSMaster/src/risk_manager/drawdown_circuit_breaker.py`.
> Pure-Python stdlib + numpy + pandas. No DB access (caller supplies equity curve). No external API.

```python
"""
src/risk_manager/drawdown_circuit_breaker.py

Drawdown circuit breaker for the master allocator.

A three-state hysteresis machine that throttles or pauses the strategy sleeves
based on rolling drawdown of the master equity curve.

  - ARMED      → leverage_multiplier = 1.0
  - THROTTLED  → leverage_multiplier = 0.5
  - PAUSED     → leverage_multiplier = 0.0

Hard contract (Team B B3 invariant):
  This module does NOT measure realized portfolio volatility. It does NOT
  compute sigma_target / sigma_realized. It does NOT introduce a second
  vol-target layer. It only reads (timestamp, equity) pairs and reports a
  leverage multiplier and a paused flag based on drawdown and dwell-time.

Citations (see C3 deliverable §4 for full annotations):
  - CPPI cushion concept: Black & Jones (1987); Perold (1986).
  - TIPP floor ratchet: Estep & Kritzman (1988).
  - Linear-in-cushion optimality: Grossman & Zhou, Mathematical Finance 3(3),
    1993.
  - Drawdown modulation + restart: Hsieh & Barmish (IFAC 2017); Hsieh (2023).
  - DD-feedback risk aversion: Nystrup, Boyd et al., Ann. Oper. Res. 2019.
  - Hysteresis precedent (5-day confirmation): VolatilityBox 4-regime VIX
    framework; Shu, Yu & Mulvey (2024) jump-penalty lambda.

Default thresholds (overridable via cfg dict):
  trip_throttle  =  -0.15   # 90d DD <= -15%  → THROTTLED (0.5x)
  trip_pause     =  -0.25   # 90d DD <= -25%  → PAUSED    (0.0x)
  rearm_throttle =  -0.075  # 30d DD >= -7.5% → THROTTLED → ARMED
  rearm_pause    =  -0.15   # 30d DD >= -15%  → PAUSED → THROTTLED
  trip_window_days   = 90
  rearm_window_days  = 30
  min_dwell_days     = 10   # re-arm only after this many days in state
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

STATE_ARMED = "ARMED"
STATE_THROTTLED = "THROTTLED"
STATE_PAUSED = "PAUSED"

# Mapping from state name to leverage multiplier emitted to the allocator.
_STATE_LEVERAGE: Dict[str, float] = {
    STATE_ARMED: 1.0,
    STATE_THROTTLED: 0.5,
    STATE_PAUSED: 0.0,
}


@dataclass
class CircuitBreakerConfig:
    """Configuration block. Mirrors the JSON key 'circuit_breaker'."""

    enabled: bool = True
    trip_throttle: float = -0.15
    trip_pause: float = -0.25
    rearm_throttle: float = -0.075
    rearm_pause: float = -0.15
    trip_window_days: int = 90
    rearm_window_days: int = 30
    min_dwell_days: int = 10
    # Numerical guard: do not trip on a curve shorter than this many points.
    min_history_days: int = 30
    # State persistence (relative paths are joined onto cfg_dir at call site).
    state_path: str = "src/risk_manager/.circuit_breaker_state.json"

    @classmethod
    def from_mapping(cls, m: Optional[Mapping]) -> "CircuitBreakerConfig":
        if not m:
            return cls()
        kwargs = {k: m[k] for k in m.keys() if k in cls.__dataclass_fields__}
        return cls(**kwargs)


@dataclass
class CircuitBreakerState:
    """Persisted state. Plain JSON on disk."""

    state: str = STATE_ARMED
    peak_equity: float = 0.0
    last_transition_ts: Optional[str] = None  # ISO-8601
    days_in_state: int = 0
    last_seen_ts: Optional[str] = None        # ISO-8601 of last update call
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


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _coerce_equity_series(equity_curve) -> pd.Series:
    """
    Accepts pd.Series indexed by timestamp, pd.DataFrame with ['timestamp',
    'equity'] columns, or a Sequence[Tuple[ts, equity]]. Returns a sorted,
    deduplicated pd.Series indexed by tz-naive (UTC) timestamps.
    """
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
        # Sequence of (ts, equity)
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
    """
    Return the *minimum* (most negative) drawdown observed inside the trailing
    `window_days` calendar days, where each point's DD is
    `(equity_t / running_peak_t) - 1`.

    "Running peak" here is the running peak *within the window*, which is the
    standard rolling-DD definition (e.g. _compute_max_drawdown in
    src/backtest/reporting.py uses the same construction over the full sample).
    """
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
    """
    Drawdown from a persisted high-water mark (TIPP floor ratchet).
    Returns (equity_now / max(peak, equity_now)) - 1, which is <= 0.
    """
    if equity.empty:
        return 0.0
    last = float(equity.iloc[-1])
    if last <= 0:
        return -1.0
    eff_peak = max(peak, last)
    return float((last / eff_peak) - 1.0)


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def update_state(
    equity_curve,
    cfg: CircuitBreakerConfig | Mapping | None = None,
    *,
    prior_state: Optional[CircuitBreakerState] = None,
    now: Optional[datetime] = None,
    persist: bool = True,
) -> Dict:
    """
    Advance the circuit breaker state machine by one day.

    Parameters
    ----------
    equity_curve : pd.Series | pd.DataFrame | Sequence[(ts, equity)]
        The master equity curve. Must contain at least `cfg.min_history_days`
        observations or the breaker no-ops (returns ARMED, 1.0x).
    cfg : CircuitBreakerConfig | Mapping | None
        Configuration. If a Mapping, parsed via `CircuitBreakerConfig.from_mapping`.
    prior_state : CircuitBreakerState | None
        Optional in-memory prior state. If None, loaded from `cfg.state_path`.
    now : datetime | None
        Current "wall-clock" instant. Defaults to the last timestamp of the
        equity curve (recommended). Used to compute `days_in_state`.
    persist : bool
        If True, writes the new state back to `cfg.state_path`. Set False in
        backtest harnesses / unit tests.

    Returns
    -------
    dict with keys:
      - state              (str)   : current state name
      - leverage_multiplier (float) : 1.0 / 0.5 / 0.0
      - paused             (bool)  : True iff state == PAUSED
      - peak_equity        (float) : the running HWM after this update
      - days_in_state      (int)
      - rolling_dd_90d     (float)
      - rolling_dd_30d     (float)
      - reason             (str)   : human-readable transition explanation
      - prior_state_obj    (CircuitBreakerState)
      - new_state_obj      (CircuitBreakerState)

    The allocator should multiply each sleeve weight by `leverage_multiplier`
    BEFORE computing the cash adjustment. When `paused`, the allocator should
    flat-cash all strategy sleeves to portfolio_id=0 (master). The breaker
    itself never touches the database.
    """
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
            "reason": "disabled",
            "prior_state_obj": prior_state or CircuitBreakerState(),
            "new_state_obj": CircuitBreakerState(),
        }

    eq = _coerce_equity_series(equity_curve)

    if len(eq) < cfg.min_history_days:
        # Not enough history — stay ARMED, do not persist a peak.
        st = prior_state or CircuitBreakerState.load(cfg.state_path)
        return {
            "state": STATE_ARMED,
            "leverage_multiplier": 1.0,
            "paused": False,
            "peak_equity": st.peak_equity,
            "days_in_state": st.days_in_state,
            "rolling_dd_90d": 0.0,
            "rolling_dd_30d": 0.0,
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
    # Also expose the HWM-ratchet DD for logging / falsification.
    dd_hwm = _hwm_drawdown(eq, prior_state.peak_equity or 0.0)

    # Resolve dwell-time relative to last_transition_ts.
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

    # ---- Transition rules ---------------------------------------------------
    # Trips: only one step down per tick.
    if prev_state == STATE_ARMED:
        if dd_90 <= cfg.trip_pause:
            # Defensive: a brutal drop could exceed pause level while ARMED.
            # We still step only once per tick → ARMED → THROTTLED first.
            new_state = STATE_THROTTLED
            reason = (
                f"trip ARMED→THROTTLED: dd_90={dd_90:.4f} crossed pause level "
                f"{cfg.trip_pause:.4f}; will reassess next tick"
            )
        elif dd_90 <= cfg.trip_throttle:
            new_state = STATE_THROTTLED
            reason = (
                f"trip ARMED→THROTTLED: dd_90={dd_90:.4f} <= "
                f"{cfg.trip_throttle:.4f}"
            )
    elif prev_state == STATE_THROTTLED:
        if dd_90 <= cfg.trip_pause:
            new_state = STATE_PAUSED
            reason = (
                f"trip THROTTLED→PAUSED: dd_90={dd_90:.4f} <= "
                f"{cfg.trip_pause:.4f}"
            )
        elif dd_30 >= cfg.rearm_throttle:
            new_state = STATE_ARMED
            reason = (
                f"rearm THROTTLED→ARMED: dd_30={dd_30:.4f} >= "
                f"{cfg.rearm_throttle:.4f}"
            )
    elif prev_state == STATE_PAUSED:
        if dd_30 >= cfg.rearm_pause and days_in_state >= cfg.min_dwell_days:
            new_state = STATE_THROTTLED
            reason = (
                f"rearm PAUSED→THROTTLED: dd_30={dd_30:.4f} >= "
                f"{cfg.rearm_pause:.4f} AND days_in_state={days_in_state} "
                f">= {cfg.min_dwell_days}"
            )

    transitioned = new_state != prev_state
    if transitioned:
        days_in_state = 0
        last_transition_ts = now.isoformat()
        logger.warning(
            "[circuit_breaker] %s → %s | %s | leverage=%.2f peak=%.2f equity=%.2f",
            prev_state,
            new_state,
            reason,
            _STATE_LEVERAGE[new_state],
            new_peak,
            last_equity,
        )
    else:
        last_transition_ts = prior_state.last_transition_ts
        # Increment dwell counter by 1 day per tick (consistent with daily
        # allocator cadence); the recomputation via timestamps above is
        # authoritative if last_transition_ts exists.
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
```

**Notes on the implementation:**

- `update_state` is the single public entry point. It is pure modulo two side-effects:
  reading and writing the JSON state file. Pass `persist=False` in backtests / unit tests.
- `peak_equity` provides the TIPP-style HWM ratchet for logging (`rolling_dd_hwm`),
  but the trip / re-arm logic uses **rolling-window** DD (S14 definition), which is more
  robust to a one-shot inflow that distorts the all-time HWM.
- The trip branch in `STATE_ARMED` deliberately steps only to `THROTTLED` (never directly to
  `PAUSED`), enforcing the invariant in §5. The next tick will then trip again to `PAUSED`.
- `min_dwell_days=10` is enforced on `PAUSED → THROTTLED` only (the most catastrophic
  rollback). `THROTTLED → ARMED` only requires the 30-day DD condition because the loss of
  risk is already smaller.
- `enabled=False` is the **single-flag rollback** (§9): returns `{1.0×, paused=False}` and
  short-circuits everything else.

---

## 7. Unified diffs (apply-ready)

> All diffs are presented as code blocks (not committed) because this report itself is
> read-only per the mandate. They are formatted to be applied with `git apply`.

### 7.1 `src/risk_manager/daily_allocator.py` — wire breaker into `run_allocation`

```diff
--- a/src/risk_manager/daily_allocator.py
+++ b/src/risk_manager/daily_allocator.py
@@ -10,11 +10,13 @@ import sys
 from decimal import Decimal
 import pytz # Import the pytz library
 
 # Add the project root to the Python path
 project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
 sys.path.insert(0, project_root)
 
 from common.database.MQSDBConnector import MQSDBConnector
 from orchestrator.marketData.fmpMarketData import FMPMarketData
+from risk_manager.drawdown_circuit_breaker import (
+    CircuitBreakerConfig,
+    update_state as cb_update_state,
+)
 
 # --- Configure Logging ---
 logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
@@ -23,16 +25,17 @@ class DailyAllocator:
     def __init__(self, config_path: str):
         self.db_connector = MQSDBConnector()
         self.market_data = FMPMarketData()
         self.db_timezone = pytz.timezone("America/New_York") # Define the target timezone
         try:
             with open(config_path, 'r') as f:
                 self.config = json.load(f)
             self.master_portfolio_id = self.config['master_portfolio_id']
             self.strategy_portfolios = self.config['portfolio_weights']
             self.currency = self.config['currency']
+            self.cb_cfg = CircuitBreakerConfig.from_mapping(self.config.get('circuit_breaker'))
         except FileNotFoundError:
             logger.exception(f"Configuration file not found at {config_path}")
             sys.exit(1)
         except (KeyError, json.JSONDecodeError) as e:
             logger.exception(f"Error parsing configuration file: {e}")
             sys.exit(1)
@@ -156,6 +159,52 @@ class DailyAllocator:
         finally:
             if conn: self.db_connector.release_connection(conn)
 
+    def _fetch_master_equity_curve(self, lookback_days: int = 365) -> "pd.DataFrame":
+        """
+        Build the daily master equity curve = sum across portfolios of
+        (cash + mark-to-market positions). Used as input to the circuit breaker.
+
+        Returns DataFrame with columns ['timestamp', 'equity']; empty if data
+        insufficient. The breaker's _coerce_equity_series accepts that shape.
+        """
+        import pandas as pd
+        sql = """
+            WITH ranked_cash AS (
+              SELECT portfolio_id, date::date AS d, notional,
+                     ROW_NUMBER() OVER (PARTITION BY portfolio_id, date::date
+                                        ORDER BY timestamp DESC) AS rn
+              FROM cash_equity_book
+              WHERE date >= CURRENT_DATE - INTERVAL '%s days'
+            ),
+            daily_cash AS (
+              SELECT d AS date, SUM(notional) AS cash_total
+              FROM ranked_cash WHERE rn = 1 GROUP BY d
+            ),
+            daily_pnl AS (
+              SELECT date::date AS date, SUM(notional) AS pnl_total
+              FROM pnl_book
+              WHERE date >= CURRENT_DATE - INTERVAL '%s days'
+              GROUP BY date::date
+            )
+            SELECT
+              COALESCE(c.date, p.date) AS timestamp,
+              COALESCE(p.pnl_total, c.cash_total, 0) AS equity
+            FROM daily_cash c FULL OUTER JOIN daily_pnl p
+              ON c.date = p.date
+            ORDER BY 1;
+        """
+        try:
+            result = self.db_connector.execute_query(
+                sql, (lookback_days, lookback_days), fetch='all'
+            )
+            if result.get('status') == 'success' and result.get('data'):
+                return pd.DataFrame(result['data'])
+        except Exception:
+            logger.exception("Failed to fetch master equity curve for circuit breaker.")
+        return pd.DataFrame(columns=['timestamp', 'equity'])
+
     def run_allocation(self):
-        # ... (no changes in this function)
+        """
+        Daily rebalance: read current equity, apply circuit breaker scaling,
+        push cash transfers toward target sleeve weights. The circuit breaker
+        only scales the *master capital allocation*; it does NOT touch the
+        sleeve-level vol target inside src/portfolios/portfolio_6/screener.py
+        (single vol-target contract per Team B B3 / SYNTHESIS §3.4).
+        """
         logger.info("Starting daily capital allocation...")
 
         master_cash = self._get_current_cash(self.master_portfolio_id)
@@ -179,16 +228,38 @@ class DailyAllocator:
         logger.info(f"Master Portfolio ({self.master_portfolio_id}) initial cash: {master_cash:.2f}")
         logger.info(f"Total System Equity calculated: {total_equity:.2f}")
 
+        # --- Drawdown circuit breaker -------------------------------------
+        cb_result = cb_update_state(
+            self._fetch_master_equity_curve(),
+            self.cb_cfg,
+        )
+        leverage_mult = Decimal(str(cb_result['leverage_multiplier']))
+        is_paused = bool(cb_result['paused'])
+        logger.info(
+            "[CB] state=%s leverage=%.2f paused=%s dd_90=%.4f dd_30=%.4f reason=%s",
+            cb_result['state'],
+            float(leverage_mult),
+            is_paused,
+            cb_result['rolling_dd_90d'],
+            cb_result['rolling_dd_30d'],
+            cb_result['reason'],
+        )
+
         conn = None
         try:
             conn = self.db_connector.get_connection()
             with conn.cursor() as cursor:
                 in_memory_master_cash = master_cash
                 for pid, weight in self.strategy_portfolios.items():
-                    weight = Decimal(str(weight))
+                    nominal_weight = Decimal(str(weight))
+                    # Scale sleeve weight by the breaker's leverage multiplier.
+                    # paused=True is functionally equivalent to leverage_mult=0,
+                    # which makes target_value = 0 → all sleeve cash returns to master.
+                    effective_weight = nominal_weight * leverage_mult
                     current_total_value = portfolio_valuations[pid]['total']
-                    target_value = total_equity * weight
+                    target_value = total_equity * effective_weight
                     adjustment = target_value - current_total_value
                     
                     logger.info(
-                        f"Portfolio {pid}: Current Value={current_total_value:.2f}, Target Value={target_value:.2f}, Adjustment={adjustment:.2f}"
+                        f"Portfolio {pid}: nominal_w={float(nominal_weight):.4f}, "
+                        f"effective_w={float(effective_weight):.4f}, "
+                        f"Current={current_total_value:.2f}, Target={target_value:.2f}, "
+                        f"Adj={adjustment:.2f}"
                     )
                     if adjustment.is_zero():
                         continue
```

> Notes:
> 1. The fetch SQL has two `%s` placeholders for `lookback_days`. If your `db_connector`
>    treats `%s` as positional, pass `(lookback_days, lookback_days)`. If it requires named
>    binds, swap to `%(lb)s` and pass `{"lb": lookback_days}`.
> 2. If your schema doesn't have `pnl_book`, the FULL OUTER JOIN gracefully falls back to
>    `cash_total` (which equals total notional when positions are 0, which they are at the
>    start of day for the master allocator's purpose).
> 3. **No** schema change is required. The breaker state lives in a JSON sidecar at
>    `src/risk_manager/.circuit_breaker_state.json` (path configurable). Optionally a
>    follow-up PR can move that into a `circuit_breaker_state` DB table.
> 4. **No change to `manage_capital.py`** — that file is the CLI for ADD/WITHDRAW funding,
>    not the daily orchestrator. Wiring there would mis-fire on every capital injection.

### 7.2 `src/portfolios/portfolio_manager_config.json` — add `circuit_breaker` block

```diff
--- a/src/portfolios/portfolio_manager_config.json
+++ b/src/portfolios/portfolio_manager_config.json
@@ -1,7 +1,21 @@
 {
+  "_invariants": [
+    "Team C C3 (2026-05-20): the 'circuit_breaker' block scales SLEEVE CAPITAL.",
+    "It does NOT measure realized volatility. It does NOT compute sigma_target/sigma_realized.",
+    "All vol-targeting lives inside src/portfolios/portfolio_6/screener.py::vol_target_scale.",
+    "Disable in emergency by setting circuit_breaker.enabled=false (single-flag rollback)."
+  ],
   "master_portfolio_id": "0",
   "currency": "USD",
   "portfolio_weights": {
     "1": 0.10,
     "2": 0.90
+  },
+  "circuit_breaker": {
+    "enabled": true,
+    "trip_throttle":   -0.15,
+    "trip_pause":      -0.25,
+    "rearm_throttle":  -0.075,
+    "rearm_pause":     -0.15,
+    "trip_window_days":   90,
+    "rearm_window_days":  30,
+    "min_dwell_days":     10,
+    "min_history_days":   30,
+    "state_path": "src/risk_manager/.circuit_breaker_state.json"
   }
 }
```

> If Team B's B3 `_invariants` array has already been merged at this key,
> *append* the four C3 entries to the existing array rather than overwriting.

### 7.3 (Optional, advisory) `docs/workflows/live-trading-flow.md` — circuit breaker note

This is documentation only and does not affect runtime. It is listed for completeness; per
the read-only mandate it is **not** committed by this report.

```diff
--- a/docs/workflows/live-trading-flow.md
+++ b/docs/workflows/live-trading-flow.md
@@ -91,11 +91,29 @@
 ## Key Features
 
-### Circuit Breaker Pattern
+### Circuit Breaker Pattern (operational, in `RunEngine`)
 - Tracks consecutive failures per portfolio
 - Stops portfolio thread after `max_consecutive_failures` (default: 5)
 - Prevents cascading failures from affecting other portfolios
 
+### Drawdown Circuit Breaker (P&L-driven, in `DailyAllocator`)
+- Lives in `src/risk_manager/drawdown_circuit_breaker.py` (Team C C3).
+- Three-state machine: ARMED (1.0x) → THROTTLED (0.5x) → PAUSED (0.0x).
+- Trips on 90-day rolling drawdown (`-15%` throttle, `-25%` pause).
+- Re-arms on 30-day rolling drawdown (`-7.5%` ARM, `-15%` THROTTLE) with
+  10-day minimum dwell (hysteresis).
+- Does NOT measure realized volatility (single vol-target contract per Team B
+  B3; sleeve-level vol target lives in `screener.vol_target_scale`).
+- Config block: `portfolio_manager_config.json::circuit_breaker`.
+- Rollback: set `circuit_breaker.enabled=false`.
+
 ### Thread Safety
 - Each portfolio runs in its own thread
```

---

## 8. Falsification test

A breaker that doesn't trip when it should is worse than no breaker at all (false sense of safety).
We define **three** falsifications, ordered cheapest → most expensive.

### F1 — Unit synthetic curve (must pass in < 1 second)

Construct a synthetic master equity curve:

```
day 000 → day 099 : equity = 100 (flat)
day 100 → day 119 : equity drops linearly from 100 → 80  (≈ -20% in 20 days)
day 120 → day 139 : equity flat at 80
day 140 → day 199 : equity recovers linearly from 80 → 100
```

**Expected breaker trace:**

| Window | dd_90d | dd_30d | Expected state |
|---|---|---|---|
| day 100..109 | ~−5% to ~−10% | ~−5% to ~−10% | ARMED |
| day 110..115 (DD crosses −15%) | ≤ −15% | ≤ −15% | **THROTTLED** |
| day 116..119 (DD still ≤ −20%) | ≤ −20% | ≤ −20% | THROTTLED (does NOT pause — needs −25%) |
| day 120..139 (DD flat at −20%) | ≤ −20% | ≤ −20% | THROTTLED (dwell accumulates) |
| day 140..170 (recovery; DD improves) | ≤ −10% | improves quickly | THROTTLED until dd_30d ≥ −7.5% |
| day ~190 onward (recovery near top) | dd_30d ≥ −7.5% | dd_30d ≥ −7.5% | **ARMED** |

**Pass criterion:** the trace contains exactly **one** transition into THROTTLED (around day
110–115) and exactly **one** transition back to ARMED (somewhere in day 180–200). No state ever
becomes PAUSED. No more than two transitions total. If the breaker flip-flops three or more
times on this monotone curve, hysteresis is broken — fail.

### F2 — 2008 historical backtest (canonical falsification)

The user request mandates this:

> "On a 2008 backtest, breaker must trigger between −15% and −20% drawdown OR the threshold is wrong."

Operationalized:

1. Run `src/main_backtest.py` over 2007-01-01 → 2009-12-31 with the current P6 config
   (or use the SPY total-return series as a proxy if the strategy backtest is too slow).
2. Feed the resulting `perf_df["portfolio_value"]` to `update_state` day by day with
   default thresholds.
3. **Pass criteria (all four):**
   a. At least one **THROTTLED** transition occurs in Sept–Oct 2008.
   b. At least one **PAUSED** transition occurs at some point during the 2008 crash.
      *(SPY's peak-to-trough 2007–2009 was −55% per S9; any reasonable strategy
      with vol target 13% should have shown at least −25% rolling DD.)*
   c. The breaker re-arms (PAUSED→THROTTLED, then THROTTLED→ARMED) by end of 2009,
      not earlier than 10 trading days after the trough.
   d. **Total number of state transitions ≤ 6** over the 2.5-year window
      (i.e. no excessive flip-flopping).

   **Fail conditions:**
   - Breaker never trips before max DD reaches −20% on SPY proxy → trip threshold too loose.
   - Breaker trips and re-arms in the same week multiple times → hysteresis insufficient
     (escalate `min_dwell_days` from 10 → 20).
   - Breaker never re-arms by 2010-01-01 → re-arm threshold too tight.

### F3 — Monte-Carlo robustness (recommended, not required)

Drive `update_state` with 1000 simulated equity curves (GBM with µ=8%, σ=13% ann, 5-year horizon).

**Pass criteria:**
- Across all 1000 paths, the breaker never increases the realized max DD vs. no-breaker baseline.
  (Sanity: a brake should reduce DD, never amplify it.)
- Median realized max DD with breaker ≤ 0.85 × median realized max DD without breaker
  (i.e. ≥ 15% DD reduction across the distribution).
- 95th-percentile state-transition count over 5 years ≤ 8 (no thrashing).

---

## 9. Risks + rollback

### 9.1 Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | False trip on a single bad day (e.g. data outage marks equity = 0) | High | `min_history_days=30` guard; `_coerce_equity_series` drops NaN & sorts/dedupes. Add a sanity check in `_fetch_master_equity_curve` rejecting any same-day move > 50% (out of scope for v1). |
| R2 | Breaker never trips because rolling-window definition uses *within-window* peak | M | `_rolling_drawdown_min` uses `np.maximum.accumulate` inside the window — same construction as `reporting.py::_compute_max_drawdown`. Cross-validated by F2. |
| R3 | Breaker traps the system in PAUSED permanently (S7 warning) | M | The `min_dwell_days=10` + `rearm_pause=-15%` recovery clause is exactly the S7 (Hsieh 2023) restart mechanism. F2.c falsifies this. |
| R4 | Single-flag disable not enough — bad code path inside `_fetch_master_equity_curve` keeps running | L | The fetch happens before `cb_update_state`; `enabled=false` makes `cb_update_state` return `{1.0×, ARMED}` and the fetch result is unused. If even the fetch is failing, the try/except returns empty → breaker no-ops. |
| R5 | Breaker scales the *sleeve* weight but the sleeve's own MAX_LEVERAGE=1.5 multiplies back to 1.5× → effective leverage = 0.75 not 0.5 | L (works as intended) | This is **correct behavior**. Sleeve-internal vol target operates over the sleeve's allocated capital; halving the allocated capital halves the gross dollar exposure regardless. Verified by reading `Portfolio6Strategy._rebalance` (`src/portfolios/portfolio_6/strategy.py:196–293`): `target_weights` are *fractions of the sleeve's notional*, not fractions of total equity. |
| R6 | Race condition: allocator runs at 16:30 ET but P6 OnData runs intra-day at 09:30 ET — sleeve might trade up to 0.5× allocation under old breaker state | L | Allocator decides daily capital share; sleeve can only trade up to *its own capital*. As long as the previous-day cash transfer respected the breaker, P6's intra-day exposure is bounded. |
| R7 | Schema mismatch: `pnl_book` does not exist in some deployments | L | The fetch SQL uses FULL OUTER JOIN and falls back to cash-only equity; the breaker still operates (just on a noisier signal). For full mark-to-market, add a `pnl_book` daily row builder (out of scope; flagged for OMS monitor work). |
| R8 | Persisted state file is wiped accidentally | L | A wipe resets to ARMED with peak=0. Worst case: one missed throttle window until the rolling DD recomputes its trip on the next pass. |
| R9 | First-deploy spurious trip: existing strategies started 2024-01-01 with −18% DD already in the books | M | On first run, `peak_equity=0` until the fetch populates it. Add a one-shot `INITIAL_PEAK_OVERRIDE` flag (future): not in v1 to keep config small. Documented operationally: run the breaker in shadow mode (`enabled=false` + parse logs) for one quarter before flipping live. |

### 9.2 Rollback path (single flag)

```jsonc
// src/portfolios/portfolio_manager_config.json
{
  ...
  "circuit_breaker": {
    "enabled": false,     // ← single flag
    ...
  }
}
```

When `enabled=false`, `cb_update_state` returns `{state: ARMED, leverage_multiplier: 1.0, paused:
False, reason: "disabled"}` and the allocator behavior is **bit-exact identical to pre-merge** —
`effective_weight == nominal_weight × Decimal("1.0")` is the original code.

Three-tier rollback:

| Tier | Action | Effect |
|---|---|---|
| 1 (config flip) | Set `enabled=false` in config; no deploy needed (re-read on next allocator run) | Breaker idle, allocator falls back to constant weights |
| 2 (state reset) | Delete `src/risk_manager/.circuit_breaker_state.json` | Forces ARMED state; useful if persisted state is corrupted |
| 3 (revert) | `git revert <merge-commit>` | Removes the import, the wiring, the fetch helper, and the config block atomically |

---

## 10. Cross-team handoffs

| Item | Affects |
|---|---|
| Breaker hooks into `DailyAllocator.run_allocation` *before* transfers | Team B B3 invariant remains intact — breaker scales master sleeve weights, not σ-anything |
| Breaker reads `cash_equity_book` + `pnl_book` | Team D backtest harness should run `update_state(persist=False)` so backtests don't pollute the live state file |
| `pnl_book` may be missing in some deployments | Team B B4 (OMS monitor — currently `src/oms/monitor.py` is a 0-byte stub) is the natural owner of populating `pnl_book` daily |
| The string "circuit breaker" is now overloaded (exception + drawdown) | docs PR (§7.3) splits them into two named sections; the Python module name `drawdown_circuit_breaker` keeps the distinction in code |

---

## 11. Apply checklist

1. Create `src/risk_manager/drawdown_circuit_breaker.py` (§6 body).
2. Apply the diff in §7.1 to `src/risk_manager/daily_allocator.py`.
3. Apply the diff in §7.2 to `src/portfolios/portfolio_manager_config.json`.
4. (Optional) Apply the doc diff in §7.3.
5. Smoke test: `python -c "from src.risk_manager.drawdown_circuit_breaker import update_state, CircuitBreakerConfig; print(update_state([(f'2024-01-{d+1:02d}', 100 - d*0.1) for d in range(60)], CircuitBreakerConfig()))"` — should report `ARMED`, dd_30d ≈ −0.06.
6. Falsification F1 (synthetic curve) — must complete in < 1 second.
7. Falsification F2 (2008 backtest) — must satisfy all four pass criteria before turning the breaker on in production. Run with `enabled=false` for one quarter ("shadow mode" — logs only) before flipping live.
8. Rollback drill: flip `enabled=false`, confirm next allocator run logs `reason: disabled`, confirm transfers unchanged vs. pre-merge.

---

*End of C3.*
