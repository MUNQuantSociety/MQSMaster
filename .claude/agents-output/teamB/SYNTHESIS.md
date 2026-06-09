# Team B Synthesis — Portfolio Construction

**Date:** 2026-05-20
**Inputs:** B1 (HRP weighting), B2 (covariance shrinkage), B3 (vol-target audit), B4 (execution + TCA).
**Status:** Read-only synthesis. Diffs apply-ready below.

---

## 1. Decisions

| # | Decision | Source |
|---|---|---|
| D1 | Add HRP (and ERC stub) as opt-in alongside `inverse_vol_weights`. Default `INV_VOL` = bit-exact back-compat. | B1 |
| D2 | Build standalone `src/portfolios/common/covariance.py` (sample / LW-CC / LW-identity / OAS). Default `ledoit_wolf_cc`. No callers today; consumed opt-in by B1's HRP via `self._shrinkage_cov` attribute. | B2 |
| D3 | **NO double vol-targeting today.** Current architecture (sleeve-level scale in `screener.vol_target_scale` + dollar-share `DailyAllocator`) is correct per Risk-Parity literature. Action = **DOCUMENT** via 4 comment-only diffs. | B3 |
| D4 | Defer VWAP/TWAP order types. Backtest realism gap is the bigger problem (current `SLIPPAGE=1e-6` = 0.01 bps vs realistic 5–15 bps large-cap). Build `src/backtest/cost_model.py` (fixed + ½ spread + α·σ·√(Q/ADV)) + surgical patches to `executor.py` + `backtest_engine.py`. | B4 |
| D5 | All four deliverables are **independent + back-compat**. Default code paths unchanged unless config flips. | new |

## 2. New files (total: 3)

| Path | Purpose | Source | Imported by |
|---|---|---|---|
| `src/portfolios/portfolio_6/hrp_weights.py` | `hrp_weights()`, `erc_weights()` stub. | B1 §6 | P6 `strategy.py` (dispatcher) |
| `src/portfolios/common/covariance.py` | `shrink_cov(df, method)` with 4 methods. | B2 §6 | (none today — opt-in by HRP via attr) |
| `src/backtest/cost_model.py` | `CostModel` (fixed + spread + sqrt impact). | B4 §6.1 | `executor.py`, `backtest_engine.py` |

Also need `src/portfolios/common/__init__.py` (empty) to register the new package.

## 3. Patches to existing files

### 3.1 `src/portfolios/portfolio_6/strategy.py`

Three concerns touch this file:
- Team A SYNTHESIS §7 (score_universe call-site + config-loaded knobs)
- B1 §7.1 (weighting dispatcher)
- B3 §7.2 (single vol-target contract comments)

**Apply order:** Team A first, then B1, then B3 (alphabetical hunks don't overlap). All three diffs stack cleanly — confirmed by line-range disjointness.

### 3.2 `src/portfolios/portfolio_6/config.json`

Concerns:
- Team A SYNTHESIS §6 (SCORE_METHOD, SCORE_WEIGHTS, USE_*, EXCLUSIONS)
- B1 §7.2 (WEIGHTING_METHOD, HRP_LINKAGE_METHOD)

Merged block (apply Team A first, then add B1 keys):

```json
"DSR_MIN_PROB": 0.5,

"WEIGHTING_METHOD": "INV_VOL",
"HRP_LINKAGE_METHOD": "single",

"SCORE_METHOD": "rank_sum",
"SCORE_WEIGHTS": {
  "vol": 1.0,
  "max_one_day": 1.0,
  "gross_profit_to_assets": 1.0,
  "momentum": 1.0,
  "op_profit": 1.0,
  "asset_growth": 1.0
},
"SCORE_WINSOR_SIGMA": 3.0,

"USE_MOMENTUM_12_2": false,
"USE_OPERATING_PROFITABILITY": false,
"USE_ASSET_GROWTH": false,
"MOMENTUM_LOOKBACK_DAYS": 252,
"MOMENTUM_SKIP_DAYS": 21,

"EXCLUSIONS": { ... see Team A SYNTHESIS §6 ... }
```

### 3.3 `src/portfolios/portfolio_6/screener.py`

B3 §7.1 adds a docstring to `vol_target_scale` declaring the single-layer contract. Stacks cleanly with Team A's unified `score_universe` rewrite (different functions).

### 3.4 `src/risk_manager/daily_allocator.py`

B3 §7.3: pure docstring addition to `run_allocation`. Zero behavior change.

### 3.5 `src/portfolios/portfolio_manager_config.json`

Two concerns:
- Team A SYNTHESIS §9 (add `"7": 0.0`, `"8": 0.0` entries)
- B3 §7.4 (add `_invariants` sibling key)

Combined:

```json
{
  "_invariants": [
    "B3 audit (2026-05-20): portfolio_weights are CAPITAL shares, not vol shares.",
    "DailyAllocator must not measure realized master-portfolio volatility.",
    "Sleeve-level vol-targeting (sigma_target / sigma_realized) lives inside",
    "src/portfolios/portfolio_6/screener.py::vol_target_scale only.",
    "Adding a second vol-target layer here (or in DailyAllocator) compounds",
    "and silently collapses realized vol. See .claude/agents-output/teamB/B3_vol_target_audit.md."
  ],
  "master_portfolio_id": "0",
  "currency": "USD",
  "portfolio_weights": {
    "1": 0.10,
    "2": 0.90,
    "7": 0.0,
    "8": 0.0
  }
}
```

### 3.6 `src/backtest/executor.py` + `src/backtest/backtest_engine.py`

B4 §6.2, §6.3: add optional `cost_model: CostModel | None` parameter to executor + engine `setup`. Default `None` → legacy constant-slippage path. When wired in `main_backtest.py`, costs become realistic.

### 3.7 `src/main_backtest.py`

Three concerns:
- Team A SYNTHESIS §9 (P7/P8 imports + AVAILABLE_PORTFOLIO_CLASSES)
- B4 (wire `CostModel.for_large_cap()` into the engine setup; raise `SLIPPAGE` constant from `1e-6` to `0.0` since cost_model now handles costs)

Combined wire-up sketch (post Team A):

```python
from src.backtest.cost_model import CostModel
# ... existing imports + Portfolio7Strategy + Portfolio8Strategy ...

SLIPPAGE = 0.0                                                       # was 1e-6; cost_model now authoritative
COST_MODEL = CostModel.for_large_cap()                               # 0.5 + 4.0 + 1.0*σ*√(Q/ADV) bps

engine.setup(
    portfolio_classes=AVAILABLE_PORTFOLIO_CLASSES,
    start_date=START_DATE,
    end_date=END_DATE,
    initial_capital=INITIAL_CAPITAL,
    slippage=SLIPPAGE,
    cost_model=COST_MODEL,                                           # NEW
    backtest_mode=BACKTEST_MODE,
    fast_config=FAST_CONFIG,
)
```

## 4. Apply order (single PR, after Team A patches land)

1. **NEW** `src/portfolios/common/__init__.py` (empty).
2. **NEW** `src/portfolios/common/covariance.py` (B2).
3. **NEW** `src/portfolios/portfolio_6/hrp_weights.py` (B1).
4. **NEW** `src/backtest/cost_model.py` (B4).
5. **PATCH** `src/portfolios/portfolio_6/screener.py` (B3 docstring on top of Team A rewrite).
6. **PATCH** `src/portfolios/portfolio_6/strategy.py` (B1 dispatcher + B3 comments on top of Team A wire-up).
7. **PATCH** `src/portfolios/portfolio_6/config.json` (B1 keys on top of Team A merged block).
8. **PATCH** `src/risk_manager/daily_allocator.py` (B3 docstring only).
9. **PATCH** `src/portfolios/portfolio_manager_config.json` (B3 `_invariants` + Team A `"7"`,`"8"` entries).
10. **PATCH** `src/backtest/executor.py` (B4 cost_model wiring).
11. **PATCH** `src/backtest/backtest_engine.py` (B4 vectorized costs).
12. **PATCH** `src/main_backtest.py` (B4 cost_model instantiation + Team A P7/P8 registration).

## 5. Defaults preserve current behavior bit-exact

| Knob | Default | Effect |
|---|---|---|
| `WEIGHTING_METHOD` | `INV_VOL` | Dispatcher routes to existing `inverse_vol_weights`; same call, same args, same result. |
| `shrink_cov` consumers | none | No code path calls it until HRP is enabled. |
| `vol_target_scale` | unchanged | B3 patch is pure docstring. |
| `CostModel` in `executor.setup(...)` | `None` | Falls back to legacy `slippage` constant. |
| `main_backtest.py SLIPPAGE` | 1e-6 → 0.0 + `for_large_cap()` cost model | Realistic costs activated. **Sharpe will drop vs old number** — this is the point. |

**Caveat:** step 12 (main_backtest.py) is the ONE place defaults change. The first backtest after merge will show different (lower) Sharpe than pre-merge. This is correct — the old `SLIPPAGE=1e-6` was 500–5000× too small. Document this in the PR description so reviewers don't panic.

## 6. Falsification gates (post-merge)

| Gate | Pass criterion |
|---|---|
| HRP back-compat | With `WEIGHTING_METHOD="INV_VOL"` (default), P6 stock-sleeve weights match pre-merge bit-exact. |
| HRP promotion | ΔSharpe ≥ +0.10 over 5y, ΔMaxDD ≥ −0.02, ΔTurnover ≤ +0.30, DSR ≥ 0.5 vs INV_VOL on identical universe. Else default stays `INV_VOL`. |
| `shrink_cov` correctness | Frobenius distance to sklearn `LedoitWolf(assume_centered=True)` on 100×252 random panel < 1e-6 (B2 §8). |
| Vol-target single-layer invariant | If anyone proposes allocator-level vol-target, run B3 §8 harness; Run C (both layers) realized vol must be < 50% of target — confirms double-counting and rejects the change. |
| Cost-model realism | After enabling cost_model, P6 monthly-rebal annualized cost drag ∈ [5, 50] bps (B4 §7). Outside this range → params mis-calibrated. |

## 7. Inter-team handoffs

| Item | Affects |
|---|---|
| `WEIGHTING_METHOD="HRP"` shifts cluster-level weights → may change top concentrations | Team C1 (trend hedge ticker selection: re-evaluate after HRP merge if HRP becomes default) |
| Realistic cost model adds 5–50 bps/year drag | Team A's P7/P8 falsification gates (`Sharpe(P7)−Sharpe(P6) ≥ 0.15`) must be re-measured under cost model ON. |
| Single-layer vol contract is now documented | Team C3 (drawdown circuit breaker) must NOT add a second vol target — it can throttle leverage or pause sleeve, but not scale by `σ_target/σ_realized`. |
| `covariance.py` module is shared utility | Team D backtest controls + Team C tail hedges may also want robust Σ. Available out of the box. |

## 8. Risks summary

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | First post-merge backtest shows lower Sharpe due to realistic cost model | High awareness, low actual risk | Document in PR. Compare with `CostModel.disabled()` to recover old number. |
| R2 | HRP single-linkage chaining | M | `HRP_LINKAGE_METHOD` toggle to `ward`. |
| R3 | LW-CC formula transcription bug | L | Auxiliary falsifications in B2 §8 + 3-way cross-validation (covShrinkage, WLM1ke, Ledoit MATLAB output). |
| R4 | Cost-model parameters mis-calibrated | M | Falsification gate §6 catches both over- and under-estimates. |
| R5 | Future change breaks single-layer vol contract silently | M | Documented at 4 sites (screener.py, strategy.py, daily_allocator.py, manager_config.json `_invariants`). |

## 9. Rollback paths (smallest sufficient first)

| Concern | Rollback |
|---|---|
| HRP misbehaves | `"WEIGHTING_METHOD": "INV_VOL"` in P6 config.json. No deploy. |
| `shrink_cov` wrong | Don't call it (nothing imports by default). Or `git rm src/portfolios/common/covariance.py`. |
| B3 docstrings out of date | `git revert <commit>` — zero behavior change either way. |
| Cost model wrong | `engine.setup(..., cost_model=CostModel.disabled())` in `main_backtest.py`, or pass `None`. |

---

End of Team B synthesis. All defaults preserve current Portfolio_6 behavior bit-exact. Apply order in §4. Falsification gates in §6 must run before promoting any default flip.
