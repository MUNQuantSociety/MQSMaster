# Team C Synthesis — Hedging & Risk

**Date:** 2026-05-20
**Inputs:** C1 (trend hedge ticker), C2 (tail-risk hedge), C3 (drawdown circuit breaker).
**Status:** Read-only synthesis. Diffs apply-ready.

---

## 1. Decisions

| # | Decision | Source |
|---|---|---|
| D1 | `TREND_HEDGE_TICKER = "DBMF"` (iMGP DBi Managed Futures). Single config line. | C1 |
| D2 | **SKIP** third tail-hedge sleeve (TAIL / VXTH / OTM puts). Trend sleeve covers crisis-alpha. ETF wrappers carry empirical drag (TAIL: −8.0%/yr 5y). Universa-style options out of scope (no options pricing layer). | C2 |
| D3 | **ADD** drawdown circuit breaker at master allocator level. 3-state hysteresis (ARMED 1.0× / THROTTLED 0.5× / PAUSED 0.0×). Trip on 90d rolling DD ≤ -15% / -25%. Re-arm on 30d rolling DD ≥ -7.5% / -15%, dwell ≥ 10 days. | C3 |
| D4 | Single C2 + C3 invariant: breaker scales SLEEVE CAPITAL only. No second vol-target layer. Honors Team B B3 contract. | C2, C3, B3 |

## 2. New files (total: 1)

| Path | Purpose | Source |
|---|---|---|
| `src/risk_manager/drawdown_circuit_breaker.py` | `CircuitBreakerConfig`, `CircuitBreakerState`, `update_state(equity_df, cfg)` → `{state, leverage_multiplier, paused, ...}`. JSON sidecar persistence at `src/risk_manager/.circuit_breaker_state.json`. | C3 §6 |

C1 and C2 introduce no new files.

## 3. Patches to existing files

### 3.1 `src/portfolios/portfolio_6/config.json`

Three concerns (Team A + Team B + Team C) touch this file. Combined block now:

```diff
--- a/src/portfolios/portfolio_6/config.json
+++ b/src/portfolios/portfolio_6/config.json
@@ -17,7 +17,7 @@
     "MAX_LEVERAGE": 1.5,
     "GLD_TICKER": "GLD",
     "GLD_WEIGHT": 0.07,
-    "TREND_HEDGE_TICKER": "",
+    "TREND_HEDGE_TICKER": "DBMF",
     "TREND_HEDGE_WEIGHT": 0.10,
     "REBALANCE_DRIFT_THRESHOLD": 0.005,
     "DSR_MIN_PROB": 0.5,
```

C2's optional `TAIL_HEDGE_TICKER` / `TAIL_HEDGE_WEIGHT` placeholders are **NOT recommended for v1** (would be dead config). Add them only if/when the C2 falsification test passes.

### 3.2 `src/portfolios/portfolio_6/strategy.py`

Stacks on top of Team A + B1 + B3 patches. C2 adds a docstring-only block explaining the SKIP decision. Optional — defer if comment density gets too high.

### 3.3 `src/risk_manager/daily_allocator.py`

Two concerns:
- B3 §7.3 (docstring declaring single-vol-target contract)
- C3 §7.1 (wire circuit breaker into `run_allocation`: import + `_fetch_master_equity_curve` helper + `cb_update_state` call + `leverage_mult` applied to each sleeve weight)

These stack cleanly — different functions / different line ranges.

### 3.4 `src/portfolios/portfolio_manager_config.json`

Four concerns now touch this file:
- Team A SYNTHESIS §9 — register P7, P8 at weight 0.0
- B3 §7.4 — vol-target invariants
- C3 §7.2 — `circuit_breaker` config block + C3 invariants

Combined target file:

```json
{
  "_invariants": [
    "B3 audit (2026-05-20): portfolio_weights are CAPITAL shares, not vol shares.",
    "DailyAllocator must not measure realized master-portfolio volatility.",
    "Sleeve-level vol-targeting (sigma_target / sigma_realized) lives inside",
    "src/portfolios/portfolio_6/screener.py::vol_target_scale only.",
    "Adding a second vol-target layer here (or in DailyAllocator) compounds.",
    "C3 (2026-05-20): the 'circuit_breaker' block scales SLEEVE CAPITAL.",
    "It does NOT measure realized volatility. It does NOT compute sigma_target/sigma_realized.",
    "Disable in emergency by setting circuit_breaker.enabled=false."
  ],
  "master_portfolio_id": "0",
  "currency": "USD",
  "portfolio_weights": {
    "1": 0.10,
    "2": 0.90,
    "7": 0.0,
    "8": 0.0
  },
  "circuit_breaker": {
    "enabled": true,
    "state_path": "src/risk_manager/.circuit_breaker_state.json",
    "throttle_dd_threshold_90d": -0.15,
    "pause_dd_threshold_90d": -0.25,
    "rearm_throttle_dd_threshold_30d": -0.075,
    "rearm_arm_dd_threshold_30d": -0.15,
    "min_dwell_days_paused": 10,
    "min_lookback_days": 90,
    "lookback_days_fetch": 365,
    "throttled_leverage_multiplier": 0.5
  }
}
```

## 4. Pre-merge blocker: DBMF market_data backfill

Per C1 §8 — before merging the `TREND_HEDGE_TICKER = "DBMF"` config flip, verify DBMF is present in `market_data` over the full backtest window. SQL check:

```sql
SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM market_data WHERE ticker = 'DBMF';
```

If missing or partial, run `scripts/APIs/` or `src/orchestrator/backfill/specific_backfill.py` to fetch from FMP. Earliest possible date = `2019-05-08` (DBMF inception). Backtests starting before that **must** set `TREND_HEDGE_TICKER = ""` in a config override.

Also ensure `realtimeDataIngestor.py` polling list includes `DBMF` for live mode.

## 5. Apply order (single PR, after Team A + Team B patches land)

1. **NEW** `src/risk_manager/drawdown_circuit_breaker.py` (C3 §6).
2. **NEW** `src/risk_manager/.circuit_breaker_state.json` is created on first run by `CircuitBreakerState.save()`; no manual seed needed. (Optionally seed with `{state: "armed", peak_equity: 0.0, ...}` to start clean.)
3. **PATCH** `src/risk_manager/daily_allocator.py` (B3 docstring + C3 wiring).
4. **PATCH** `src/portfolios/portfolio_6/config.json` (C1 ticker line + earlier Team A + B1 keys).
5. **PATCH** `src/portfolios/portfolio_manager_config.json` (Team A weights + B3 invariants + C3 circuit_breaker block).
6. **OPTIONAL** docstring patch in `src/portfolios/portfolio_6/strategy.py` (C2 SKIP rationale).
7. **DATA** Backfill DBMF in `market_data` if missing.

## 6. Defaults preserve current behavior — almost

| Knob | Default after merge | Effect |
|---|---|---|
| `TREND_HEDGE_TICKER` | `"DBMF"` (was `""`) | Hedge sleeve activates at 10%. **First behavior change** — was previously inactive. |
| `circuit_breaker.enabled` | `true` (recommended) | Breaker monitors equity curve; in calm markets `leverage_multiplier = 1.0×` → no behavior change. **Behavior changes only on drawdown.** |
| `TAIL_HEDGE_TICKER` | not added | No change. |

**Caveat for merge PR:** the trend sleeve activation is a behavioral change. Run the C1 §9 falsification test (rolling 36m correlation ∈ [-0.20, +0.30]) on at least 36 months of post-2019 backtest data before promoting.

## 7. Falsification gates (post-merge)

| Gate | Pass criterion |
|---|---|
| C1 — DBMF diversification | Rolling 36m correlation of DBMF returns vs P6 stock sleeve ∈ [-0.20, +0.30], mean ≤ +0.10 over 2019-05 → present. Else demote to KMLM. |
| C1 — SG Trend signal alive | SG Trend Index 36m Sharpe > 0. Else cut sleeve weight to 0.05. |
| C2 — TAIL reversal | In 2008-2026 walk-forward, adding TAIL @ 5% (paid from stock sleeve) must yield ΔCAGR ≥ 0 AND Δmax-DD ≤ -4pp. If passes → flip SKIP to ADD. |
| C3 — Circuit breaker trips correctly | On 2008 backtest, breaker must enter THROTTLED between -15% and -20% master-curve DD and PAUSED between -20% and -30%. Outside these bands → thresholds wrong. |
| C3 — Hysteresis | Toggle frequency ≤ 4 transitions/year on 2008-2026 backtest. >4 indicates dwell-time too short. |

## 8. Inter-team handoffs

| Item | Affects |
|---|---|
| DBMF activates 10% hedge sleeve | Team A: P7/P8 inherit hedge unchanged (correct). Team B B4 cost-model: DBMF is liquid large-cap-equivalent so impact is modest. |
| Circuit breaker scales sleeve capital | Team A: P7/P8 sleeves both scale together — correct (master allocator scales all sleeves by same `leverage_mult`). |
| No tail sleeve added | Team D2 / D4: validation gates don't need to test a tail sleeve. Team A's P7/P8 hedge construction stays as-is. |
| `_invariants` JSON sibling now used by C3 too | If Team D adds JSON schema enforcement, `_invariants` must be allowed. |

## 9. Risks summary

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | DBMF backfill incomplete → hedge silently drops | High | §4 pre-merge check is a blocker, not optional. |
| R2 | Backtest start date < 2019-05-08 → DBMF has no data | M | Document override path: set `TREND_HEDGE_TICKER = ""` for legacy windows. |
| R3 | Circuit breaker false trip during flash crash + immediate recovery (e.g. Aug 2024) | M | 10-day dwell on PAUSED→THROTTLED prevents over-eager re-entry. 90d rolling window vs 5d intraday avoids most false trips. |
| R4 | Future change adds a second vol-target layer (forbidden by B3) | M | `_invariants` blob in manager config + comments in 4 files. C3 §7.2 explicit. |
| R5 | Master equity curve SQL fetch slow on first run | L | Single query against indexed `cash_equity_book` + `pnl_book`, lookback 365d, executes in <100ms typical. |
| R6 | JSON sidecar `.circuit_breaker_state.json` corruption | L | `CircuitBreakerState.load()` returns defaults on parse error. Worst case: lose hysteresis dwell counter; breaker auto-rebuilds on next tick. |

## 10. Rollback paths

| Concern | Rollback |
|---|---|
| DBMF misbehaves | `"TREND_HEDGE_TICKER": ""` in P6 config.json (single line). Stock + GLD continue. |
| Swap to KMLM | `"TREND_HEDGE_TICKER": "KMLM"` (single line; verify KMLM backfill first). |
| Circuit breaker false-trips | `"enabled": false` in `circuit_breaker` config block (single key). Allocator returns to dollar-share-only behavior. |
| Threshold tuning | Edit the four DD thresholds in `circuit_breaker` config; no code change. |
| Full revert | `git revert <commits>` on C-team PR. |

---

End of Team C synthesis. One new file, four file patches, one DBMF data backfill prerequisite, five falsification gates.
