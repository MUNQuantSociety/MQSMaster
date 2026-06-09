# MQSMaster Portfolio_7 + Portfolio_8 — Master Synthesis

**Date:** 2026-05-20
**Branch:** dev
**Method:** 16-agent parallel review across 4 teams (A signal, B construction, C hedging/risk, D validation).
**Inputs:** team-level syntheses at `.claude/agents-output/team{A,B,C,D}/SYNTHESIS.md`.
**Status:** Plan locked. Awaiting apply phase.

---

## 1. Top-line outcome

Two new portfolios, both subclassing `Portfolio6Strategy`:

- **Portfolio_7** = P6 + cross-sectional FinBERT sentiment z-score tilt (`w ← w · exp(λ · z)`, λ=0.25, 21d EWM, strict ex-ante).
- **Portfolio_8** = P6 with extra RBP-rank term inside `score_universe` composite (monthly refresh via `RBPPipeline.run()`).

Both register at capital weight **0.0** in `portfolio_manager_config.json` — loaded, monitored, attribution-tracked, but no capital allocated until falsification gates pass.

Portfolio_9 is **not warranted** at this stage — A3 explicitly rejected stacking RBP on top of P7. Three separable sleeves (P6 baseline, P7 sentiment, P8 RBP) cover the design space.

## 2. Decisions matrix

| # | Decision | Team | File impact |
|---|---|---|---|
| 1 | Build P7 (sentiment overlay) | A2 | NEW src/portfolios/portfolio_7/ |
| 2 | Build P8 (RBP rank overlay) | A3 | NEW src/portfolios/portfolio_8/ |
| 3 | Refactor `score_universe`: `method ∈ {rank_sum, weighted_z}` + `SCORE_WEIGHTS` + winsor; back-compat default `rank_sum` | A4 | PATCH src/portfolios/portfolio_6/screener.py |
| 4 | Add 3 factor flags (momentum 12-2, op-profit, asset-growth), all OFF | A1 | PATCH screener.py + config.json |
| 5 | Add lottery exclusion module (8 rules, all OFF) | A5 | NEW src/portfolios/portfolio_6/exclusions.py |
| 6 | HRP weighting opt-in; default `INV_VOL` = bit-exact | B1 | NEW src/portfolios/portfolio_6/hrp_weights.py |
| 7 | Covariance shrinkage utility (LW-CC default) | B2 | NEW src/portfolios/common/covariance.py |
| 8 | **No double vol-targeting** — document the single-layer contract at 4 sites | B3 | DOC patches only |
| 9 | Defer VWAP/TWAP; add backtest cost model (fixed + ½ spread + α·σ·√(Q/ADV)) | B4 | NEW src/backtest/cost_model.py + executor/engine patches |
| 10 | `TREND_HEDGE_TICKER = "DBMF"` | C1 | 1-line config edit + DBMF backfill |
| 11 | **SKIP** tail-hedge sleeve. Documented rationale + falsification trigger | C2 | DOC comment only |
| 12 | Drawdown circuit breaker: 3-state hysteresis at master allocator | C3 | NEW src/risk_manager/drawdown_circuit_breaker.py |
| 13 | CSCV + PBO + purged k-fold infrastructure | D1 | NEW src/backtest/cscv.py + purged_kfold.py |
| 14 | Survivorship-bias Tier 1 (doc only); Tier 2 PIT pipeline = follow-up PR | D2 | DOC patch in main_backtest.py |
| 15 | **P0 BLOCKER**: Fix `NLP/core/paths.py` MODEL_DIR typo | D3 | 1-line PATCH |
| 16 | Default `SENTIMENT_FALLBACK_TO_MARKET_DATA = false` until aggregate-refresh fixed | D3 | 1-line config edit |
| 17 | Add P7 look-ahead CI test + live/backtest parity CI test | D3, D4 | NEW tests/ files |

## 3. New files (12 total)

| Path | Source |
|---|---|
| `src/portfolios/portfolio_6/exclusions.py` | A5 |
| `src/portfolios/portfolio_6/hrp_weights.py` | B1 |
| `src/portfolios/common/__init__.py` (empty) | B2 |
| `src/portfolios/common/covariance.py` | B2 |
| `src/portfolios/portfolio_7/__init__.py` | A2 |
| `src/portfolios/portfolio_7/strategy.py` | A2 |
| `src/portfolios/portfolio_7/config.json` | A2 |
| `src/portfolios/portfolio_8/__init__.py` | A3 |
| `src/portfolios/portfolio_8/strategy.py` | A3 |
| `src/portfolios/portfolio_8/config.json` | A3 |
| `src/backtest/cost_model.py` | B4 |
| `src/backtest/cscv.py` | D1 |
| `src/backtest/purged_kfold.py` | D1 |
| `src/risk_manager/drawdown_circuit_breaker.py` | C3 |
| `tests/portfolios/test_p7_lookahead.py` | D3 |
| `tests/integration/test_live_backtest_parity.py` | D4 |

## 4. Patched files (12 total)

| Path | Source(s) |
|---|---|
| `NLP/core/paths.py` | D3 (P0) |
| `src/portfolios/portfolio_6/screener.py` | A1+A4+A5 (unified rewrite per Team A SYNTHESIS §5) + B3 docstring |
| `src/portfolios/portfolio_6/strategy.py` | Team A SYNTHESIS §7 + B1 dispatcher + B3 comments + C2 doc |
| `src/portfolios/portfolio_6/config.json` | Team A §6 + B1 + C1 (DBMF ticker) |
| `src/portfolios/portfolio_manager_config.json` | Team A §9 (P7+P8 weights) + B3 invariants + C3 circuit_breaker block |
| `src/main_backtest.py` | Team A §9 (imports) + B4 (cost model) + D2 (doc comment) |
| `src/risk_manager/daily_allocator.py` | B3 docstring + C3 wiring |
| `src/backtest/executor.py` | B4 (cost_model kwarg) |
| `src/backtest/backtest_engine.py` | B4 (cost_model + vectorized costs) + D1 (walk-forward optional) |
| `scripts/Backtest_Analysis/backtest_analyzer.py` | D1 (CSCV/PBO/DSR CLI flags) |
| `src/orchestrator/backfill/update/refresh.py` | D2 Tier 1 (history sidecar) |

## 5. Apply order — single PR (sequential)

### Wave 1 — P0 BLOCKER + back-compat new files (no behavior change)

1. `NLP/core/paths.py` (D3 P0 fix).
2. `src/portfolios/common/__init__.py` (empty).
3. `src/portfolios/common/covariance.py` (B2).
4. `src/portfolios/portfolio_6/exclusions.py` (A5).
5. `src/portfolios/portfolio_6/hrp_weights.py` (B1).
6. `src/backtest/cost_model.py` (B4).
7. `src/backtest/cscv.py` (D1).
8. `src/backtest/purged_kfold.py` (D1).
9. `src/risk_manager/drawdown_circuit_breaker.py` (C3).

### Wave 2 — P6 patches (defaults preserve behavior bit-exact)

10. `src/portfolios/portfolio_6/screener.py` — unified rewrite per Team A SYNTHESIS §5 + B3 docstring.
11. `src/portfolios/portfolio_6/strategy.py` — Team A §7 + B1 dispatcher + B3 comments.
12. `src/portfolios/portfolio_6/config.json` — merged block per Team A §6 + B1 keys + C1 DBMF ticker.

### Wave 3 — Allocator + master config patches

13. `src/risk_manager/daily_allocator.py` (B3 + C3).
14. `src/portfolios/portfolio_manager_config.json` (Team A weights + B3 invariants + C3 breaker).

### Wave 4 — P7 + P8

15. `src/portfolios/portfolio_7/*` (NEW; A2 verbatim).
16. `src/portfolios/portfolio_7/config.json` — **set `SENTIMENT_FALLBACK_TO_MARKET_DATA: false`** per D3b.
17. `src/portfolios/portfolio_8/*` (NEW; A3 verbatim).

### Wave 5 — Backtest infrastructure

18. `src/backtest/executor.py` (B4).
19. `src/backtest/backtest_engine.py` (B4 + D1 walk-forward optional).
20. `scripts/Backtest_Analysis/backtest_analyzer.py` (D1 CSCV/PBO/DSR CLI).
21. `src/main_backtest.py` (Team A P7/P8 imports + B4 cost_model + D2 doc).

### Wave 6 — Tests + data follow-ups

22. `tests/portfolios/test_p7_lookahead.py` (D3).
23. `tests/integration/test_live_backtest_parity.py` (D4).
24. `src/orchestrator/backfill/update/refresh.py` (D2 Tier 1 history sidecar).

### Wave 7 — Out-of-band data backfill (not a code change)

25. Refresh `fundamentals/fundamentals.csv` on full P6 universe (Team A B1 blocker — currently 10/519 rows).
26. Backfill DBMF rows in `market_data` (Team C C1 blocker — DBMF must be in DB before live).
27. Verify `news_sentiment` table has FinBERT-pinned scores for the backtest window (D3 R1 — the FinBERT path bug means historical scores were generated by mutable Hub model).

## 6. Falsification gates (BLOCK promotion of P7/P8 from weight 0.0 → > 0.0)

| Gate | Criterion | Source |
|---|---|---|
| G1 (P0 ship) | `NLP/core/paths.py` MODEL_DIR points to existing local checkpoint | D3 |
| G2 (CI) | `tests/portfolios/test_p7_lookahead.py` all 6 cases pass | D3 |
| G3 (CI) | `tests/integration/test_live_backtest_parity.py` all 5 cases pass | D4 |
| G4 (data) | `fundamentals.csv` populated for ≥ 95% of P6 universe | A1 B1 |
| G5 (data) | DBMF present in `market_data` since 2019-05-08 | C1 |
| G6 (backtest) | Sharpe(P7) − Sharpe(P6) ≥ +0.15 on walk-forward purged-k-fold 2021-2026 | A2 |
| G7 (backtest) | MaxDD(P7) ≤ 1.10 · MaxDD(P6) over any 12-month rolling window | A2 |
| G8 (backtest) | DSR(P7) ≥ 0.5 across OOS years | A2 |
| G9 (backtest) | ΔIR(P8) ≥ +0.10 vs P6 averaged across purged-k-fold splits | A3 |
| G10 (backtest) | DSR(P8) ≥ 0.95 with corrected `n_trials` (universe + grid) | A3 |
| G11 (backtest) | RBI top-3 stable in ≥ 9/12 monthly refreshes | A3 |
| G12 (backtest) | CSCV PBO ≤ 0.5 on a ≥16-config strategy grid | D1 |
| G13 (audit) | DBMF rolling-36m correlation to P6 stock sleeve ∈ [−0.20, +0.30] | C1 |
| G14 (backtest) | Circuit breaker trips between −15% and −20% master DD on 2008 backtest | C3 |
| G15 (audit) | Cost model on P6 monthly rebal: turnover-weighted cost ∈ [5, 50] bps/yr | B4 |

## 7. Inter-team contract: vol-targeting

**Single-layer** lives at `src/portfolios/portfolio_6/screener.py::vol_target_scale`. **NO** other layer is allowed to measure `σ_target / σ_realized` and apply that as a multiplier. Documented at 4 sites:

1. `screener.py::vol_target_scale` docstring.
2. `strategy.py::_rebalance` inline comment near `vol_scale = ...`.
3. `daily_allocator.py::run_allocation` docstring.
4. `portfolio_manager_config.json::_invariants` JSON sibling key.

The circuit breaker (C3) and master allocator scale **gross sleeve capital** only. They never compute realized vol.

## 8. Inter-team contract: ex-ante features

Portfolio_7 sentiment fetch uses strict `published_at < context.time`. SQL filter operator is `<`, never `<=`. CI test (`tests/portfolios/test_p7_lookahead.py`) enforces this on every PR.

Market-data sentiment fallback is **disabled by default** (D3b) until `update_market_data_sentiment` aggregate-refresh is patched.

## 9. Known limitations (documented; not blocking ship)

| # | Limitation | Mitigation path |
|---|---|---|
| L1 | Universe is current-snapshot (no PIT membership). Backtests have +50 to +400 bps/yr survivorship inflation. | Tier 1 doc in main_backtest.py. Tier 2 PIT pipeline = follow-up PR. |
| L2 | `news_sentiment` historical rows generated by HuggingFace mutable `ProsusAI/finbert` due to path bug. | Fix is P0. Optional re-score on a window before P7 OOS gate. |
| L3 | `RBP/pipeline.py` does not return ticker labels with predictions. P8 has defensive fallback. | A3 §7.5 1-line patch in a separate PR before P8 promotion. |
| L4 | RBP `train_test_split_date` hardcoded in `RBP/config.py`. P8 backtests starting < 2023 leak future data. | P8 config knob exists; backtest runner must override at run-time. |
| L5 | `update_market_data_sentiment` rewrites historical `market_data.avg_sentiment` on every NLP cycle. | Fix in a follow-up PR. Until then, market_data fallback is off. |
| L6 | A1's profitability legs broken because `fundamentals.csv` has 10/519 rows + null `roe`. | Re-run `scripts/APIs/fmp_fundamentals.py` on full universe before enabling factor flags. |
| L7 | A5's IPO/microcap/short-interest/PEAD rules inert until FMP fetcher populates 4 new columns. | Follow-up PR per A5 §9. |

## 10. Risks summary

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Default-path regression after merging A1+A4+A5 unified `score_universe` | High | Tier 1: pure-Python unit test asserting `pd.Series.equals(old_score, new_score_with_default_args)` on fixture. |
| R2 | First post-merge backtest Sharpe drops vs old number (because realistic cost model) | Awareness | Document in PR. `CostModel.disabled()` restores old number. |
| R3 | DBMF data missing → hedge sleeve drops silently | High | C1 §8 backfill is a merge blocker. |
| R4 | Circuit breaker over-trips on noisy 2024 conditions | M | Hysteresis dwell = 10 days. Falsification gate validates on 2008. |
| R5 | CSCV gate requires ≥16 strategy variants — only 3 strategies today | M | Build `SCORE_WEIGHTS × WEIGHTING_METHOD × LAMBDA` permutation grid. |
| R6 | NLP path P0 fix lands; existing DB rows still carry Hub-FinBERT scores | M | Optional re-score on a backfill window. Document caveat in P7 backtest. |

## 11. Rollback matrix

| Concern | Rollback level 1 (config) | Rollback level 2 (code revert) |
|---|---|---|
| P7 misbehaves | `"7": 0.0` in manager config | `git rm src/portfolios/portfolio_7/` |
| P8 misbehaves | `"8": 0.0` in manager config | `git rm src/portfolios/portfolio_8/` |
| New score modes wrong | `"SCORE_METHOD": "rank_sum"` (default) | revert screener.py |
| HRP wrong | `"WEIGHTING_METHOD": "INV_VOL"` (default) | revert hrp_weights.py |
| Cost model wrong | `engine.setup(..., cost_model=CostModel.disabled())` | revert cost_model.py + executor patch |
| Circuit breaker wrong | `circuit_breaker.enabled = false` in manager config | revert daily_allocator.py |
| Trend hedge wrong | `"TREND_HEDGE_TICKER": ""` | swap to KMLM (single-line config) |
| Exclusions wrong | `EXCLUSIONS.*.ENABLED = false` (all already false by default) | revert exclusions.py |
| P0 path fix wrong | `git revert` (falls back to broken behavior) | restore mutable Hub-FinBERT |

## 12. Next steps

The plan is complete. Next phases:

1. **Apply** all diffs per §5 wave order.
2. **Run pytest smoke** — all existing tests must still pass with bit-exact defaults.
3. **Run new tests** — D3 + D4 must pass.
4. **Run backtest** — verify §6 G1–G15 gates one by one.
5. **Promote** P7 (and/or P8) by flipping its weight in `portfolio_manager_config.json` only if all gates pass.

Phases 1–3 are reversible config / code edits. Phase 4 is read-only research output. Phase 5 is a single JSON line per portfolio.

---

End of master synthesis. Apply phase awaits operator go-ahead.
