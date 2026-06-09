# Team D Synthesis — Validation & Data

**Date:** 2026-05-20
**Inputs:** D1 (CSCV/PBO/purged-kfold), D2 (PIT data audit), D3 (NLP look-ahead audit), D4 (live-vs-backtest parity).
**Status:** Read-only synthesis. Apply-ready diffs in each sub-deliverable.

---

## 1. Decisions

| # | Decision | Source | Priority |
|---|---|---|---|
| D1 | Add CSCV/PBO + purged-k-fold infrastructure. NEW `src/backtest/cscv.py` + `src/backtest/purged_kfold.py`. Patch `backtest_analyzer.py` with `--cscv-portfolios`, `--dsr-n-trials`, `--gate-pbo 0.5`. PBO > 0.5 blocks P7/P8 promotion. | D1 | P1 |
| D2 | Survivorship bias confirmed real. Ship **Tier 1** minimum-viable (1 doc comment + 1 sidecar history JSON in `refresh.py`) for v1. **Defer Tier 2** full PIT pipeline (new file + DB table + `--as-of` flag) to a follow-up PR. | D2 | P2 |
| D3 | **1 VIOLATION found**: FinBERT `MODEL_DIR` in `NLP/core/paths.py:15` points to non-existent `finbert-combined-final/` → silent fallback to mutable Hugging Face Hub. **Fix immediately**. 1-line diff. | D3 | P0 BLOCKER |
| D3b | **Recommended default change**: flip P7 `SENTIMENT_FALLBACK_TO_MARKET_DATA` from `true` → `false` until `update_market_data_sentiment` aggregate-refresh is fixed (currently rewrites historical rows on every NLP cycle = non-PIT). | D3 | P1 |
| D4 | P6 `_rebalance` is parity-safe per audit. Add CI integration test `tests/integration/test_live_backtest_parity.py` to lock in the invariant against future regressions. | D4 | P1 |

## 2. New files (total: 3 + 2 tests)

| Path | Purpose | Source |
|---|---|---|
| `src/backtest/cscv.py` | `cscv_pbo(returns_per_strategy, *, S=16) → {pbo, performance_degradation, ...}` per Bailey-LdP 2014. | D1 |
| `src/backtest/purged_kfold.py` | `PurgedKFold(n_splits, t1, embargo_td)` per AFML Ch. 7. | D1 |
| `tests/portfolios/test_p7_lookahead.py` | 6-test pytest enforcing strict-`<` cutoff on `_fetch_sentiment_ex_ante`, MODEL_DIR existence, model-hash pin, no-restate invariant on `news_sentiment` ON CONFLICT DO NOTHING, market_data fallback gate. | D3 |
| `tests/integration/test_live_backtest_parity.py` | 5-case pytest asserting `_target_weights` bit-identical across backtest and live modes on identical input. | D4 |

(No new files for D2 Tier 1; if Tier 2 lands later, adds `src/orchestrator/backfill/historical_constituents.py` + `index_membership` table.)

## 3. Patches to existing files

### 3.1 `NLP/core/paths.py` — P0 BLOCKER fix

```diff
--- a/NLP/core/paths.py
+++ b/NLP/core/paths.py
@@ -12,5 +12,5 @@ ROOT = Path(__file__).resolve().parents[2]
-MODEL_DIR = ROOT / "finbert-combined-final"
+MODEL_DIR = ROOT / "finbert-finetuned-final"
```

(Exact path: see D3 §6. Verify actual on-disk folder name with `ls` before applying.)

### 3.2 `src/portfolios/portfolio_7/config.json` — D3b default flip

```diff
-    "SENTIMENT_FALLBACK_TO_MARKET_DATA": true
+    "SENTIMENT_FALLBACK_TO_MARKET_DATA": false
```

Per-article path (`news_sentiment` SELECT with strict `<`) is PIT-safe (D3 V1 SAFE verdict). `market_data.avg_sentiment` aggregate is **not** PIT-safe today because `update_market_data_sentiment` rewrites historical rows on every NLP cycle. Once that aggregate-refresh is patched in a follow-up PR (D3 §7 recommended fix), flip back to `true`.

### 3.3 `src/backtest/backtest_engine.py` — wire purged-k-fold + CSCV

Adds optional `walk_forward` config block. Default = legacy single-shot. See D1 §6.

### 3.4 `scripts/Backtest_Analysis/backtest_analyzer.py` — PBO + DSR in summary

D1 §7 unified diff adds three CLI flags:
- `--cscv-portfolios <N>` (default 16) — number of strategy variants for CSCV.
- `--dsr-n-trials <N>` (default derived) — n_trials for `deflated_sharpe_ratio`.
- `--gate-pbo <FLOAT>` (default 0.5) — exit code 2 if PBO exceeds threshold.

### 3.5 `src/main_backtest.py` — D2 Tier 1 doc comment

Document survivorship-bias limitation in header comment + cite literature (BGIR 1992, AnalyticalPlatform 145 bps S&P 500). Per D2.

### 3.6 `src/orchestrator/backfill/update/refresh.py` — D2 Tier 1 sidecar

Append history to `_history.json` when universe changes (no behavior change, just diff log). Per D2.

## 4. Apply order (single PR, after Teams A/B/C land)

1. **PATCH** `NLP/core/paths.py` (D3 P0 fix). — fastest, most urgent.
2. **PATCH** `src/portfolios/portfolio_7/config.json` (D3b default flip).
3. **NEW** `src/backtest/cscv.py`, `src/backtest/purged_kfold.py` (D1).
4. **PATCH** `src/backtest/backtest_engine.py` (D1 walk-forward).
5. **PATCH** `scripts/Backtest_Analysis/backtest_analyzer.py` (D1 CSCV/DSR/PBO CLI).
6. **PATCH** `src/main_backtest.py` (D2 Tier 1 comment + cite literature).
7. **PATCH** `src/orchestrator/backfill/update/refresh.py` (D2 Tier 1 history sidecar).
8. **NEW** `tests/portfolios/test_p7_lookahead.py` (D3).
9. **NEW** `tests/integration/test_live_backtest_parity.py` (D4).

## 5. Falsification gates (PRE-MERGE blocker for P7/P8 promotion)

| Gate | Criterion | Owner |
|---|---|---|
| F1 (D1) | Build ≥16-config strategy grid; CSCV PBO ≤ 0.5. | P7/P8 |
| F2 (D1) | Deflated Sharpe ≥ 0.5 with corrected `n_trials` (universe size × grid cells). | P7/P8 |
| F3 (D1) | Purged-k-fold OOS Sharpe(P7) − Sharpe(P6) ≥ 0.15 (matches A2 falsification). | P7 |
| F4 (D1) | Purged-k-fold OOS ΔIR(P8) ≥ +0.10 vs P6 (matches A3 falsification). | P8 |
| F5 (D2) | Tier 1 survivorship-bias doc shipped + flagged in main_backtest header. Tier 2 promotion blocked behind PIT pipeline (separate PR). | All |
| F6 (D3) | `tests/portfolios/test_p7_lookahead.py` all 6 cases green in CI. | P7 |
| F7 (D4) | `tests/integration/test_live_backtest_parity.py` all 5 cases green in CI. | All portfolios |

## 6. Inter-team handoffs / corrections

| Item | Affects |
|---|---|
| D3 P0 fix to `NLP/core/paths.py` | Team A A2 (P7 was assuming local FinBERT but the path bug made it use the Hub version) — `SENTIMENT_FALLBACK_TO_MARKET_DATA` default flip downstream. |
| D3 `news_sentiment` ON CONFLICT DO NOTHING is sticky → per-article path is PIT | Team A A2 (P7 design is correct, just disable the market_data fallback). |
| D1 PBO gate added to backtest_analyzer | Team A SYNTHESIS §11 promotion gates now reference PBO ≤ 0.5 as additional precondition. |
| D2 survivorship bias acknowledged in code header | Team A SYNTHESIS §10 B7 closed (Tier 1) + reopened as follow-up (Tier 2). |
| D4 parity test enforces no-RNG / no-wall-clock invariant | Team C C3 circuit breaker MUST also be parity-safe (no `datetime.now()` in `update_state`); verify when wiring. |

## 7. Risks summary

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | NLP FinBERT path bug means historical sentiment scores in DB were generated by mutable HuggingFace `ProsusAI/finbert`, not pinned local checkpoint | High | D3 fix is P0 BLOCKER. Existing `news_sentiment` rows are stuck on whatever the Hub returned at scrape time; future scoring will use the local checkpoint after the path fix. Recommend re-scoring on a backfill window before P7 OOS gate. |
| R2 | Survivorship-bias inflation (~50-400 bps/yr CAGR) infects all current backtests | High | D2 Tier 1 documents the limitation. Tier 2 PIT pipeline is a follow-up PR. ALL Sharpe / Calmar metrics produced before Tier 2 carry an asterisk. |
| R3 | CSCV requires ≥16 strategy variants. P6+P7+P8 only = 3. | M | D1 documents the workaround: build a config grid (`SCORE_WEIGHTS`, `WEIGHTING_METHOD`, `SENTIMENT_TILT_LAMBDA`) of ≥16 permutations and feed to `cscv_pbo`. |
| R4 | `market_data.avg_sentiment` aggregate refresh is non-PIT today | M | D3b default flip to `false`. Fix in follow-up PR before re-enabling. |
| R5 | New `test_live_backtest_parity.py` breaks if anyone adds RNG/wall-clock | L | Test fails CI immediately — early signal, easy fix. |

## 8. Rollback paths

| Concern | Rollback |
|---|---|
| FinBERT path fix breaks scoring | `git revert` — falls back to current broken-but-running path. |
| Market-data sentiment fallback wanted again | `"SENTIMENT_FALLBACK_TO_MARKET_DATA": true` (single config line). |
| CSCV/PBO gate too strict | Disable via `--gate-pbo 1.0` in `backtest_analyzer.py` invocation. |
| New tests fail | `pytest -k 'not test_p7_lookahead and not test_live_backtest_parity'` to skip. |

---

End of Team D synthesis. **P0 BLOCKER**: apply `NLP/core/paths.py` fix first — affects all of Portfolio_7's sentiment pipeline.
