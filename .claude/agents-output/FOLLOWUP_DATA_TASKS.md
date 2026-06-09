# Follow-Up Data Tasks (Wave 7 — Operator Required)

**Date:** 2026-05-20
**Status:** Code apply complete. Three data tasks remain out-of-band.

These tasks require **FMP API keys** + **DB write access** + (for task 3) a **GPU/CPU box with FinBERT weights**. They are NOT part of the code PR. Run independently as separate batched jobs.

---

## Task 1 — Refresh `fundamentals.csv` on full P6 universe (Team A B1)

**Why:** Current `fundamentals/fundamentals.csv` has 10/519 ticker rows populated and `roe` is entirely null. A1's profitability rank fires on ~2% of universe → noise. Any A1 factor flag (`USE_MOMENTUM_12_2`, `USE_OPERATING_PROFITABILITY`, `USE_ASSET_GROWTH`) **must not be turned on** until this is fixed.

**Command:**
```bash
python3 scripts/APIs/fmp_fundamentals.py --tickers src/portfolios/portfolio_6/universe.json --output fundamentals/fundamentals.csv --full
```

**Verify:**
```bash
awk -F, 'NR>1 && $2!=""' fundamentals/fundamentals.csv | wc -l   # roe non-null rows >= 0.95 * 519
awk -F, 'NR>1 && $3!=""' fundamentals/fundamentals.csv | wc -l   # gross_profit non-null rows >= 0.95 * 519
```

**For A1 R2/R3 to fire**, additionally extend the fetcher to write columns: `revenue, cost_of_revenue, sga, interest_expense, book_equity, asset_growth`. Code in `score_universe` guards on `.issubset(cols)` so the legs simply no-op until the columns exist.

---

## Task 2 — Backfill DBMF in `market_data` (Team C C1)

**Why:** P6 config now sets `TREND_HEDGE_TICKER = "DBMF"`. If DBMF is missing from the `market_data` table, the 10% sleeve silently drops and stock book + GLD absorb the slack.

**Verify presence:**
```sql
SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM market_data WHERE ticker = 'DBMF';
-- Expect: MIN >= 2019-05-08 (DBMF inception), COUNT >= 1700
```

**Backfill if missing:**
```bash
python3 src/orchestrator/backfill/specific_backfill.py --ticker DBMF --start 2019-05-08
```

**Add to live ingestor:**
```bash
# Edit src/orchestrator/realTime/realtimeDataIngestor.py polling list to include DBMF.
```

**Note:** Backtests with `START_DATE < 2019-05-08` MUST override `TREND_HEDGE_TICKER = ""` to disable the sleeve, or DBMF data will be missing for the early window.

---

## Task 3 — Re-score `news_sentiment` with pinned FinBERT (Team D D3)

**Why:** The NLP path bug (`NLP/core/paths.py`) was just fixed. Historical `news_sentiment` rows already in the DB were scored by the mutable Hugging Face Hub `ProsusAI/finbert` model. The local pinned checkpoint at `NLP/finbert-finetuned-final/` may produce slightly different scores. For P7 OOS gate accuracy, re-score the backtest window.

**Recommended approach:** Add a `force_rescore=True` flag to the FinBERT scorer + a backfill script. Per-article scores are sticky (ON CONFLICT DO NOTHING per `repository.py`), so a forced re-score must DELETE the old row first.

**Quick estimate of impact:**
```sql
-- Count articles scored before vs after the NLP/core/paths.py fix.
SELECT
  DATE_TRUNC('month', created_at) AS month,
  COUNT(*) AS rows
FROM news_sentiment
GROUP BY 1
ORDER BY 1;
```

**Less risky alternative:** Document the caveat in P7 backtest reports. The FinBERT score is bounded in [-1, 1], so the score drift between Hub and local versions is typically < 0.1 absolute. Cross-sectional ranking matters more than absolute level; the P7 z-score is invariant to a constant shift.

---

## Task 4 (optional) — Run the falsification gates

Once the code PR is merged + Tasks 1–3 complete:

```bash
# G2 / G3: CI tests
pytest tests/portfolios/test_p7_lookahead.py tests/integration/test_live_backtest_parity.py -v

# G6–G10: walk-forward purged-k-fold backtest on a strategy grid
# (Build the grid: SCORE_WEIGHTS x WEIGHTING_METHOD x SENTIMENT_TILT_LAMBDA permutations, >= 16 variants)
# Pipe outputs to CSCV/PBO via backtest_analyzer (D1).
python3 -m src.main_backtest    # produces fast-mode + event-mode results
python3 scripts/Backtest_Analysis/backtest_analyzer.py --cscv-portfolios 16 --gate-pbo 0.5
```

Promote P7 / P8 (capital weight 0 → > 0) only after **all** gates G1–G15 pass.

---

## Quick status summary

| # | Task | Owner | Status |
|---|------|-------|--------|
| 1 | Refresh fundamentals.csv full universe | Data ops | TODO |
| 2 | Backfill DBMF market_data | Data ops | TODO |
| 3 | Re-score news_sentiment with pinned FinBERT | ML ops | OPTIONAL (caveat doc OK) |
| 4 | Run falsification gates G1-G15 | Quant ops | TODO post-merge |

Code PR ships independently. These tasks unblock P7/P8 capital allocation but do not block code merge.
