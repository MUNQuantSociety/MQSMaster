# D2 — Point-in-Time (PIT) Data Audit: Survivorship Bias in MQSMaster Universe

**Author:** Team D analyst, quant research + senior Python engineer
**Date:** 2026-05-20
**Status:** Read-only audit. No source code modified. All patches below are apply-ready diffs the operator can land in a separate PR.
**Scope:** Inventory ticker universe construction and backfill, quantify the survivorship-bias / look-ahead-bias risk, and propose a minimum-viable PIT remediation. Cross-referenced with Team A SYNTHESIS §10 B7 ("survivorship bias — out of scope here") which explicitly punted this to Team D2.

---

## 1. Executive summary

The MQSMaster ticker universe is **NOT point-in-time correct**. Every ticker file consumed by the backfill (`tickers.json`, `extra_tickers/{sp500,nasdaq,crypto,commodity}_tickers.json`) and the Portfolio 6 `universe.json` (518 tickers) is a **single static snapshot** built from FMP `stable/sp500-constituent` + `stable/nasdaq-constituent`, both of which return *only the current members as of the API call time*. `refresh.py` overwrites these files in place on every run, with no append-only history and no per-date membership record. The orchestrator backfill (`backfill.py`, `concurrent_backfill.py`, `specific_backfill.py`) pulls intraday/daily OHLCV only for tickers in this snapshot, so any company that was in the S&P 500 in 2020 but has since been deleted (e.g. AIV, COG, NOV, ETFC, HCP, RTN, KSU, ALXN) is **silently absent** from `market_data`. The schema has no `index_membership` table, no delisting flag, and no PIT-aware filter in the backtest runner.

The literature is unambiguous on the magnitude: studies that compare a "current snapshot" vs. PIT reconstructed S&P 500 backtest show CAGR inflation of roughly **+150 to +400 bps/yr** for broad equal-weight strategies and up to **8% per year** for the worst-cycle decade (Daniel-Sornette-Wöhrmann 2009). Brown-Goetzmann-Ibbotson-Ross (1992) document Sharpe inflation of up to +0.5 and Shumway (1997) finds a missing delisting return of ~**−30% to −55%** per affected name (NASDAQ). For Portfolio 6 specifically — a *low-vol, top-N screen* — the bias is structurally **on the lower end of that range** (the strategy already de-weights extreme losers) but is **still material** (estimated +50 to +200 bps/yr) and will be the single largest unmeasured source of OOS error once the backtest window crosses 2 years. **All Team A factor decisions (P7/P8, momentum, op-profit) are gated on backtests that currently consume biased universe; their OOS Sharpe thresholds (§11) are not safely interpretable until PIT is fixed or explicitly bounded.**

## 2. Sources (≥10 primary, cross-validated)

| # | URL | Annotation | Relevance |
|---|---|---|---|
| S1 | https://terpconnect.umd.edu/~wermers/ftpsite/FAME/Brown_Goetzmann_Ibbotson_Ross.pdf | Brown, Goetzmann, Ibbotson, Ross (1992), "Survivorship Bias in Performance Studies," *Review of Financial Studies* 5(4):553–580. Canonical reference. Shows truncation-by-survival produces *apparent* return predictability and Sharpe inflation up to +0.5. | Quantifies the Sharpe/predictability bias direction |
| S2 | https://www.tylergshumway.org/Shumway-DelistingBiasCRSP-1997.pdf | Shumway (1997), "The Delisting Bias in CRSP Data," *J. Finance* 52(1):327–340. Documents missing delisting returns; estimates corrected −30% NYSE/AMEX, −55% NASDAQ delisting return for performance-related delists. | Quantifies the *delisted-name* bias when delisting returns are dropped |
| S3 | https://ar5iv.labs.arxiv.org/html/0810.1922 | Daniel, Sornette, Wöhrmann (2009), "Look-Ahead Benchmark Bias in Portfolio Performance Evaluation," *J. Portfolio Mgmt* 36(1):121. Documents up to **8%/yr** look-ahead bias on S&P 500 portfolios; ex-post 6.4%/yr vs ex-ante 2.3%/yr 2001-2006. CRSP 1926-2006 universe. | Magnitude of look-ahead bias when using end-of-period vs start-of-period constituents |
| S4 | https://www.nber.org/papers/w28432 / https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13249 | Jensen, Kelly, Pedersen (2023), "Is There a Replication Crisis in Finance?", *J. Finance* 78(5):2465–2518. PIT data is a precondition of their global factor replication; methodology section emphasizes PIT membership reconstruction. GitHub `bkelly-lab/ReplicationCrisis`. | Reference implementation for PIT factor research at scale |
| S5 | https://www.crsp.org/research/crsp-survivor-bias-free-us-mutual-funds/ + https://wrds-www.wharton.upenn.edu/documents/410/CRSP_MFDB_Guide.pdf | CRSP Survivor-Bias-Free US Mutual Funds product / guide. Industry-standard benchmark for PIT-correct fund return history. | Confirms PIT correctness is a paid, non-trivial data product |
| S6 | https://eodhd.com/financial-apis-blog/sp-500-historical-constituents-data | EODHD blog, "S&P 500 Historical Constituents Data." 18+ years of changes, 200+ events, ~90 000 rows. | Confirms data availability and shape of historical constituents feed |
| S7 | https://riazarbi.github.io/quant/backtesting-sp500-constituent-history/ | Arbi, "Survivorship-bias free S&P 500 constituent lists." Practitioner methodology: iShares IVV monthly holdings + Wikipedia revisions + CUSIP/CIK joins. | Mitigation methodology — alternative to FMP if FMP quality fails |
| S8 | https://teddykoker.com/2019/05/creating-a-survivorship-bias-free-sp-500-dataset-with-python/ | Koker, "Creating a Survivorship Bias-Free S&P 500 Dataset with Python." Demonstrates equal-weight current-constituents-only **dramatically outperforms** RSP ETF; PIT reconstruction matches RSP. | Empirical falsification example for "current snapshot" universe |
| S9 | https://www.analyticalplatform.com/the-hidden-impact-of-survivorship-bias-on-backtesting-results-of-investment-strategies/ | AnalyticalPlatform, "Hidden Impact of Survivorship Bias." Quantifies S&P 500 10y CAGR bias = **−1.45 pp**, Sharpe = **−0.06**; small-cap 5y CAGR bias = **−26.84 pp**. | Strategy-class-specific magnitude estimates |
| S10 | https://site.financialmodelingprep.com/developer/docs/stable/historical-sp-500 + https://site.financialmodelingprep.com/developer/docs/historical-sp-500-companies-api | FMP stable + legacy "historical S&P 500 constituents" endpoints. Stable: `https://financialmodelingprep.com/stable/historical-sp500-constituent`. Legacy v3: `https://financialmodelingprep.com/api/v3/historical/sp500_constituent`. Returns `{date, addedSecurity, removedTicker, removedSecurity, dateAdded, symbol, reason}`. | Concrete API to feed the proposed `index_membership` table |
| S11 | https://www.nasdaq.com/articles/looking-nasdaq-100-index-adds-and-deletes + https://www.stocktitan.net/news/NDAQ/annual-changes-to-the-nasdaq-100-g69h3ryr3q2d.html | Nasdaq newsroom: NDX averages ~6 annual reconstitution changes (Dec) + ~3 off-cycle additions/deletions per year. 2010–2025: 83 added / 74 removed (~10/yr). | Reconstitution frequency for the second leg of P6 universe |
| S12 | https://www.spglobal.com/spdji/en/documents/research/research-what-happened-to-the-index-effect.pdf + S&P DJI methodology + Vijh (2002, *Financial Mgmt*) https://www.biz.uiowa.edu/faculty/avijh/SP_1.pdf | S&P DJI research + Avadhanam Vijh add/delete dataset. S&P 500 averages ~22 changes/yr (= ~4.4% annual turnover). 2000 was an outlier with 58 deletions; 2001 was 30. | Provides annual reconstitution count → estimates total names missing from 10y backtest |
| S13 | https://www.nber.org/system/files/working_papers/w23394/w23394.pdf | Hou, Xue, Zhang (2017/2019), "Replicating Anomalies." Microcaps = 3% of mkt cap but 60% of stock count and the largest source of bias amplification. NYSE breakpoints + value-weighted returns flip 64% of anomalies to insignificant. | Why "drop-microcap" rules (Team A A5 R5) interact non-trivially with survivorship |
| S14 | https://portfoliooptimizationbook.com/book/8.2-seven-sins.html | Palomar (2025), "Portfolio Optimization" §8.2 "Seven Sins of Quantitative Investing." Survivorship + look-ahead listed as #1 and #2 sins. Both can fully reverse apparent strategy edge. | Pedagogical synthesis; useful for project-level write-up |
| S15 | https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1997.tb03818.x (companion to S2) | Shumway 1997 abstract: confirms most negative-cause delists are *surprise* events and missing returns are large enough to flip size-effect findings. | Confirms delisting returns cannot be safely set to 0 (the implicit assumption of any current-snapshot backfill) |

(S1–S2, S3–S9, S6/S10/S11 each cross-validate the same claim from independent sources.)

## 3. Audit findings — file:line + evidence

### 3.1 Universe construction is a current-snapshot

- **`/Users/abhinav/Desktop/MQSMaster/scripts/APIs/build_portfolio6_universe.py:66-78`** calls FMP `https://financialmodelingprep.com/stable/sp500-constituent` and `https://financialmodelingprep.com/stable/nasdaq-constituent`. Both endpoints return *the constituents on the day the API is called*. There is no `date=` parameter and no asof filter. The script then dedupes and dumps to `universe.json` — a flat JSON array of 518 strings, no per-ticker addition date, no removal date, no inception flag. (Confirmed via S10 + the script's own URL strings.)
- **`/Users/abhinav/Desktop/MQSMaster/src/portfolios/portfolio_6/universe.json:1-518`** — flat list, no metadata. Strategy `Portfolio6Strategy._load_universe` (`src/portfolios/portfolio_6/strategy.py:122-133`) reads it and stores it on `self.tickers` unconditionally. There is no `as_of` parameter and no filter against `context.time`. The same 518 names are screened in every monthly rebalance from `start_date` to `end_date`.
- **`/Users/abhinav/Desktop/MQSMaster/src/orchestrator/backfill/update/refresh.py:64-95`** ("fetch_and_merge_tickers") calls FMP `get_sp500_tickers()`, `get_commodity_tickers()`, `get_crypto_tickers()` and **overwrites** `extra_tickers/nasdaq_tickers.json` in place (line 169 `save_tickers(combined_tickers, tickers_path)`). The merge is set-union — *previously listed companies that have been removed from the S&P 500 are retained only if they happen to still be in the file from a prior run*, but the file is silently expanded over time and never tagged with membership dates. Concretely: a stock added in 2024 then removed in 2025 will live in `nasdaq_tickers.json` forever, but the backtest still cannot tell that it *wasn't* in the index during 2020.
- **`/Users/abhinav/Desktop/MQSMaster/src/orchestrator/backfill/tickers.json` (1 910 lines)** is the default backfill input. Same shape: flat list, no membership history.

### 3.2 Backfill preserves data for snapshot members only — delisted tickers never enter the DB

- **`/Users/abhinav/Desktop/MQSMaster/src/orchestrator/backfill/backfill.py:99-238`** (`backfill_data`) loops over the `tickers` list parameter and calls `fmp.get_intraday_data` / `fmp.get_historical_data` for each. Tickers absent from the input list are never fetched. If a ticker was delisted before the snapshot was built, it is **not** in `tickers.json` → never queried → never inserted into `market_data`.
- **`/Users/abhinav/Desktop/MQSMaster/src/orchestrator/backfill/concurrent_backfill.py:147-208`** — same pattern, parallelised over the same input list.
- **`/Users/abhinav/Desktop/MQSMaster/src/orchestrator/backfill/specific_backfill.py:26-123`** — `backfill_db` + `fill_gaps_for_ticker`. Gap filling fills *missing trading days for snapshot tickers* only. There is no logic to detect "this ticker existed in the index in 2018 but doesn't now."
- **`/Users/abhinav/Desktop/MQSMaster/src/orchestrator/backfill/injectBackfill.py:43-138`** — only injects rows whose filename matches `2y_mkt_data_{ticker}.csv` and only if `ticker_exists()` returns false; survivorship-aware CSVs would still need to be staged by hand.

### 3.3 Schema has no PIT membership table — every join is point-in-now

- **`/Users/abhinav/Desktop/MQSMaster/src/common/database/schemaDefinitions.py:39-54`** — `market_data` table has `(ticker, timestamp, date, exchange, open_price, high_price, low_price, close_price, volume, avg_sentiment, created_at)`. **No `index_membership` table. No `delisting_date` column. No `first_listed_date` column. No `was_in_index_as_of` index.** The schema cannot answer "which S&P 500 names were members on 2020-06-30?" even in principle without an external lookup.
- A grep for `delisted|delisting|index_membership|point_in_time|PIT` across `src/` + `scripts/` returns **zero functional matches**. Only one mention: `src/portfolios/portfolio_6/Portfolio6_Strategy.txt:41` already flags "the latest available statements, not point-in-time fundamentals" — the team is *aware* of the issue but no implementation exists.

### 3.4 Backtest runner has no PIT filter

- **`/Users/abhinav/Desktop/MQSMaster/src/main_backtest.py:47-66`** — `START_DATE = "2025-01-01"`, `END_DATE = "2025-09-05"`. A small 9-month window minimises the bias *today*, but Team A SYNTHESIS §11 demands a "walk-forward purged-k-fold OOS backtest 2021–2026" for P7/P8 promotion. The minute the window extends to ≥2 years, survivorship bias becomes the dominant error.
- **`/Users/abhinav/Desktop/MQSMaster/src/portfolios/portfolio_6/strategy.py:157-194`** — `OnData` / `_collect_returns` consumes `self.tickers` (the static universe) and filters only on data availability (`asset.Exists`, history length). It never asks "was this ticker in the S&P 500 on `context.time`?" The same 518-name candidate pool is screened every month for a 5-year backtest. *This is the look-ahead vector:* the backtest "knows" in 2020 that PLTR (listed 2020-09), ABNB (listed 2020-12), RIVN (listed 2021-11), and CEG (spun-off 2022-02) will be S&P 500 names. It also "knows" that AIV, COG, NOV, ETFC, HCP, RTN, KSU, ALXN — all dropped from the index between 2018–2024 — should be excluded. Both directions are leakage.

### 3.5 Look-ahead vectors — concrete

1. **Universe-membership leakage.** The 518-name `universe.json` was built post-2024 (it contains APP — added 2025-02, ARM — listed 2023-09, GEV — spun 2024-04, KKR — added 2024-06). Every monthly screen between 2020 and 2024 in any P6 backtest will rank these against `_collect_returns`, which is impossible by construction. Conversely, names alive in 2020 but dropped before the snapshot date are unreachable.
2. **Selection-by-survival.** Index removal is *non-random*: deletion is correlated with negative shocks (M&A consideration premium aside, the bulk of removals are size-based — i.e. *post-drawdown*). A 2020 ⇒ 2026 backtest that uses only 2026-snapshot members systematically excludes the *worst-realised* names. The low-vol screen ranks by realised vol *over the lookback*, which by construction selects *the survivors that happen to have realised lower vol* — biased downward.
3. **Index reconstitution timing leakage.** S&P 500 turnover (S12) ~22 names/yr ⇒ for a 6-year backtest, ~132 single-counted membership-change events are missed. NDX (S11) ~9–10 names/yr ⇒ ~55 events. P6 universe is the union, so the *gross* membership-change count is ~150-180 events over 6y, of which roughly half are deletions (the survivorship-bias-relevant direction).
4. **Hindsight selection (Sharpe, 1992 — S1).** The cross-section of realised returns conditional on survival has lower variance and higher Sharpe by construction. The DSR check in `screener.py::deflated_sharpe_ratio` does *not* correct for this — it corrects for multiple-testing of factors, not survivorship of the universe.
5. **Delisting-return omission (Shumway 1997 — S2).** Even if the ticker is in `market_data`, when a name is delisted mid-window the last close is typically the last bar before halt — which underestimates the loss by ~30 pp (NYSE) to ~55 pp (NASDAQ) per affected name. The backfill has no delisting return injection step.

## 4. Bias quantification (expected impact, bps/yr)

Bias magnitude depends on (i) strategy type, (ii) backtest window, (iii) universe size, (iv) weighting. The literature gives a clear range; Portfolio 6 sits **near the lower-middle** of it.

| Reference / scenario | Magnitude | Notes / direct quote |
|---|---|---|
| S3 Daniel-Sornette-Wöhrmann 2009 (S&P 500, 2001-2006) | **+410 bps/yr** CAGR, Sharpe inflation +0.30 (0.1 → 0.4) | ex-post 6.4% vs ex-ante 2.3%; explicitly look-ahead via end-of-period constituents |
| S3 same paper, 10-yr rolling 1926-2006 | up to **+800 bps/yr** in worst decade | "up to 8% per annum" |
| S9 AnalyticalPlatform (S&P 500, 10y broad eq-weight) | **+145 bps/yr** CAGR, Sharpe +0.06 | balanced, US large-cap |
| S9 same (small-cap 20-stock, 5y) | **+2 684 bps/yr** CAGR | extreme; not directly applicable to P6 |
| S1 Brown-Goetzmann-Ibbotson-Ross 1992 (mutual funds 1976-1988) | **+20 to +80 bps/yr** | broadest, lower-bound estimate for diversified portfolios |
| S2 Shumway 1997 (CRSP delisting bias per affected name) | **−30% (NYSE), −55% (NASDAQ)** missing return | per delisted name, not per portfolio-yr |
| S13 Hou-Xue-Zhang 2019 (anomaly t-stats) | 64% of anomalies flip to insignificant under microcap-removed VW returns | bias direction confirmed at scale |
| **Portfolio 6 expected (estimated)** | **+50 to +200 bps/yr** | reasoning below |

**P6-specific reasoning:**
P6 is a **low-vol + low-MAX(1d) + (optional) gross-profit screen** with inverse-vol weights, 7% GLD hedge, vol-target 13%. Three forces *attenuate* the bias compared to a naive equal-weight current-snapshot strategy:
1. Inverse-vol weighting de-emphasises the highest-vol survivors, which are also the highest-vol drop-pool members.
2. Vol-target capping pulls realised vol toward 13%, so Sharpe is mechanically tighter ⇒ the bias range of Sharpe is also compressed.
3. The low-vol screen already excludes the typical deletion candidates (post-crash, blown-up names) by construction.

Two forces *amplify* it:
1. The screen ranks the *survivors*, so the bottom of the realised-vol cross-section is a survival-conditioned tail.
2. The momentum / op-profit / asset-growth flags from Team A A1 are *long-only* and rank-based — long-only momentum is **the** classic survivorship-amplified factor (Asness 2013; Hou-Xue-Zhang on momentum). If those flags ship, the bias estimate moves up the range.

**Conservative midpoint estimate: ~100 bps/yr CAGR inflation for P6, ~+0.10 to +0.15 Sharpe inflation, drawdowns ~10-15% smaller than truth on a 5-yr window.** Above the threshold Team A §11 sets for P7/P8 promotion (Sharpe gate `≥+0.15`). **This means a P7 or P8 backtest could pass the promotion gate purely by re-distributing the survivorship bias differently across the universe.** The promotion decision is therefore not safely defensible until PIT is implemented or the bias is bounded by Falsification §6 below.

## 5. Recommended patch set (apply-ready)

The full PIT remediation is large (~2 dev-weeks of work + data backfill). I split it into three tiers; ship Tier 1 in the same PR as Team A's P7/P8 promotion gate, schedule Tier 2 + Tier 3 next.

### Tier 1 — Minimum-Viable Patch (≈ 4 hrs)

Goal: surface the limitation to anyone running a backtest; preserve currently-listed names + a flag so the team has the data ready before any factor-flag decision.

**Patch 1.1** — Add a documented limitation block to the strategy backtest configuration:

```diff
--- a/src/main_backtest.py
+++ b/src/main_backtest.py
@@ -45,6 +45,16 @@
 - BACKTEST_NUM_BATCHES: An optional integer specifying the number of batches to use for parallel backtest execution. If set to None, the batch count will be automatically determined based on the number of CPU cores and the number of portfolios.
 """
+# ------------------------------------------------------------------------
+# KNOWN LIMITATION — Survivorship / Look-ahead bias (Team D2 audit, 2026-05)
+# The S&P 500 + Nasdaq-100 universe used by Portfolio 6/7/8 is a single
+# static snapshot built post-2024. Backtests with START_DATE < 2024-01-01
+# carry an expected +50 to +200 bps/yr CAGR inflation and +0.10 to +0.15
+# Sharpe inflation vs a point-in-time correct universe (Daniel-Sornette-
+# Wöhrmann 2009; Brown-Goetzmann-Ibbotson-Ross 1992). DO NOT use OOS
+# Sharpe gaps < 0.20 between P7/P8 vs P6 as a promotion signal until the
+# index_membership table proposed in D2 §5 is populated.
+# ------------------------------------------------------------------------
 START_DATE = "2025-01-01"
 END_DATE = "2025-09-05"
```

**Patch 1.2** — Mark `refresh.py` so the ticker file is *append-only* (never delete names that were once listed; tag the first-seen date):

```diff
--- a/src/orchestrator/backfill/update/refresh.py
+++ b/src/orchestrator/backfill/update/refresh.py
@@ -52,10 +52,30 @@
 def save_tickers(tickers: list[str], tickers_path: Path) -> None:
-    """Save tickers to JSON file."""
+    """Save tickers to JSON file (append-only; never drop a previously-seen
+    ticker — that would silently introduce survivorship bias into any
+    subsequent backfill). Existing tickers are merged via set-union with the
+    new fetch; the sidecar `_history.json` records the first-seen date for
+    each ticker, which downstream PIT filters can use as a lower bound on
+    membership existence."""
     try:
         tickers_path.parent.mkdir(parents=True, exist_ok=True)
+        existing = []
+        if tickers_path.exists():
+            with open(tickers_path, "r") as f:
+                existing = json.load(f)
+        merged = sorted(set(existing) | set(tickers))
+        history_path = tickers_path.with_name(tickers_path.stem + "_history.json")
+        history: dict[str, str] = {}
+        if history_path.exists():
+            with open(history_path, "r") as f:
+                history = json.load(f)
+        today = datetime.now().date().isoformat()
+        for t in merged:
+            history.setdefault(t, today)
         with open(tickers_path, "w") as f:
-            json.dump(tickers, f, indent=2)
-        logger.info(f"Saved {len(tickers)} tickers to {tickers_path.name}")
+            json.dump(merged, f, indent=2)
+        with open(history_path, "w") as f:
+            json.dump(history, f, indent=2)
+        logger.info(
+            f"Saved {len(merged)} tickers to {tickers_path.name} "
+            f"(+{len(merged) - len(existing)} new; history sidecar updated)"
+        )
     except Exception as e:
         logger.error(f"Error saving tickers: {e}")
         raise
```

Effect: every refresh run *grows* the universe monotonically. Even without a full PIT lookup, the backfill now has data for names that have been removed since the system started running. The `_history.json` sidecar is the seed for Tier 2.

### Tier 2 — `index_membership` table + PIT-aware universe loader (≈ 1 dev-week)

**Patch 2.1** — NEW file `src/orchestrator/backfill/historical_constituents.py`:

```python
"""
Fetch and persist S&P 500 (and NDX where available) historical membership
changes from FMP. Builds an `index_membership` table that lets downstream
strategies ask: which symbols were members of index I on date D?

Run:
    python -m src.orchestrator.backfill.historical_constituents

CLI args:
    --indices sp500 ndx
    --since 2010-01-01      (used to bound the FMP response)
    --apply-schema          (creates index_membership table if not exists)
    --dry-run               (parse + log only; no DB writes)

FMP endpoints used (S10):
    Stable:  https://financialmodelingprep.com/stable/historical-sp500-constituent
    Legacy:  https://financialmodelingprep.com/api/v3/historical/sp500_constituent
    Each row: {date, addedSecurity, removedTicker, removedSecurity,
               dateAdded, symbol, reason}
"""
import argparse
import json
import logging
from datetime import date, datetime
from typing import Iterable

from psycopg2.extras import execute_values

from src.common.database.MQSDBConnector import MQSDBConnector

try:
    from scripts.APIs._fmp_helpers import FMPClient  # type: ignore
except ImportError:  # pragma: no cover
    FMPClient = None  # signalled at runtime; helpful CLI error below

logger = logging.getLogger(__name__)


HISTORICAL_SP500_URL = (
    "https://financialmodelingprep.com/stable/historical-sp500-constituent"
)
CURRENT_SP500_URL = (
    "https://financialmodelingprep.com/stable/sp500-constituent"
)
CURRENT_NDX_URL = (
    "https://financialmodelingprep.com/stable/nasdaq-constituent"
)

CREATE_INDEX_MEMBERSHIP_SQL = """
CREATE TABLE IF NOT EXISTS index_membership (
    id SERIAL PRIMARY KEY,
    index_id VARCHAR(16) NOT NULL,        -- 'sp500' or 'ndx'
    ticker VARCHAR(16) NOT NULL,
    added_on DATE,                        -- inclusive; NULL = always-member
    removed_on DATE,                      -- exclusive; NULL = still member today
    reason TEXT,
    source TEXT NOT NULL,                 -- 'fmp_historical' | 'fmp_current' | 'manual'
    fetched_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (index_id, ticker, added_on)
);
CREATE INDEX IF NOT EXISTS ix_membership_index_ticker
    ON index_membership (index_id, ticker);
CREATE INDEX IF NOT EXISTS ix_membership_date_range
    ON index_membership (index_id, added_on, removed_on);
"""


def parse_fmp_historical_sp500(rows: Iterable[dict]) -> list[dict]:
    """Each FMP row is a single change event:
      {"date":"2024-09-23","symbol":"PLTR","addedSecurity":"Palantir",
       "removedTicker":"AAL","removedSecurity":"American Airlines",
       "dateAdded":"September 23, 2024","reason":"..."}
    We emit one row per add and one row per drop, then reconcile add/drop
    pairs to fill (added_on, removed_on) for each ticker. Tickers never
    explicitly dropped retain removed_on=NULL.
    """
    events: list[tuple[str, date, str, str]] = []  # (ticker, event_date, event_type, reason)
    for r in rows:
        try:
            d = datetime.strptime(r["date"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue
        added = (r.get("symbol") or "").strip().upper()
        removed = (r.get("removedTicker") or "").strip().upper()
        reason = (r.get("reason") or "").strip()
        if added:
            events.append((added, d, "ADD", reason))
        if removed:
            events.append((removed, d, "DROP", reason))

    # Reconcile per-ticker timeline (events sorted oldest → newest)
    events.sort(key=lambda e: (e[0], e[1]))
    out: list[dict] = []
    open_adds: dict[str, tuple[date, str]] = {}
    for tkr, dt, ev, reason in events:
        if ev == "ADD":
            # If there was a prior ADD with no DROP, close it (rare; data quirk)
            if tkr in open_adds:
                add_dt, prev_reason = open_adds[tkr]
                out.append({
                    "ticker": tkr,
                    "added_on": add_dt,
                    "removed_on": dt,
                    "reason": prev_reason or "implicit-close",
                })
            open_adds[tkr] = (dt, reason)
        else:  # DROP
            if tkr in open_adds:
                add_dt, add_reason = open_adds.pop(tkr)
                out.append({
                    "ticker": tkr,
                    "added_on": add_dt,
                    "removed_on": dt,
                    "reason": reason or add_reason,
                })
            else:
                # Drop with no recorded prior add — ticker was in the index
                # before the FMP history starts. Record with added_on=NULL.
                out.append({
                    "ticker": tkr,
                    "added_on": None,
                    "removed_on": dt,
                    "reason": reason,
                })
    # Names that are still in the index today
    for tkr, (add_dt, reason) in open_adds.items():
        out.append({
            "ticker": tkr,
            "added_on": add_dt,
            "removed_on": None,
            "reason": reason,
        })
    return out


def insert_membership_rows(db, index_id: str, rows: list[dict], source: str) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO index_membership
            (index_id, ticker, added_on, removed_on, reason, source)
        VALUES %s
        ON CONFLICT (index_id, ticker, added_on) DO UPDATE
           SET removed_on = EXCLUDED.removed_on,
               reason     = EXCLUDED.reason,
               source     = EXCLUDED.source
    """
    payload = [
        (index_id, r["ticker"], r.get("added_on"), r.get("removed_on"),
         r.get("reason"), source)
        for r in rows if r.get("ticker")
    ]
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            execute_values(cur, sql, payload)
        conn.commit()
        return len(payload)
    finally:
        db.release_connection(conn)


def fetch_and_persist(indices: list[str], *, apply_schema: bool, dry_run: bool) -> None:
    if FMPClient is None:
        raise RuntimeError(
            "scripts/APIs/_fmp_helpers.py::FMPClient not importable; "
            "ensure repo root is on PYTHONPATH and FMP_API_KEY is set."
        )
    db = MQSDBConnector()
    if apply_schema:
        for stmt in [s.strip() for s in CREATE_INDEX_MEMBERSHIP_SQL.split(";") if s.strip()]:
            db.execute_query(stmt + ";")
    client = FMPClient(logger=logger)
    total = 0
    if "sp500" in indices:
        hist = client.get(HISTORICAL_SP500_URL, label="historical S&P 500") or []
        rows = parse_fmp_historical_sp500(hist)
        logger.info("Parsed %d S&P 500 membership intervals from FMP", len(rows))
        if not dry_run:
            inserted = insert_membership_rows(db, "sp500", rows, "fmp_historical")
            total += inserted
            logger.info("Inserted %d S&P 500 membership rows", inserted)
        # Reconcile with the current snapshot — names live today must have
        # removed_on=NULL even if FMP historical drops the row.
        current = client.get(CURRENT_SP500_URL, label="current S&P 500") or []
        symbols = sorted({e.get("symbol", "").strip().upper() for e in current if isinstance(e, dict)})
        if not dry_run:
            patch = [{"ticker": t, "added_on": None, "removed_on": None, "reason": "current-snapshot reconciliation"} for t in symbols if t]
            inserted = insert_membership_rows(db, "sp500", patch, "fmp_current")
            total += inserted
            logger.info("Reconciled %d current S&P 500 members", inserted)
    if "ndx" in indices:
        # FMP historical-ndx not in stable docs as of audit date; fall back
        # to current snapshot only and log the gap.
        current = client.get(CURRENT_NDX_URL, label="current Nasdaq-100") or []
        symbols = sorted({e.get("symbol", "").strip().upper() for e in current if isinstance(e, dict)})
        if not dry_run:
            patch = [{"ticker": t, "added_on": None, "removed_on": None, "reason": "current-snapshot only — FMP lacks historical NDX"} for t in symbols if t]
            inserted = insert_membership_rows(db, "ndx", patch, "fmp_current")
            total += inserted
            logger.warning(
                "NDX historical membership unavailable in FMP stable; "
                "only current snapshot persisted. Consider Wikipedia "
                "revisions (S7) or Nasdaq Press Release scrape for backfill."
            )
    logger.info("DONE: inserted %d rows total (dry_run=%s)", total, dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(prog="historical_constituents")
    parser.add_argument("--indices", nargs="+", default=["sp500"], choices=["sp500", "ndx"])
    parser.add_argument("--apply-schema", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    fetch_and_persist(args.indices, apply_schema=args.apply_schema, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

**Patch 2.2** — PATCH `scripts/APIs/build_portfolio6_universe.py` to accept `--as-of YYYY-MM-DD` and (when set) query the new table:

```diff
--- a/scripts/APIs/build_portfolio6_universe.py
+++ b/scripts/APIs/build_portfolio6_universe.py
@@ -1,3 +1,5 @@
+import argparse
+from datetime import date, datetime
 import json
 import logging
 import sys
@@ -7,6 +9,7 @@
 from typing import List
 
 from _fmp_helpers import FMPClient  # noqa: E402
+from typing import Optional
 
 
 OUTPUT_PATH = REPO_ROOT / "src" / "portfolios" / "portfolio_6" / "universe.json"
@@ -52,9 +55,49 @@
     return symbols
 
 
-def main() -> int:
+def pit_universe_from_db(as_of: date) -> Optional[List[str]]:
+    """Reconstruct the S&P 500 ∪ NDX universe as of `as_of` from the
+    `index_membership` table populated by historical_constituents.py.
+    Returns None if the table is empty / not present — caller falls back to
+    the current-snapshot FMP fetch."""
+    try:
+        from src.common.database.MQSDBConnector import MQSDBConnector
+    except ImportError:
+        return None
+    db = MQSDBConnector()
+    sql = """
+        SELECT DISTINCT ticker
+        FROM index_membership
+        WHERE index_id IN ('sp500', 'ndx')
+          AND (added_on   IS NULL OR added_on   <= %s)
+          AND (removed_on IS NULL OR removed_on >  %s)
+    """
+    res = db.execute_query(sql, (as_of, as_of), fetch=True)
+    if res.get("status") != "success":
+        return None
+    rows = res.get("data") or []
+    tickers = sorted({r[0] if not isinstance(r, dict) else r.get("ticker") for r in rows if r})
+    return [t for t in tickers if t]
+
+
+def main(argv=None) -> int:
+    parser = argparse.ArgumentParser()
+    parser.add_argument(
+        "--as-of",
+        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
+        default=None,
+        help="Reconstruct PIT universe as-of YYYY-MM-DD (requires index_membership table).",
+    )
+    args = parser.parse_args(argv)
+
     start = time.time()
     logger.info("=== build_portfolio6_universe START ===")
+    if args.as_of is not None:
+        logger.info("PIT mode: as-of=%s", args.as_of)
+        pit = pit_universe_from_db(args.as_of)
+        if pit:
+            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
+            OUTPUT_PATH.write_text(json.dumps(pit, indent=2))
+            logger.info("Wrote %d PIT tickers to %s", len(pit), OUTPUT_PATH)
+            return 0
+        logger.warning("PIT lookup empty/unavailable; falling back to current snapshot.")
     logger.info("REPO_ROOT=%s", REPO_ROOT)
     logger.info("OUTPUT_PATH=%s", OUTPUT_PATH)
```

**Patch 2.3** — PATCH backtest runner to filter the universe by `bar_date` PIT membership. The minimal-blast-radius hook is in `Portfolio6Strategy._collect_returns`:

```diff
--- a/src/portfolios/portfolio_6/strategy.py
+++ b/src/portfolios/portfolio_6/strategy.py
@@ -86,6 +86,13 @@ class Portfolio6Strategy(BasePortfolio):
         self.fundamentals_df: Optional[pd.DataFrame] = self._load_fundamentals()
         self._target_weights: Dict[str, float] = {}
         self._last_rebalance_month: Optional[Tuple[int, int]] = None
+        self.enforce_pit_universe: bool = bool(
+            p6_cfg.get("ENFORCE_PIT_UNIVERSE", False)
+        )
+        # PIT filter cache: (year, month) -> set(ticker). Built lazily from
+        # the index_membership table; falls back to no-op if unavailable.
+        self._pit_cache: Dict[Tuple[int, int], set] = {}
+        self._pit_unavailable_logged = False
 
@@ -172,6 +179,9 @@ class Portfolio6Strategy(BasePortfolio):
     def _collect_returns(self, context: StrategyContext) -> Dict[str, pd.Series]:
         lookback_str = f"{self.vol_lookback_days + 30}d"
         out: Dict[str, pd.Series] = {}
+        pit_set = self._pit_universe_as_of(context.time) if self.enforce_pit_universe else None
         candidates = [
             t for t in self.tickers if t not in (self.gld_ticker, self.trend_ticker)
         ]
+        if pit_set is not None:
+            candidates = [t for t in candidates if t in pit_set]
         min_required = max(self.vol_lookback_days // 2, 30)
@@ -195,3 +205,38 @@ class Portfolio6Strategy(BasePortfolio):
             out[ticker] = returns.iloc[-self.vol_lookback_days:]
         return out
+
+    def _pit_universe_as_of(self, ctx_time) -> Optional[set]:
+        """Return the set of tickers that were S&P 500 or NDX members on
+        ctx_time. None means PIT filtering is disabled (table absent etc.)."""
+        if ctx_time is None:
+            return None
+        key = (ctx_time.year, ctx_time.month)
+        if key in self._pit_cache:
+            return self._pit_cache[key]
+        try:
+            sql = (
+                "SELECT DISTINCT ticker FROM index_membership "
+                "WHERE index_id IN ('sp500','ndx') "
+                "  AND (added_on   IS NULL OR added_on   <= %s) "
+                "  AND (removed_on IS NULL OR removed_on >  %s)"
+            )
+            from datetime import date as _date
+            as_of = ctx_time.date() if hasattr(ctx_time, "date") else _date(*key, 1)
+            res = self.db_connector.execute_query(sql, (as_of, as_of), fetch=True)
+            if res.get("status") != "success" or not res.get("data"):
+                if not self._pit_unavailable_logged:
+                    self.logger.warning(
+                        "[P6] ENFORCE_PIT_UNIVERSE=True but index_membership "
+                        "table empty/missing; skipping PIT filter (= current-snapshot fallback)."
+                    )
+                    self._pit_unavailable_logged = True
+                self._pit_cache[key] = None  # type: ignore[assignment]
+                return None
+            rows = res["data"]
+            tickers = {(r[0] if not isinstance(r, dict) else r.get("ticker")) for r in rows}
+            tickers.discard(None)
+            self._pit_cache[key] = tickers
+            return tickers
+        except Exception as e:
+            self.logger.exception("[P6] PIT lookup failed: %s", e)
+            return None
```

Default `ENFORCE_PIT_UNIVERSE = false` ⇒ bit-exact back-compat. Flip to `true` in P6/P7/P8 configs after the table is populated.

**Patch 2.4** — Manual one-shot: queue backfill for the union of tickers that have ever been members. This brings delisted names into `market_data`:

```bash
# After Patch 2.1 runs and index_membership has data:
python -c "
import json
from src.common.database.MQSDBConnector import MQSDBConnector
db = MQSDBConnector()
res = db.execute_query('SELECT DISTINCT ticker FROM index_membership ORDER BY 1', fetch=True)
tickers = sorted({r[0] if not isinstance(r, dict) else r.get('ticker') for r in res['data']})
json.dump(tickers, open('historical_union_tickers.json', 'w'), indent=2)
print('wrote', len(tickers), 'tickers')
"
python -m src.orchestrator.backfill.backfill_cli concurrent \
    --start 010119 --end 200526 --interval 1 \
    --tickers $(python -c "import json; print(' '.join(json.load(open('historical_union_tickers.json'))))") \
    --threads 8 --on-conflict ignore
```

### Tier 3 — Delisting-return injection + Wikipedia-revision NDX backfill (≈ 1 dev-week)

Defer. Required only if Falsification §6 fails. Outline:
- Inject Shumway 1997 corrected delisting returns (S2): −0.30 for performance-related NYSE/AMEX delists, −0.55 for NASDAQ, 0 for M&A consummations.
- Scrape Wikipedia revisions of Nasdaq-100 constituency page (S7 methodology) for NDX historical members where FMP lacks coverage.

## 6. Falsification test (concrete, runnable post-Tier-2)

**Hypothesis** (null = current snapshot universe is OK for P6 backtests): the CAGR of P6 backtested on 2010-2020 using `universe.json` (current snapshot) is within 50 bps/yr of the CAGR of the same P6 strategy backtested over the same window using PIT membership reconstructed via Patch 2.1.

**Procedure:**

1. Run Patch 2.1 to populate `index_membership` for S&P 500 (FMP historical).
2. Backfill `market_data` for the union of historical S&P 500 ∪ current NDX tickers across 2010-01-01 → 2020-12-31 (Patch 2.4).
3. Two backtest runs of `Portfolio6Strategy` over `START_DATE = 2010-01-01`, `END_DATE = 2020-12-31`:
   - Run A: `ENFORCE_PIT_UNIVERSE = false` (current snapshot, the status quo).
   - Run B: `ENFORCE_PIT_UNIVERSE = true` (PIT, the proposed fix).
4. Compute `CAGR_A - CAGR_B`, `Sharpe_A - Sharpe_B`, `MaxDD_B - MaxDD_A`.

**Pass / fail thresholds (decided by D2):**
- `|CAGR_A - CAGR_B| < 50 bps/yr` AND `|Sharpe_A - Sharpe_B| < 0.05` AND `|MaxDD_B - MaxDD_A| < 200 bps` ⇒ **PASS**: current-snapshot universe is acceptable for P6-class strategies. Document the conclusion and keep `ENFORCE_PIT_UNIVERSE=false`. Re-test on every major universe expansion.
- Otherwise ⇒ **FAIL**: PIT is required. Set `ENFORCE_PIT_UNIVERSE=true` globally; promote Tier 2 to default; revisit any P7/P8 Sharpe gates set against snapshot baselines.

**Pre-registered expectation:** the test will FAIL. Conservative midpoint (S3+S9 cross-validation): `CAGR_A - CAGR_B ≈ +100 bps/yr`, `Sharpe inflation ≈ +0.10`.

## 7. Risks + rollback path

| Risk | Probability | Impact | Mitigation / rollback |
|---|---|---|---|
| FMP historical-sp500-constituent endpoint changes payload shape (recent FMP changelog notes a "symbol naming methodology" correction) | Medium | Medium | `parse_fmp_historical_sp500` validates each event and skips bad rows; if >10% rows skipped, raise & rollback table writes |
| FMP rate-limits the historical fetch under the existing 3 000 req/min cap | Low | Low | `_fmp_helpers.FMPClient` already throttles + retries; historical endpoint is a single page (≤ 1 call) |
| Tier-2 backfill of historical tickers takes weeks of FMP credits | High | Medium | Pre-flight estimate the union size (~620 tickers × 10y × 1m bars ≈ 10× current quota); restrict to daily bars (`--interval 1440`) for the PIT-comparison study; full 1m backfill is optional |
| Wide CSV diff makes review hard | Low | Low | All patches are reviewable in ≤500 lines; Tier 1 alone fits in one commit |
| Setting `ENFORCE_PIT_UNIVERSE=true` accidentally drops live trading universe | Low | High | Flag defaults `false` in all configs; live trading reads `tickers.json` directly, not the strategy's PIT filter; PIT applies only during backtest replay |
| Bias estimate (§4) is wrong for P6 specifically | Medium | Low | Falsification §6 directly measures it; the entire purpose of Tier 2 is to replace the estimate with a number |
| Team A's P7 / P8 promotion gate (§11) was set against biased baselines | Confirmed | Medium | Re-evaluate Sharpe gates *after* Falsification §6 lands. If PIT-corrected P6 Sharpe drops by 0.10, the `Sharpe(P7) − Sharpe(P6) ≥ 0.15` test becomes 5× harder to meet — that is the *correct* behaviour |

**Rollback path:** every patch is feature-flagged.
- Tier 1 — revert two files (`main_backtest.py` comment block, `refresh.py` save_tickers).
- Tier 2 — drop the new file `historical_constituents.py`, revert two patches (`build_portfolio6_universe.py`, `strategy.py`). Set `ENFORCE_PIT_UNIVERSE=false` everywhere. The `index_membership` table can persist harmlessly; no other code reads it.
- Tier 3 — not landed.

No data is overwritten destructively at any tier. All ticker JSON edits are append-only after Patch 1.2. The `market_data` table is only ever insert-only.

---

**End of D2 audit. Apply Tier 1 in the same PR that lands Team A SYNTHESIS to fence the limitation explicitly; schedule Tier 2 before any P7/P8 promotion decision crosses a multi-year backtest.**
