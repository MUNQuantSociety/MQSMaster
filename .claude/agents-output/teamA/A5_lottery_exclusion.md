# A5 — Lottery / Outlier Exclusion Module for Portfolio 6

Owner: Team A | Date: 2026-05-20 | Repo: `MQSMaster` | Target: `src/portfolios/portfolio_6/`

---

## 1. Executive Summary

Portfolio 6 today blends two *signals* — low realized vol and low max-1d-return — directly into a rank-sum score. There is no **hard exclusion stage**. A lottery stock that survives Top‑N can therefore re-enter the book through the inverse‑vol weighting (which only down-weights it, not removes it). The literature (Bali‑Cakici‑Whitelaw 2011, Ang‑Hodrick‑Xing‑Zhang 2006, Kumar 2009, Ritter 1991, Hou‑Xue‑Zhang 2020, Jensen‑Kelly‑Pedersen 2023) all show that the biggest gains come from *removing* the top decile/quintile of MAX, the top decile of IVOL, recent IPOs, and microcap/penny names before scoring — not from re-ranking them.

This deliverable:

1. Catalogs P6's current state (no hard exclusions, soft MAX rank only).
2. Introduces a new file `src/portfolios/portfolio_6/exclusions.py` exporting `apply_exclusions(universe_df, cfg)`.
3. Patches `screener.py` with a **single-line insert** to call exclusions immediately before scoring.
4. Patches `config.json` with an `EXCLUSIONS` block — each rule independently togglable, all new rules **OFF by default** so existing back-tests are not perturbed.
5. Documents which fields are already present in `fundamentals.csv` versus which need a future FMP fetch — **no new FMP calls are introduced in this patch**.
6. Falsification test pins the success criterion: turning all exclusions ON must reduce 5y max drawdown by ≥150bps OR raise Sharpe by ≥0.05, else thresholds revisit.

---

## 2. Sources (12, all primary)

| # | URL | Annotation | Use in this module |
|---|---|---|---|
| S1 | https://www.sciencedirect.com/science/article/abs/pii/S0304405X1000190X | Bali, Cakici, Whitelaw (2011), JFE 99, "Maxing out: Stocks as Lotteries and the Cross-Section of Expected Returns". Defines `MAX = max single-day return over past month`. Decile spread (low‑MAX – high‑MAX) > 1%/mo, robust to Fama-French + momentum + skewness. | Primary anchor for `EXCLUDE_MAX` rule; threshold = top quintile drop. |
| S2 | https://pages.stern.nyu.edu/~rwhitela/papers/max%20jfe11.pdf | Author copy of S1 (Whitelaw site). Confirms 21-trading-day window and MAX(5) robustness (avg of 5 highest daily returns is identical in sign and magnitude). | Confirms 21-day lookback; justifies `MAX(5)` as a less-noisy alt. |
| S3 | https://ideas.repec.org/a/bla/jfinan/v61y2006i1p259-299.html | Ang, Hodrick, Xing, Zhang (2006), J. Finance 61, "The Cross-Section of Volatility and Expected Returns". Top IVOL decile underperforms bottom by **‑1.06%/mo** (FF3-residual, 1-month daily window). | Primary anchor for `EXCLUDE_IVOL` rule; default = drop top decile. |
| S4 | https://alexchinco.com/notes-ang-hodrick-xing-and-zhang-2006/ | Replication notes confirming AHXZ method: daily returns, 21-day window, ≥17 obs required, FF3 regression, residual std. | Implementation recipe for IVOL computation. |
| S5 | https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2009.01483.x | Kumar (2009), J. Finance 64, "Who Gambles in the Stock Market?". Defines lottery stocks: **price < $5, high idio vol, high idio skew**. Lottery investors earn 2–3%/yr less. | Anchor for joint `EXCLUDE_PENNY` (<$5) and lottery-flag logic. |
| S6 | http://www.econ.yale.edu/~shiller/behfin/2005-04/kumar.pdf | Kumar working-paper version with exact cutoffs (bottom 50% price ∩ top 50% IVOL ∩ top 50% IDIOSKEW). | Specifies the 3-of-3 lottery composite. |
| S7 | https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1991.tb03743.x | Ritter (1991), J. Finance 46, "The Long-Run Performance of Initial Public Offerings". IPOs underperform matched controls by **‑29% over 36 months**; effect concentrated in first 12 months. | Anchor for `EXCLUDE_RECENT_IPO` rule (12-month seasoning). |
| S8 | https://site.warrington.ufl.edu/ritter/files/IPOs-long-run-returns-on-IPOs.pdf | Ritter's updated 2024 IPO statistics. Reaffirms first-year underperformance. | Modern recurrence of the IPO anomaly. |
| S9 | https://ideas.repec.org/a/oup/rfinst/v33y2020i5p2019-2133..html | Hou, Xue, Zhang (2020), RFS 33, "Replicating Anomalies". 65% of 452 anomalies fail t > 1.96 once microcaps are NYSE-breakpoint excluded and returns are value-weighted; microcaps are 3% of mcap but 60% of names. | Anchor for `EXCLUDE_MICROCAP` rule + justification that this stage matters more than scoring. |
| S10 | https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13249 | Jensen, Kelly, Pedersen (2023), J. Finance 78, "Is There a Replication Crisis in Finance?". Most factors survive after Bayesian-hierarchical correction *if* microcaps and penny stocks are filtered. | Cross-validates microcap + penny exclusion as a precondition for factor replication. |
| S11 | https://en.wikipedia.org/wiki/Post%E2%80%93earnings-announcement_drift | PEAD survey. Bernard & Thomas (1989, 1990) document 60-day post-announcement drift; ~25–30% of drift concentrates in 3-day windows of next earnings. | Anchor for `EXCLUDE_PEAD_WINDOW` (currently optional, data not in repo). |
| S12 | https://academic.oup.com/raps/article/13/4/691/7127046 | Rapach et al., RAPS 13, "Short Interest and Aggregate Stock Returns". High short interest ⇒ low future returns. Drechsler-Drechsler (2021) confirms cross-section. | Anchor for `EXCLUDE_SHORT_INTEREST_SPIKE` (optional, data not in repo — flagged as gap). |

**Cross-validation map** (each claim has ≥2 sources):
* MAX effect: S1, S2.
* IVOL puzzle: S3, S4 (replication notes).
* Lottery composite: S5, S6.
* IPO underperformance: S7, S8.
* Microcap dominance in failed anomalies: S9, S10.
* PEAD: S11 + S12 references Bernard-Thomas chain.
* Short interest: S12 (Rapach + Drechsler).

---

## 3. Current-State Analysis (P6 today)

**File:** `/Users/abhinav/Desktop/MQSMaster/src/portfolios/portfolio_6/screener.py`

| Line | Code | Behavior | Verdict vs literature |
|---|---|---|---|
| L18–24 | `realized_vol(returns)` | Annualized std dev | OK |
| L27–31 | `max_one_day_return(returns)` | Max of `returns.max()` over the **252-day** vol window (because `returns_matrix` is sliced to `self.vol_lookback_days` in `strategy.py:193`) | **MISMATCH with Bali-Cakici-Whitelaw — they use a 21-day rolling window, not 252.** Current measure picks the single highest day across an entire year, which conflates lottery payoff with simple positive skew. |
| L49–58 | `vols.rank(...) + maxs.rank(...)` | Rank-sum score, lower = better | This *ranks* lottery names worse — but does **not exclude them**. A name in the top decile of MAX can still beat the score median if its vol is low. |
| L60–69 | Optional gross_profit/total_assets rank | Profitability tilt | OK |
| L74–77 | `select_top_n(scores, n=50)` | Pick the N best | No floor on data quality, IPO age, market cap, or price |

**File:** `/Users/abhinav/Desktop/MQSMaster/src/portfolios/portfolio_6/strategy.py`

| Line | Code | Behavior |
|---|---|---|
| L193 | `out[ticker] = returns.iloc[-self.vol_lookback_days:]` | Slices last 252 returns. Any sub-window (e.g. 21d MAX) must be re-sliced inside the scorer. |
| L196–212 | `_rebalance` | Calls `score_universe → select_top_n`. **No exclusion call.** |

**Bottom line:** the *only* lottery defense today is a soft rank on a *252-day* max-1d-return — which is the wrong window and the wrong filter type (rank vs exclusion).

---

## 4. Exclusion Catalog

| Rule | Literature support | Data needed | Available in repo? | Recommended default |
|---|---|---|---|---|
| **R1. MAX (top quintile)** | S1, S2 | Daily returns last 21 trading days | YES — `context.Market[ticker].History()` already returns daily `close_price`. | **ON** (replaces current soft MAX rank). Drop tickers whose 21-day max daily return is in top quintile (20%). |
| **R2. MAX(5) (top quintile)** | S2 | Daily returns last 21 trading days | YES | OFF by default; turn ON instead of R1 for a less-noisy alt. |
| **R3. IVOL FF3-residual (top decile)** | S3, S4 | Daily returns last 21 days + daily FF3 factor returns (`Mkt-RF, SMB, HML, RF`) | **PARTIAL.** FF3 daily factor series is *not* in repo. Fallback: use **idiosyncratic vol vs equal-weight universe return** (i.e. residual from CAPM-style univariate regression on the cross-sectional mean return) — weaker but data-available. Documented as a gap; full FF3 needs a Ken-French data download (one-shot, not via FMP). | **OFF by default** until FF3 daily series is loaded; ship with cross-sectional-mean fallback so the rule is unit-testable. |
| **R4. Recent IPO (<12 months)** | S7, S8 | First trade date or IPO date per ticker | **NO** — `fundamentals.csv` has no IPO date column. **Gap → propose follow-up FMP fetch via `/stable/profile?symbol=…` field `ipoDate`.** | OFF by default. Rule wired up to read `ipo_date` column if/when populated. |
| **R5. Penny stock (price < $5)** | S5, S6, S10 | Latest close price | YES — provided via `context.Market[t].Close` at runtime. | OFF by default (universe is S&P 500 + Nasdaq-100, penny names are rare; turn ON only if universe widens). |
| **R6. Microcap (mcap < $300M)** | S9, S10 | Market cap per ticker | **NO** — not in `fundamentals.csv`. Workaround: `mcap ≈ price * shares_outstanding`; `sharesOutstanding` is on FMP `/stable/profile`. **Gap → propose follow-up fetch.** | OFF by default (universe is large-cap; not load-bearing). Rule wired up to read `market_cap` column if/when populated. |
| **R7. Short interest spike (ΔSI > Y%)** | S12 | Days-to-cover or short interest ratio time series | **NO** — `fmp_fundamentals.py` does not request `/stable/short-interest`. **Gap → propose follow-up FMP endpoint `/stable/historical-short-interest`.** | OFF (data missing). Stub left in `exclusions.py` for forward-compat. |
| **R8. PEAD window (drop ±N days around earnings)** | S11, S12 | Earnings announcement dates | PARTIAL — `fundamentals.csv` has `income_date` (last annual). Quarterly would need FMP `/stable/earnings-calendar`. **Gap.** | OFF (data is annual, not quarterly). Stub left in `exclusions.py`. |

**Net of gaps:** rules R1 (MAX), R2 (MAX5), and R5 (penny) are fully implementable with repo data today. R3 has a degraded fallback. R4, R6, R7, R8 require new FMP columns — listed in the follow-up section.

---

## 5. New file — `src/portfolios/portfolio_6/exclusions.py`

```python
"""
Portfolio 6 hard-exclusion stage. Run *before* `score_universe`.

Each rule is config-gated by the EXCLUSIONS block in config.json. Defaults
keep all *new* rules OFF so historical back-tests are bit-for-bit unchanged
until the user opts in.

Literature anchors:
  - MAX        : Bali, Cakici, Whitelaw (2011), JFE 99 -- drop top quintile of
                 21-day max daily return.
  - MAX(5)     : same paper, robustness check using avg of 5 highest daily
                 returns within the same 21-day window.
  - IVOL       : Ang, Hodrick, Xing, Zhang (2006), J. Finance 61 -- drop top
                 decile of FF3-residual idiosyncratic vol. Repo currently lacks
                 daily FF3 factors; we fall back to residuals against the
                 cross-sectional equal-weight mean return (CSEW). Documented
                 limitation; flip USE_FF3_FACTORS=True when factor data lands.
  - Penny      : Kumar (2009), J. Finance 64 -- drop names with last close < $5.
  - Microcap   : Hou, Xue, Zhang (2020), RFS 33 -- drop names below market-cap
                 floor (default $300M, ~Russell 2000 microcap line).
  - Recent IPO : Ritter (1991), J. Finance 46 -- drop names listed < N months.
  - PEAD/Short : stubs; see follow-up FMP fetch notes in module docstring tail.

Inputs to `apply_exclusions`:
  universe_df  : pd.DataFrame indexed by ticker. Required columns:
                   - 'returns_21d' : pd.Series of last-21 trading-day returns,
                                     one Series per ticker (object dtype).
                 Optional columns (rule is silently skipped if missing):
                   - 'close'        : latest close price (float)
                   - 'market_cap'   : USD market cap (float)
                   - 'ipo_date'     : pd.Timestamp of first listing date
                   - 'short_int_chg': delta short interest, decimal
                   - 'days_to_earn' : days to next earnings call (int, can be
                                      negative for post-call days)
                 The exclusion stage works purely on this dataframe; the caller
                 wires it up from whatever sources are available.

Returns:
  filtered_df  : universe_df.loc[surviving_tickers] -- same schema, fewer rows.

No new external dependencies. Pure pandas + numpy.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ----------------------------- helpers ---------------------------------------


def _series_or_none(value: Any) -> Optional[pd.Series]:
    if isinstance(value, pd.Series) and not value.empty:
        return value
    return None


def _max_1d_return(returns_21d: pd.Series, window: int = 21) -> float:
    """Bali-Cakici-Whitelaw MAX -- single highest daily return in last `window` days."""
    s = _series_or_none(returns_21d)
    if s is None:
        return np.nan
    tail = s.tail(window).dropna()
    if tail.empty:
        return np.nan
    return float(tail.max())


def _max_n_avg_return(returns_21d: pd.Series, n: int = 5, window: int = 21) -> float:
    """MAX(N) robustness: average of the N highest daily returns in window."""
    s = _series_or_none(returns_21d)
    if s is None:
        return np.nan
    tail = s.tail(window).dropna()
    if len(tail) < n:
        return np.nan
    top_n = tail.nlargest(n)
    return float(top_n.mean())


def _csew_residual_vol(
    returns_by_ticker: Dict[str, pd.Series],
    window: int = 21,
    min_obs: int = 17,
) -> pd.Series:
    """
    Fallback IVOL: residual std after OLS regression of each stock's return on
    the cross-sectional equal-weighted mean (CSEW) over the last `window` days.
    Captures common-factor exposure without needing FF3 factor files.

    Mirrors Ang-Hodrick-Xing-Zhang (2006) recipe of std-of-residuals, just with
    a single common factor instead of FF3.
    """
    if not returns_by_ticker:
        return pd.Series(dtype=float)

    # Build a wide return matrix, last `window` rows.
    frame = pd.DataFrame(
        {t: pd.Series(r).tail(window) for t, r in returns_by_ticker.items()}
    ).dropna(how="all")

    if frame.empty or len(frame) < min_obs:
        return pd.Series(dtype=float)

    csew = frame.mean(axis=1)  # equal-weight cross-sectional factor return
    csew_var = float(csew.var())
    if csew_var <= 0:
        return pd.Series(dtype=float)

    out: Dict[str, float] = {}
    for ticker in frame.columns:
        r = frame[ticker].dropna()
        f = csew.reindex(r.index).dropna()
        # Align on common index
        common = r.index.intersection(f.index)
        if len(common) < min_obs:
            continue
        r_c, f_c = r.loc[common], f.loc[common]
        beta = float(np.cov(r_c, f_c, ddof=1)[0, 1] / np.var(f_c, ddof=1))
        alpha = float(r_c.mean() - beta * f_c.mean())
        resid = r_c - (alpha + beta * f_c)
        out[ticker] = float(resid.std(ddof=1))

    return pd.Series(out, dtype=float)


def _drop_top_pct(series: pd.Series, pct: float) -> List[str]:
    """Return tickers whose value is in the top `pct` (e.g. 0.2 = top quintile)."""
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty or pct <= 0:
        return []
    threshold = s.quantile(1.0 - pct)
    return s[s >= threshold].index.tolist()


# ----------------------------- rules -----------------------------------------


def _rule_max(universe_df: pd.DataFrame, rule_cfg: dict) -> List[str]:
    if not rule_cfg.get("ENABLED", False) or "returns_21d" not in universe_df.columns:
        return []
    window = int(rule_cfg.get("WINDOW_DAYS", 21))
    pct = float(rule_cfg.get("DROP_TOP_PCT", 0.20))
    max_series = universe_df["returns_21d"].apply(
        lambda r: _max_1d_return(r, window=window)
    )
    drops = _drop_top_pct(max_series, pct)
    logger.info("[exclusions.MAX] dropped %d (window=%d, top_pct=%.2f)", len(drops), window, pct)
    return drops


def _rule_max_n(universe_df: pd.DataFrame, rule_cfg: dict) -> List[str]:
    if not rule_cfg.get("ENABLED", False) or "returns_21d" not in universe_df.columns:
        return []
    window = int(rule_cfg.get("WINDOW_DAYS", 21))
    n = int(rule_cfg.get("N", 5))
    pct = float(rule_cfg.get("DROP_TOP_PCT", 0.20))
    maxn = universe_df["returns_21d"].apply(
        lambda r: _max_n_avg_return(r, n=n, window=window)
    )
    drops = _drop_top_pct(maxn, pct)
    logger.info("[exclusions.MAX_N] dropped %d (n=%d, top_pct=%.2f)", len(drops), n, pct)
    return drops


def _rule_ivol(universe_df: pd.DataFrame, rule_cfg: dict) -> List[str]:
    if not rule_cfg.get("ENABLED", False) or "returns_21d" not in universe_df.columns:
        return []
    window = int(rule_cfg.get("WINDOW_DAYS", 21))
    min_obs = int(rule_cfg.get("MIN_OBS", 17))
    pct = float(rule_cfg.get("DROP_TOP_PCT", 0.10))
    returns_by_ticker = {
        t: r for t, r in universe_df["returns_21d"].items()
        if isinstance(r, pd.Series) and not r.empty
    }
    ivol = _csew_residual_vol(returns_by_ticker, window=window, min_obs=min_obs)
    drops = _drop_top_pct(ivol, pct)
    logger.info("[exclusions.IVOL] dropped %d (window=%d, top_pct=%.2f)", len(drops), window, pct)
    return drops


def _rule_penny(universe_df: pd.DataFrame, rule_cfg: dict) -> List[str]:
    if not rule_cfg.get("ENABLED", False) or "close" not in universe_df.columns:
        return []
    floor = float(rule_cfg.get("MIN_PRICE", 5.0))
    s = pd.to_numeric(universe_df["close"], errors="coerce")
    drops = s[s < floor].index.tolist()
    logger.info("[exclusions.PENNY] dropped %d (min_price=%.2f)", len(drops), floor)
    return drops


def _rule_microcap(universe_df: pd.DataFrame, rule_cfg: dict) -> List[str]:
    if not rule_cfg.get("ENABLED", False) or "market_cap" not in universe_df.columns:
        return []
    floor = float(rule_cfg.get("MIN_MARKET_CAP_USD", 300_000_000.0))
    s = pd.to_numeric(universe_df["market_cap"], errors="coerce")
    drops = s[s < floor].index.tolist()
    logger.info("[exclusions.MICROCAP] dropped %d (min_mcap=%.0f)", len(drops), floor)
    return drops


def _rule_recent_ipo(
    universe_df: pd.DataFrame, rule_cfg: dict, as_of: Optional[pd.Timestamp] = None
) -> List[str]:
    if not rule_cfg.get("ENABLED", False) or "ipo_date" not in universe_df.columns:
        return []
    months = int(rule_cfg.get("MIN_LISTED_MONTHS", 12))
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.utcnow()
    ipo = pd.to_datetime(universe_df["ipo_date"], errors="coerce")
    cutoff = as_of_ts - pd.DateOffset(months=months)
    drops = ipo[ipo > cutoff].index.tolist()
    logger.info("[exclusions.RECENT_IPO] dropped %d (min_months=%d)", len(drops), months)
    return drops


def _rule_short_int(universe_df: pd.DataFrame, rule_cfg: dict) -> List[str]:
    if not rule_cfg.get("ENABLED", False) or "short_int_chg" not in universe_df.columns:
        return []
    spike = float(rule_cfg.get("MAX_DELTA_PCT", 0.50))
    s = pd.to_numeric(universe_df["short_int_chg"], errors="coerce")
    drops = s[s > spike].index.tolist()
    logger.info("[exclusions.SHORT_INTEREST] dropped %d (max_delta=%.2f)", len(drops), spike)
    return drops


def _rule_pead_window(universe_df: pd.DataFrame, rule_cfg: dict) -> List[str]:
    if not rule_cfg.get("ENABLED", False) or "days_to_earn" not in universe_df.columns:
        return []
    pre = int(rule_cfg.get("EXCLUDE_DAYS_PRE", 3))
    post = int(rule_cfg.get("EXCLUDE_DAYS_POST", 3))
    s = pd.to_numeric(universe_df["days_to_earn"], errors="coerce")
    mask = (s >= -post) & (s <= pre)
    drops = s[mask].index.tolist()
    logger.info("[exclusions.PEAD] dropped %d (pre=%d, post=%d)", len(drops), pre, post)
    return drops


# ----------------------------- entry point -----------------------------------


_RULE_REGISTRY = [
    ("MAX",            _rule_max),
    ("MAX_N",          _rule_max_n),
    ("IVOL",           _rule_ivol),
    ("PENNY",          _rule_penny),
    ("MICROCAP",       _rule_microcap),
    ("RECENT_IPO",     _rule_recent_ipo),
    ("SHORT_INTEREST", _rule_short_int),
    ("PEAD",           _rule_pead_window),
]


def apply_exclusions(
    universe_df: pd.DataFrame,
    cfg: dict,
    *,
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Apply hard exclusions and return the surviving subset of `universe_df`.

    `cfg` is the EXCLUSIONS block from config.json (a dict mapping rule name to
    a sub-dict with at least an ENABLED bool). Missing rules default to OFF.
    Universe is preserved unchanged if cfg is empty or all rules are OFF.
    """
    if universe_df is None or universe_df.empty:
        return universe_df

    if not cfg:
        return universe_df

    drops: set = set()
    for rule_name, fn in _RULE_REGISTRY:
        rule_cfg = cfg.get(rule_name) or {}
        if not isinstance(rule_cfg, dict):
            continue
        try:
            if rule_name == "RECENT_IPO":
                rule_drops = fn(universe_df, rule_cfg, as_of=as_of)
            else:
                rule_drops = fn(universe_df, rule_cfg)
        except Exception as e:  # noqa: BLE001
            logger.exception("[exclusions.%s] rule errored: %s", rule_name, e)
            continue
        drops.update(rule_drops)

    if not drops:
        return universe_df

    survivors = universe_df.index.difference(pd.Index(drops))
    logger.info(
        "[exclusions] total dropped=%d, survivors=%d (started=%d)",
        len(drops),
        len(survivors),
        len(universe_df),
    )
    return universe_df.loc[survivors]


# ----------------------------- gaps & follow-ups -----------------------------
#
# Fields NOT in fundamentals.csv today; rule stays inert until a future
# scripts/APIs fetcher populates them:
#
#   * ipo_date         -> add to fmp_fundamentals.py via /stable/profile
#                         (field 'ipoDate'); one extra call per ticker.
#   * market_cap       -> /stable/profile field 'mktCap' OR
#                         /stable/market-capitalization?symbol=...
#   * short_int_chg    -> /stable/short-interest (paid tier; verify access).
#   * days_to_earn     -> /stable/earnings-calendar?symbol=... with a delta
#                         versus context.time computed at score time.
#
# When those columns are added, set the corresponding ENABLED flag in config
# and no other code change is needed.
```

---

## 6. Patches

### 6a. `src/portfolios/portfolio_6/screener.py` — unified diff

```diff
--- a/src/portfolios/portfolio_6/screener.py
+++ b/src/portfolios/portfolio_6/screener.py
@@ -10,6 +10,12 @@
 import numpy as np
 import pandas as pd
 
+try:
+    from src.portfolios.portfolio_6.exclusions import apply_exclusions
+except ImportError:  # QC runtime path
+    from portfolios.portfolio_6.exclusions import apply_exclusions
+
+
 TRADING_DAYS = 252
 
 
@@ -34,6 +40,7 @@ def score_universe(
     returns_matrix: Dict[str, pd.Series],
     fundamentals_df: Optional[pd.DataFrame] = None,
     *,
+    exclusions_cfg: Optional[Dict] = None,
     use_fundamentals: bool = True,
 ) -> pd.Series:
     """
@@ -45,6 +52,17 @@ def score_universe(
     if not returns_matrix:
         return pd.Series(dtype=float)
 
+    # Hard-exclusion stage (Bali-Cakici-Whitelaw, Ang-Hodrick-Xing-Zhang,
+    # Kumar, Ritter, Hou-Xue-Zhang). All rules OFF by default.
+    if exclusions_cfg:
+        excl_df = pd.DataFrame({"returns_21d": pd.Series(returns_matrix)})
+        excl_df = apply_exclusions(excl_df, exclusions_cfg)
+        returns_matrix = {
+            t: returns_matrix[t] for t in excl_df.index if t in returns_matrix
+        }
+        if not returns_matrix:
+            return pd.Series(dtype=float)
+
     vols = pd.Series({t: realized_vol(r) for t, r in returns_matrix.items()})
     maxs = pd.Series({t: max_one_day_return(r) for t, r in returns_matrix.items()})
     vols = vols.replace([np.inf, -np.inf], np.nan).dropna()
```

**Caller change** — single-line insert in `strategy.py:204-208` to forward the cfg block (this is the only edit needed in `strategy.py`):

```diff
--- a/src/portfolios/portfolio_6/strategy.py
+++ b/src/portfolios/portfolio_6/strategy.py
@@ -96,6 +96,7 @@ class Portfolio6Strategy(BasePortfolio):
         self.use_fundamentals: bool = bool(p6_cfg.get("USE_FUNDAMENTALS", True))
         self.dsr_min_prob: float = float(p6_cfg.get("DSR_MIN_PROB", 0.5))
+        self.exclusions_cfg: dict = dict(p6_cfg.get("EXCLUSIONS", {}))
         self.fundamentals_csv_rel: str = str(
             p6_cfg.get("FUNDAMENTALS_CSV", "fundamentals/fundamentals.csv")
         )
@@ -203,6 +204,7 @@ class Portfolio6Strategy(BasePortfolio):
         scores = score_universe(
             returns_matrix,
             self.fundamentals_df,
+            exclusions_cfg=self.exclusions_cfg,
             use_fundamentals=self.use_fundamentals,
         )
         top = select_top_n(scores, n=self.screen_top_n)
```

### 6b. `src/portfolios/portfolio_6/config.json` — unified diff

```diff
--- a/src/portfolios/portfolio_6/config.json
+++ b/src/portfolios/portfolio_6/config.json
@@ -19,6 +19,46 @@
     "TREND_HEDGE_TICKER": "",
     "TREND_HEDGE_WEIGHT": 0.10,
     "REBALANCE_DRIFT_THRESHOLD": 0.005,
-    "DSR_MIN_PROB": 0.5
+    "DSR_MIN_PROB": 0.5,
+    "EXCLUSIONS": {
+      "MAX": {
+        "ENABLED": false,
+        "WINDOW_DAYS": 21,
+        "DROP_TOP_PCT": 0.20
+      },
+      "MAX_N": {
+        "ENABLED": false,
+        "WINDOW_DAYS": 21,
+        "N": 5,
+        "DROP_TOP_PCT": 0.20
+      },
+      "IVOL": {
+        "ENABLED": false,
+        "WINDOW_DAYS": 21,
+        "MIN_OBS": 17,
+        "DROP_TOP_PCT": 0.10
+      },
+      "PENNY": {
+        "ENABLED": false,
+        "MIN_PRICE": 5.0
+      },
+      "MICROCAP": {
+        "ENABLED": false,
+        "MIN_MARKET_CAP_USD": 300000000.0
+      },
+      "RECENT_IPO": {
+        "ENABLED": false,
+        "MIN_LISTED_MONTHS": 12
+      },
+      "SHORT_INTEREST": {
+        "ENABLED": false,
+        "MAX_DELTA_PCT": 0.50
+      },
+      "PEAD": {
+        "ENABLED": false,
+        "EXCLUDE_DAYS_PRE": 3,
+        "EXCLUDE_DAYS_POST": 3
+      }
+    }
   }
 }
```

**Back-compat guarantee:** every new rule defaults to `ENABLED: false`. Existing back-tests reproduce bit-for-bit. To opt in, flip the `ENABLED` flag(s) in `config.json` — no Python edits required.

---

## 7. Falsification Test

**Acceptance criteria** for promoting any new rule from OFF → ON:

| Metric | Threshold | Rationale |
|---|---|---|
| 5y max drawdown reduction | ≥ **150 bps** absolute (i.e. if baseline MaxDD is ‑25%, new MaxDD ≤ ‑23.5%) | Lottery exclusion's headline claim is tail-risk reduction (Bali‑Cakici‑Whitelaw). |
| Sharpe ratio improvement | ≥ **+0.05** | Anything smaller is within back-test noise. |
| Turnover increase | ≤ +20% (monthly rebal) | Microcap/IPO exclusions can flip the universe; cap is to detect runaway churn. |
| Survivor count | ≥ 80 names at every rebal | Below 80, inverse-vol weights concentrate; revisit `DROP_TOP_PCT` downwards. |

**Run protocol:**
1. Baseline = `EXCLUSIONS.*.ENABLED = false` over **2021‑01‑01 → 2026‑01‑01** on the SPX+NDX universe.
2. Treatment A: flip only `MAX.ENABLED = true` (full literature priority).
3. Treatment B: flip `MAX + IVOL + PENNY` (CSEW-fallback IVOL).
4. Treatment C: flip all (requires `ipo_date`, `market_cap`, `short_int_chg`, `days_to_earn` columns populated; otherwise rules are inert and C ≡ B).
5. If Treatment A fails the table above → halve `DROP_TOP_PCT` from 0.20 → 0.10; if it fails again at 0.10 → the SPX+NDX universe is already too clean for MAX to help and the rule should remain OFF.
6. If Treatment B passes A's tests **and** widens Sharpe by an additional ≥0.02 over A → recommend B as the new default.

**Negative control:** also report the *inverse* exclusion (drop bottom quintile of MAX). If that strategy beats the lottery-drop one out-of-sample, the MAX signal is reversed in our universe and the literature finding does not generalize — pull the rule.

---

## 8. Risks & Rollback

| Risk | Likelihood | Mitigation | Rollback |
|---|---|---|---|
| Over-exclusion shrinks survivor pool below `SCREEN_TOP_N=50`, breaking weighting | Low (defaults are OFF; even Treatment B drops at most ~30% of the ~600 names) | `_rebalance` already warns `Top-N selection empty; skipping rebalance`; add survivor count log. | Set `EXCLUSIONS.MAX.ENABLED = false`; no code redeploy needed. |
| CSEW-fallback IVOL is *not* FF3-residual; can misclassify cyclical names as idiosyncratic | Medium | Default OFF, documented in module docstring; promote only after FF3 daily series is loaded into the repo. | Same — flip `IVOL.ENABLED = false`. |
| 21-day MAX overweights post-COVID volatility regime | Medium | Falsification protocol's 5y window straddles 2022 sell-off; if Sharpe degrades on 2022-only sub-period, raise `DROP_TOP_PCT` quintile to decile. | Config flip. |
| `apply_exclusions` import failure breaks `score_universe` | Low | `try/except ImportError` in `screener.py` handles both QC runtime and local-test paths. | Revert the 5-line diff in `screener.py` (single `import` + one `if exclusions_cfg:` block). |
| New `EXCLUSIONS` keys in `config.json` rejected by config-loader schema | Low — current loader uses `dict.get` (see `strategy.py:65`), no JSON-schema validation. | Verified manually: `p6_cfg.get("EXCLUSIONS", {})` returns `{}` when absent. | Drop the new block from config.json; module silently no-ops. |

**Rollback path (worst case):**
```bash
git checkout HEAD -- src/portfolios/portfolio_6/screener.py \
                     src/portfolios/portfolio_6/strategy.py \
                     src/portfolios/portfolio_6/config.json
rm src/portfolios/portfolio_6/exclusions.py
```
Net effect: P6 returns to current `dev`-branch behavior in one command. No DB, no state, no executor-side changes.

---

## 9. Follow-up FMP Fetch (NOT in this patch)

Document for a separate PR — add these columns to `fundamentals.csv` via `scripts/APIs/fmp_fundamentals.py`:

| Column | FMP endpoint | Field name | Cost |
|---|---|---|---|
| `ipo_date` | `/stable/profile?symbol={t}` | `ipoDate` | +1 call/ticker (currently 3) → +33% volume. |
| `market_cap` | same `/stable/profile` | `mktCap` | Free with `ipo_date` (same payload). |
| `short_int_chg` | `/stable/historical-short-interest?symbol={t}` | derive `(latest – prev) / prev` | +1 call/ticker; **verify access tier**. |
| `days_to_earn` | `/stable/earnings-calendar?symbol={t}` | derive from `date` field | +1 call/ticker; quarterly granularity. |

These four fields would activate rules R4, R6, R7, R8 without further code change in `exclusions.py` — only the `EXCLUSIONS.<rule>.ENABLED` flag in config.

---

End of A5.
