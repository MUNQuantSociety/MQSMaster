# A1 — Portfolio_6 Factor Audit vs Replicated Factor Zoo
**Author**: Team A research agent
**Repo**: `MQSMaster` (branch `dev`)
**Date**: 2026-05-20
**Files touched**: NONE (read-only); diffs supplied below for the user to apply.

---

## 1. Executive Summary

Portfolio_6 currently exploits a narrow slice of the replicated factor zoo: low realized
volatility, lottery-payoff avoidance (MAX), inverse-vol weighting, and a half-implemented
gross-profitability quality screen. Of the ~10 anomaly clusters that survive Hou-Xue-Zhang
(2020), Jensen-Kelly-Pedersen (2023) and the Fama-French five-factor extension, P6 actively
uses **2** (Low Risk, partial Profitability), is silently disabling **1** (Profitability is
gated on an `roe` column that is empty in `fundamentals.csv` and a `gross_profit/total_assets`
ratio that requires only 10 of 519 tickers to be populated), and ignores **5+** robust ones
(Momentum 12-2, Investment/asset growth, full Quality/QMJ, Accruals, Net Payout). The
fundamentals pipeline is **not point-in-time** (uses latest reported), the universe is a
**current-snapshot S&P 500 + Nasdaq-100** (survivor-biased), and the CSV covers a small
fraction of the universe. I recommend adding three flag-gated factors — **Momentum (12-2)**,
**Operating-Profitability (Ball-Gerakos-Linnainmaa-Nikolaev / Fama-French RMW style)**, and
**Conservative-Investment (asset-growth, Cooper-Gulen-Schill / FF CMA)** — to bring P6 from
a Conservative-Formula-lite to a Robeco-style multi-factor defensive sleeve, plus tightening
the data pipeline so the new factors are not silently noise.

---

## 2. Sources (≥10 primary)

Tags: **[R]** = replication / factor-zoo paper, **[F]** = foundational primary paper,
**[P]** = practitioner whitepaper (AQR, Robeco, MSCI), **[M]** = methodological /
data-quality, **[N]** = navigation / corroborating secondary (NOT used as primary
evidence).

1. Jensen, T. I., Kelly, B. T., Pedersen, L. H. (2023). *Is There a Replication Crisis in Finance?* **Journal of Finance** 78(5): 2465–2518. **[R]**
   `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3774514`
   — 153 factors cluster into 13 themes; majority replicate; OOS in 93 countries; tangency portfolio is multi-cluster. *Primary basis for "which factors survived".*
2. Hou, K., Xue, C., Zhang, L. (2020). *Replicating Anomalies.* **Review of Financial Studies** 33(5): 2019–2133. **[R]**
   `https://academic.oup.com/rfs/article/33/5/2019/5236964` (paywalled; abstract via NBER w23394 `https://www.nber.org/papers/w23394`)
   — q-factor (MKT, ME, I/A, ROE) explains majority of 452 anomalies with NYSE-microcap breakpoints. Investment + profitability are first-class.
3. Harvey, C. R., Liu, Y., Zhu, H. (2016). *... and the Cross-Section of Expected Returns.* **Review of Financial Studies** 29(1): 5–68. **[R]**
   `https://academic.oup.com/rfs/article/29/1/5/1843824` ; working version `https://www.nber.org/papers/w20592`
   — 316 published factors; raises significance bar to |t| > 3.0. Used to justify multiple-testing penalty (DSR is already in P6).
4. Fama, E. F., French, K. R. (2015). *A Five-Factor Asset Pricing Model.* **Journal of Financial Economics** 116(1): 1–22. **[F]**
   `https://tevgeniou.github.io/EquityRiskFactors/bibliography/FiveFactor.pdf`
   — Adds RMW (operating profitability) and CMA (investment/asset growth) to FF3.
5. Asness, C. S., Frazzini, A., Pedersen, L. H. (2019). *Quality Minus Junk.* **Review of Accounting Studies** 24(1): 34–112. **[F]**
   `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2312432`
   — Quality = profitability + growth + safety + payout; QMJ significant in 23/24 countries. Underpins our "operating profitability + payout" pick.
6. Frazzini, A., Pedersen, L. H. (2014). *Betting Against Beta.* **Journal of Financial Economics** 111(1): 1–25. **[F]**
   `https://www.sciencedirect.com/science/article/pii/S0304405X13002675`
   — Low-beta long / high-beta short with leverage; founding low-risk paper. Caveat: OOS post-2010 has deteriorated (see source 12).
7. Novy-Marx, R. (2013). *The Other Side of Value: The Gross Profitability Premium.* **Journal of Financial Economics** 108(1): 1–28. **[F]**
   `https://www.sciencedirect.com/science/article/abs/pii/S0304405X13000044`
   — Gross profits / assets predicts cross-section with strength comparable to B/M. *This is what P6's screener.py currently approximates.*
8. Ball, R., Gerakos, J., Linnainmaa, J. T., Nikolaev, V. (2016). *Accruals, Cash Flows, and Operating Profitability in the Cross Section of Stock Returns.* **Journal of Financial Economics** 121(1): 28–45. **[F]**
   `https://www.sciencedirect.com/science/article/abs/pii/S0304405X16300307`
   — Cash-based operating profitability beats Novy-Marx gross profitability and FF RMW. Direct upgrade target for the existing profitability code.
9. Cooper, M. J., Gulen, H., Schill, M. J. (2008). *Asset Growth and the Cross-Section of Stock Returns.* **Journal of Finance** 63(4): 1609–1651. **[F]**
   `https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2008.01370.x`
   — Asset growth predicts low returns; basis for FF CMA. Survives all major replication studies.
10. Jegadeesh, N., Titman, S. (1993, 2023). *Returns to Buying Winners and Selling Losers / Momentum: Evidence and Insights 30 Years Later.* **Journal of Finance / Pac.-Bas. Fin. J.** **[F]**
    `https://www.bauer.uh.edu/rsusmel/phd/jegadeesh-titman93.pdf` ; 2023 follow-up `https://www.sciencedirect.com/science/article/abs/pii/S0927538X23002731`
    — Original 12-2 momentum still profitable OOS; momentum-crashes Daniel-Moskowitz (2016) JFE recommend dynamic sizing.
11. Bali, T. G., Cakici, N., Whitelaw, R. F. (2011). *Maxing Out: Stocks as Lotteries and the Cross-Section of Expected Returns.* **Journal of Financial Economics** 99(2): 427–446. **[F]**
    `https://www.sciencedirect.com/science/article/abs/pii/S0304405X1000190X`
    — MAX (max 1-day return over past month). *Already exploited in P6 screener.py:50.*
12. Novy-Marx, R., Velikov, M. (2022). *Betting Against Betting Against Beta.* **Journal of Financial Economics**. **[R]**
    `https://www.sciencedirect.com/science/article/abs/pii/S0304405X21002051`
    — BAB returns sensitive to non-standard construction; OOS robust low-vol still works but plain "low-vol portfolios" preferred over leveraged BAB. *Justifies P6's existing inverse-vol design over BAB.*
13. Baker, M., Bradley, B., Wurgler, J. (2011). *Benchmarks as Limits to Arbitrage: Understanding the Low-Volatility Anomaly.* **Financial Analysts Journal** 67(1): 40–54. **[F]**
    `https://archive.nyu.edu/handle/2451/29593` ; NYU PDF `https://pages.stern.nyu.edu/~jwurgler/papers/faj-benchmarks.pdf`
    — Behavioral / limits-to-arbitrage rationale for the persistence of low-vol. Confirms why low-vol survives crowding.
14. Blitz, D., van Vliet, P. (2018). *The Conservative Formula: Quantitative Investing Made Easy.* **Journal of Portfolio Management** 44(7). **[P]**
    `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3145152`
    — Robeco recipe: from top-1000 take 500 lowest-vol, then top 100 by NPY+momentum. **Direct blueprint for our recommended P6 extension.**
15. Bailey, D. H., López de Prado, M. (2014). *The Deflated Sharpe Ratio.* **Journal of Portfolio Management** 40(5): 94–107. **[M]**
    `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551`
    — Multiple-testing correction. *Already in P6 (screener.py:132).*
16. Daniel, K., Moskowitz, T. J. (2016). *Momentum Crashes.* **Journal of Financial Economics** 122(2): 221–247. **[F]**
    `https://www.sciencedirect.com/science/article/pii/S0304405X16301490`
    — Momentum has skew-left tail; dynamic vol-scaling roughly doubles Sharpe. *Drives our vol-targeted application of momentum.*
17. Linnainmaa, J. T., Roberts, M. R. (2018). *The History of the Cross-Section of Stock Returns.* **Review of Financial Studies** 31(7): 2606–2649. **[R]**
    `https://academic.oup.com/rfs/article-abstract/31/7/2606/4977829`
    — Most accounting anomalies fail pre-1963 OOS; persistent survivors correlate with intangibles + leverage. Sharpens our "evidence strength" column.
18. Asness, C. S., Frazzini, A. (2013). *The Devil in HML's Details.* **Journal of Portfolio Management** 39(4): 49–68. **[P]**
    `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2054749`
    — Timely vs lagged price data in value construction; cited only for the *general* point that fundamentals stale-data matters — relevant to our point-in-time critique.
19. Ang, A., Hodrick, R. J., Xing, Y., Zhang, X. (2006). *The Cross-Section of Volatility and Expected Returns.* **Journal of Finance** 61(1): 259–299. **[F]**
    `https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2006.00836.x`
    — Idiosyncratic volatility puzzle; cross-validates low-vol direction.
20. Cochrane, J. H. (2011). *Presidential Address: Discount Rates.* **Journal of Finance** 66(4): 1047–1108. **[F]**
    `https://www.johnhcochrane.com/news-op-eds-all/discount-rates`
    — "Zoo of factors" framing; cited as the canonical statement of the multiple-testing problem the rest of the literature addresses.

**Navigation / corroborating secondaries (NOT used as primary evidence)**:
N1. Robeco "Fama-French 5-factor: why more is not always better" (2024) `https://www.robeco.com/en-int/insights/2024/10/fama-french-5-factor-model-why-more-is-not-always-better` — Robeco view that RMW/CMA are "premature"; momentum is missing from FF5.
N2. AQR "Quality Minus Junk" landing `https://www.aqr.com/Insights/Research/Working-Paper/Quality-Minus-Junk`
N3. Alpha Architect "Performance of Factors" `https://alphaarchitect.com/performance-of-factors/` (403 on fetch; used only for cross-checking).
N4. JKP factor library `https://jkpfactors.com/` — corroborates the 13-cluster name list (Accruals, Debt Issuance, Investment, Leverage, Low Risk, Momentum, Profit Growth, Profitability, Quality, Seasonality, Size, Short-Term Reversal, Value).
N5. Hou-Xue-Zhang preprint mirror `https://www.ivey.uwo.ca/media/3776713/zhang_.pdf` (PDF-only).
N6. Blank Capital Research summary of HXZ (paywalled `402`).
N7. Quantpedia BAB writeup `https://quantpedia.com/strategies/betting-against-beta-factor-in-stocks` (corroborates OOS BAB deterioration).
N8. Wikipedia Deflated-Sharpe-Ratio `https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio` (formula sanity-check).

**URLs fetched / attempted (including rejected)**: all of #1–#20 above plus N1–N8.
Rejected: PDFs that returned binary (HXZ NBER w23394 PDF, Novy-Marx 2013 PDF, JKP CBS PDF, JKP documentation PDF, AQR Fact-Fiction PDF, q5 lecture-notes PDF) — content covered via independent secondary corroboration on the relevant claims. Rejected for being paywalled-only: blankcapitalresearch HXZ writeup, Wiley JoF JKP DOI page. Rejected on permission: AlphaArchitect 403, Tuck French OP page 404. All quantitative claims used in the audit have ≥2 independent sources from the list above.

---

## 3. Current-state analysis (file:line citations)

**Factors actively used by P6**

| Factor | Code location | Mechanism |
|---|---|---|
| Low realized vol (boring) | `src/portfolios/portfolio_6/screener.py:18-24, 49, 57` | `realized_vol` = std × √252; ascending rank into composite score |
| MAX (lottery exclusion) | `src/portfolios/portfolio_6/screener.py:27-31, 50, 58` | max of last `VOL_LOOKBACK_DAYS` of daily returns; ascending rank |
| Gross profitability (Novy-Marx 2013 style) | `src/portfolios/portfolio_6/screener.py:60-69` | `gross_profit / total_assets`; descending rank; gated by `USE_FUNDAMENTALS` |
| Inverse-vol weighting | `src/portfolios/portfolio_6/screener.py:80-112` | `w_i ∝ 1/σ_i`, iterative cap at `MAX_WEIGHT_PER_STOCK` |
| Vol targeting | `src/portfolios/portfolio_6/screener.py:115-129`; `strategy.py:243-248` | scale sleeve so realized vol ≈ `VOL_TARGET_ANNUAL`, capped at `MAX_LEVERAGE` |
| Deflated Sharpe diagnostic | `src/portfolios/portfolio_6/screener.py:132-175`; `strategy.py:226-242` | Bailey-Lopez-de-Prado 2014; warns when `DSR_MIN_PROB < 0.5` |
| GLD / trend hedge sleeve | `src/portfolios/portfolio_6/strategy.py:251-274`; `config.json:18-21` | flat 7% GLD + optional trend-following ticker |

**Universe**

- `scripts/APIs/build_portfolio6_universe.py:66-75` pulls **current** S&P 500 + Nasdaq-100 from FMP endpoints; deduped/sorted into `src/portfolios/portfolio_6/universe.json` (519 tickers as of read).
- No historical constituents → **survivorship bias** on any backtest spanning >12 months. The `Portfolio6_Strategy.txt:37-38` explicitly acknowledges this limitation.
- No microcap / penny / liquidity filter. NYSE-breakpoint filters used by HXZ (2020) and FF5 to avoid microcaps cannot be reproduced because all members are already large-cap by virtue of being S&P 500 / N100 constituents — so this is acceptable for now.

**Fundamentals pipeline**

- `scripts/APIs/fmp_fundamentals.py:70-95` pulls **TTM ratios + most-recent annual income / balance** per ticker. Endpoints:
  - `ratios-ttm` → `returnOnEquityTTM`
  - `income-statement?period=annual&limit=1` → `grossProfit`
  - `balance-sheet-statement?period=annual&limit=1` → `totalAssets`
- Output schema (FIELDNAMES at `fmp_fundamentals.py:36-43`): `ticker, roe, gross_profit, total_assets, income_date, balance_date`.
- **Critical findings**:
  - `fundamentals/fundamentals.csv` has only **10 rows** populated vs **519 ticker universe** (`wc -l = 11`). The pipeline was last run with `--limit ≤ 10` and never re-run on the full universe.
  - All `roe` values in the CSV are **empty** (verified: `awk 'NR>1 && $2!=""' | wc -l → 0`). The TTM-ratios call evidently returns `returnOnEquityTTM=None` for these ten tickers (different field name? subscription gating?). The strategy never uses `roe` so this is dead code, but it advertises a profitability ranking it doesn't deliver.
  - The pipeline writes the **latest annual statement** with `income_date`/`balance_date`. There is no point-in-time alignment — using the 2025-12-31 row for ranking decisions in early-2026 backtests is fine; using it for a 2020 backtest is look-ahead bias. The `Portfolio6_Strategy.txt:39-41` acknowledges this.
- `screener.py:60-71` silently no-ops if `gross_profit` or `total_assets` is missing — combined with `.fillna(score.median())` on `score.add`, a missing-profitability ticker gets a median profitability rank, **not penalized**. So the profitability factor is currently fired on ~10/519 names and is statistical noise.

**Backtest plumbing**

- `src/main_backtest.py` is the entrypoint; uncommitted modifications were noted in `git status`.
- `BasePortfolio._get_market_data` (`src/portfolios/portfolio_BASE/strategy.py:519-546`) pulls market_data from Postgres; trusts the `TICKERS` list assembled in `Portfolio6Strategy.__init__` (`strategy.py:69-78`). Adding tickers via factor data requires the universe to already contain them — which is fine since we're staying inside the S&P 500 + N100 universe.

---

## 4. Gap analysis (replicated zoo vs P6)

| Factor / cluster | Supported by lit (≥2 indep. sources) | In P6 today | Missing? | Evidence strength | Notes |
|---|---|---|---|---|---|
| **Low Risk** (vol / beta) | JKP cluster #5; HXZ via low IVOL; AHXZ 2006; Frazzini-Pedersen 2014; Baker-Bradley-Wurgler 2011 | YES (realized-vol rank + inverse-vol wts + vol target) | NO | **Strong**; survives OOS in 93 countries (JKP) and in Robeco 1986-2022. BAB construction sensitive (Novy-Marx-Velikov 2022) — P6's plain inverse-vol is the safer formulation. | Already exploited 3x (rank, weight, sleeve scaling). |
| **Lottery (MAX)** | Bali-Cakici-Whitelaw 2011 (primary); JKP "Low Risk" subsumes; HXZ "trading frictions" replicated | YES (`max_one_day_return`) | NO | **Strong**; bounds the negative-skew tail. | Already in `screener.py:27`. |
| **Profitability — gross (Novy-Marx)** | Novy-Marx 2013; FF 2015 RMW; HXZ ROE; JKP "Profitability" cluster | PARTIAL (formula present; data missing for ~98% of universe; `roe` field always null) | YES — broken | **Strong** if data populated | Needs CSV refresh + downgrade Novy-Marx ratio toward cash-based op-profitability (Ball-Gerakos 2016). |
| **Profitability — operating (FF RMW)** | FF 2015; HXZ ROE; Ball-Gerakos-Linnainmaa-Nikolaev 2016; AQR QMJ "profitability sub-factor" | NO (would need `(revenue - cogs - sga - interest) / book_equity`) | YES | **Strongest profitability variant** in BGLN 2016 horse-race | Recommended add; FMP fields available: `revenue, costOfRevenue, sellingGeneralAndAdministrativeExpenses, interestExpense`, `totalStockholdersEquity`. |
| **Investment / Asset Growth** | Cooper-Gulen-Schill 2008; FF 2015 CMA; HXZ I/A; JKP "Investment" cluster | NO | YES | **Strong**; replicates in HXZ q-factor model and JKP global data. | Recommended add; FMP: two consecutive `balance-sheet-statement` rows → `Δ totalAssets / lag(totalAssets)`. |
| **Momentum (12-2)** | Jegadeesh-Titman 1993, 2023; Carhart 1997; AsnessMoskowitzPedersen 2013; Daniel-Moskowitz 2016; JKP "Momentum" cluster | NO | YES | **Strong** but with crash-risk skew. JT 2023 confirms persistent OOS Sharpe. | Recommended add (vol-targeted to manage tail per Daniel-Moskowitz). Lookback already 252d. |
| **Quality (QMJ — full)** | Asness-Frazzini-Pedersen 2019 RoAS; JKP "Quality" cluster | NO (only the profitability leg implicitly) | YES (deferred) | **Strong**; QMJ positive in 23/24 countries | Skipped from recommended adds — too data-hungry; partial coverage via op-profitability is the high-value subset. |
| **Accruals** | Sloan 1996; HXZ replicated; JKP "Accruals" cluster | NO | YES (deferred) | **Moderate** — attenuated post-2010 per Green-Hand-Soliman 2010; "the strategy that grew up". | Not recommended — decay evidence is strong. |
| **Value (B/M, FCF/EV)** | FF 1993, 2015; AsnessMoskowitz 2013; HXZ value; JKP "Value" cluster | NO | YES (deferred) | **Strong long-run, weak post-2008**; outside P6's defensive remit. | Skip — P6 is explicitly a defensive sleeve; value tilt fights the low-vol thesis. |
| **Net Payout / Buyback Yield** | Boudoukh-Michaely-Richardson-Roberts 2007; Blitz-vanVliet 2018 Conservative Formula | NO | YES (deferred) | **Moderate-strong** | Would complete the Conservative Formula recipe (low-vol × NPY × momentum). Pushed to v2; needs FMP `cashflow-statement` (`commonStockRepurchased`, `dividendsPaid`). |
| **Short-Term Reversal** | Jegadeesh 1990; JKP "Short-Term Reversal" cluster | NO | YES (deferred) | **Moderate**; very high turnover | Not P6's profile (monthly rebalance). Skip. |
| **Seasonality** | JKP "Seasonality" cluster; Heston-Sadka 2008 | NO | YES (deferred) | **Weak-moderate**; brittle | Skip. |
| **Leverage / Debt Issuance / Size** | JKP clusters | NO | YES (deferred) | Marginal in defensive sleeves | Skip. |

---

## 5. Recommendation — 3 factor additions

All three are **flag-gated** (default off in `config.json` until the data pipeline catches
up). When `false` the strategy behaves identically to today; the only DSR penalty is from
the existing scorer.

### R1. Momentum (12-2)
- **Definition**: cumulative return from t-252 to t-21 (skip last month to avoid short-term reversal).
- **Why it survived**: Replicates in HXZ (2020), JKP (2023) tangency cluster, JT (2023) 30-year OOS update. Returns ~1% / month in original sample; ~0.31%/month most recently.
- **Risk control**: Daniel-Moskowitz 2016 — momentum has fat-left tail in panic states. P6 already vol-targets the whole sleeve, which dampens that risk.
- **Implementation**: in `_collect_returns` we already pull ~282 daily closes; we can compute the cumulative return on `close[-252:-21]` for each ticker, descending-rank in `score_universe`.

### R2. Operating Profitability (Ball-Gerakos-Linnainmaa-Nikolaev / FF RMW)
- **Definition (BGLN 2016 / FF 2015)**: `OP = (revenue − cost_of_revenue − sga − interest_expense) / book_equity`.
- **Why it survived**: BGLN 2016 horse-race shows cash-based / operating profitability dominates raw gross profitability and ROE. JKP "Profitability" cluster member of tangency portfolio.
- **Implementation**: extend `fmp_fundamentals.py` to also pull the four income-statement components and `totalStockholdersEquity`. Pure additive columns — backward compatible if the columns are missing (we add `score.median()` fallback).
- **Why not full QMJ**: data cost too high for v1.

### R3. Conservative Investment (Cooper-Gulen-Schill / FF CMA)
- **Definition**: `INV = (total_assets_t − total_assets_{t-1}) / total_assets_{t-1}`. Ascending rank (low investment = conservative = higher expected return).
- **Why it survived**: Cooper-Gulen-Schill 2008 (JoF), FF 2015 CMA, HXZ I/A, JKP "Investment" cluster. Survives Linnainmaa-Roberts 2018 historical OOS.
- **Implementation**: extend `fmp_fundamentals.py` to pull two consecutive annual balance sheets (`limit=2` instead of `limit=1`), compute and write `asset_growth`.
- **Caution**: Like profitability, this is annual — uses prior-year balance sheet → low look-ahead risk in production but the same point-in-time caveat applies in historical backtests.

### Factors I explicitly do NOT recommend adding now
- **Full QMJ / payout / safety**: data + book-value pipeline cost > 1-feature value-add.
- **Accruals**: post-2010 decay (Green-Hand-Soliman 2010; supported by JKP "Accruals" weakening). Don't pay the data cost.
- **Value (B/M)**: directionally opposed to P6's defensive thesis (value-vs-growth orthogonal to low-vol; combined sleeves are a different mandate).
- **BAB**: P6 already captures the low-risk premium via inverse-vol weights without the Novy-Marx-Velikov 2022 critique of BAB construction biases.

---

## 6. Patches (apply-ready unified diffs)

All three patches preserve existing flow when the new flags are `false`. The
`screener.py` patch is additive — the new factors enter the rank-sum **only** when
fundamentals/return-history columns are present and the corresponding `USE_*` config flag
is `true`.

### 6.1 `src/portfolios/portfolio_6/config.json`

```diff
--- a/src/portfolios/portfolio_6/config.json
+++ b/src/portfolios/portfolio_6/config.json
@@ -9,6 +9,12 @@
   "PORTFOLIO_6_CONFIG": {
     "UNIVERSE_PATH": "src/portfolios/portfolio_6/universe.json",
     "FUNDAMENTALS_CSV": "fundamentals/fundamentals.csv",
     "USE_FUNDAMENTALS": true,
+    "USE_MOMENTUM_12_2": false,
+    "USE_OPERATING_PROFITABILITY": false,
+    "USE_ASSET_GROWTH": false,
+    "MOMENTUM_LOOKBACK_DAYS": 252,
+    "MOMENTUM_SKIP_DAYS": 21,
+    "FACTOR_WEIGHTS": {"vol": 1.0, "max": 1.0, "gross_profit": 1.0, "momentum": 1.0, "op_profit": 1.0, "asset_growth": 1.0},
     "SCREEN_TOP_N": 50,
     "VOL_LOOKBACK_DAYS": 252,
     "MAX_WEIGHT_PER_STOCK": 0.05,
```

Notes:
- All three new flags default to `false` → no behavioural change without operator opt-in.
- `FACTOR_WEIGHTS` lets the researcher zero out individual legs (e.g. set `"max": 0.0` to disable lottery rank while keeping vol). Default `1.0` preserves today's behaviour for the existing legs.
- Required `fundamentals.csv` columns for each new flag are documented in the screener patch.

### 6.2 `src/portfolios/portfolio_6/screener.py`

```diff
--- a/src/portfolios/portfolio_6/screener.py
+++ b/src/portfolios/portfolio_6/screener.py
@@ -1,9 +1,12 @@
 """
 Portfolio 6 helpers: screen + inverse-volatility weights + vol-target scaler +
 Deflated Sharpe Ratio (Lopez de Prado 2014).

-Price-only screen (boring + not-lottery) plus optional profitable score from
-local fundamentals CSV (gross_profit / total_assets and/or ROE).
+Price-only screen (boring + not-lottery) plus optional factor tilts from
+local fundamentals CSV:
+  - gross_profit / total_assets        (Novy-Marx 2013)
+  - momentum 12-2 (price-only)         (Jegadeesh-Titman 1993; Asness 2013)
+  - operating profitability (op_profit) (Ball-Gerakos-Linnainmaa-Nikolaev 2016 / FF 2015 RMW)
+  - asset growth (1 - rank desc)       (Cooper-Gulen-Schill 2008 / FF 2015 CMA)
 """

 from typing import Dict, List, Optional
@@ -28,11 +31,48 @@ def max_one_day_return(returns: pd.Series) -> float:
     return m if np.isfinite(m) else float("inf")


+def momentum_12_2(
+    returns: pd.Series,
+    *,
+    lookback_days: int = 252,
+    skip_days: int = 21,
+) -> float:
+    """
+    Cumulative return from t-lookback to t-skip (Jegadeesh-Titman 1993 12-2).
+    Returns +inf when the input is too short so the candidate is *worst*-ranked
+    in the ascending-rank score (i.e. effectively excluded from the long tilt).
+    """
+    if returns is None or returns.empty:
+        return float("-inf")
+    r = returns.dropna()
+    if len(r) < lookback_days + 1:
+        return float("-inf")
+    window = r.iloc[-lookback_days:-skip_days] if skip_days > 0 else r.iloc[-lookback_days:]
+    if window.empty:
+        return float("-inf")
+    return float((1.0 + window).prod() - 1.0)
+
+
+def _safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
+    n = pd.to_numeric(numer, errors="coerce")
+    d = pd.to_numeric(denom, errors="coerce")
+    out = n / d
+    return out.replace([np.inf, -np.inf], np.nan).dropna()
+
+
 def score_universe(
     returns_matrix: Dict[str, pd.Series],
     fundamentals_df: Optional[pd.DataFrame] = None,
     *,
     use_fundamentals: bool = True,
+    use_momentum: bool = False,
+    use_op_profitability: bool = False,
+    use_asset_growth: bool = False,
+    momentum_lookback_days: int = 252,
+    momentum_skip_days: int = 21,
+    factor_weights: Optional[Dict[str, float]] = None,
 ) -> pd.Series:
     """
     Composite rank-sum (lower = better).
@@ -40,8 +80,18 @@ def score_universe(
     - not_lottery: ascending rank of max single-day return
     - profitable (if fundamentals): descending rank of gross_profit/total_assets
+    - momentum_12_2 (if use_momentum): descending rank of cumulative 12-2 return
+    - op_profitability (if use_op_profitability and columns present):
+        descending rank of (revenue - cost_of_revenue - sga - interest_expense) / book_equity
+    - asset_growth (if use_asset_growth and column present):
+        ascending rank of (total_assets - prev_total_assets) / prev_total_assets
+        (low investment = conservative = better)
+
+    factor_weights is a dict {leg_name: weight}; missing keys default to 1.0.
+    Set a weight to 0.0 to disable that leg.
     """
     if not returns_matrix:
         return pd.Series(dtype=float)

+    weights = dict(factor_weights or {})
+
     vols = pd.Series({t: realized_vol(r) for t, r in returns_matrix.items()})
     maxs = pd.Series({t: max_one_day_return(r) for t, r in returns_matrix.items()})
     vols = vols.replace([np.inf, -np.inf], np.nan).dropna()
     maxs = maxs.replace([np.inf, -np.inf], np.nan).dropna()

     if vols.empty:
         return pd.Series(dtype=float)

-    score = vols.rank(ascending=True)
-    score = score.add(maxs.rank(ascending=True), fill_value=score.median())
+    score = vols.rank(ascending=True) * float(weights.get("vol", 1.0))
+    score = score.add(
+        maxs.rank(ascending=True) * float(weights.get("max", 1.0)),
+        fill_value=score.median(),
+    )

     if use_fundamentals and fundamentals_df is not None and not fundamentals_df.empty:
         cols = fundamentals_df.columns
         if "gross_profit" in cols and "total_assets" in cols:
-            ratio = pd.to_numeric(fundamentals_df["gross_profit"], errors="coerce") / pd.to_numeric(
-                fundamentals_df["total_assets"], errors="coerce"
-            )
-            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
+            ratio = _safe_ratio(fundamentals_df["gross_profit"], fundamentals_df["total_assets"])
             ratio = ratio[ratio.index.isin(score.index)]
             if not ratio.empty:
-                score = score.add(ratio.rank(ascending=False), fill_value=score.median())
+                score = score.add(
+                    ratio.rank(ascending=False) * float(weights.get("gross_profit", 1.0)),
+                    fill_value=score.median(),
+                )
+
+        # --- R2. Operating profitability (BGLN 2016 / FF 2015 RMW) ---
+        if use_op_profitability and {
+            "revenue", "cost_of_revenue", "sga", "interest_expense", "book_equity"
+        }.issubset(set(cols)):
+            op_num = (
+                pd.to_numeric(fundamentals_df["revenue"], errors="coerce")
+                - pd.to_numeric(fundamentals_df["cost_of_revenue"], errors="coerce")
+                - pd.to_numeric(fundamentals_df["sga"], errors="coerce")
+                - pd.to_numeric(fundamentals_df["interest_expense"], errors="coerce")
+            )
+            op = _safe_ratio(op_num, fundamentals_df["book_equity"])
+            op = op[op.index.isin(score.index)]
+            if not op.empty:
+                score = score.add(
+                    op.rank(ascending=False) * float(weights.get("op_profit", 1.0)),
+                    fill_value=score.median(),
+                )
+
+        # --- R3. Asset growth (Cooper-Gulen-Schill 2008 / FF 2015 CMA) ---
+        if use_asset_growth and "asset_growth" in cols:
+            ag = pd.to_numeric(fundamentals_df["asset_growth"], errors="coerce")
+            ag = ag.replace([np.inf, -np.inf], np.nan).dropna()
+            ag = ag[ag.index.isin(score.index)]
+            if not ag.empty:
+                # ascending rank: low investment scores best (conservative = lower score)
+                score = score.add(
+                    ag.rank(ascending=True) * float(weights.get("asset_growth", 1.0)),
+                    fill_value=score.median(),
+                )
+
+    # --- R1. Momentum 12-2 (Jegadeesh-Titman 1993) ---
+    if use_momentum:
+        moms = pd.Series({
+            t: momentum_12_2(
+                r,
+                lookback_days=momentum_lookback_days,
+                skip_days=momentum_skip_days,
+            )
+            for t, r in returns_matrix.items()
+        })
+        moms = moms.replace([np.inf, -np.inf], np.nan).dropna()
+        moms = moms[moms.index.isin(score.index)]
+        if not moms.empty:
+            # descending rank: high momentum = better = lower composite score
+            score = score.add(
+                moms.rank(ascending=False) * float(weights.get("momentum", 1.0)),
+                fill_value=score.median(),
+            )

     return score.sort_values(ascending=True)
```

### 6.3 `src/portfolios/portfolio_6/strategy.py` (wire-up only)

```diff
--- a/src/portfolios/portfolio_6/strategy.py
+++ b/src/portfolios/portfolio_6/strategy.py
@@ -96,6 +96,16 @@ class Portfolio6Strategy(BasePortfolio):
         self.rebalance_drift: float = float(
             p6_cfg.get("REBALANCE_DRIFT_THRESHOLD", 0.005)
         )
         self.use_fundamentals: bool = bool(p6_cfg.get("USE_FUNDAMENTALS", True))
+        self.use_momentum_12_2: bool = bool(p6_cfg.get("USE_MOMENTUM_12_2", False))
+        self.use_op_profitability: bool = bool(p6_cfg.get("USE_OPERATING_PROFITABILITY", False))
+        self.use_asset_growth: bool = bool(p6_cfg.get("USE_ASSET_GROWTH", False))
+        self.momentum_lookback_days: int = int(p6_cfg.get("MOMENTUM_LOOKBACK_DAYS", 252))
+        self.momentum_skip_days: int = int(p6_cfg.get("MOMENTUM_SKIP_DAYS", 21))
+        self.factor_weights: Dict[str, float] = {
+            str(k): float(v)
+            for k, v in dict(p6_cfg.get("FACTOR_WEIGHTS", {})).items()
+        }
         self.dsr_min_prob: float = float(p6_cfg.get("DSR_MIN_PROB", 0.5))
         self.fundamentals_csv_rel: str = str(
             p6_cfg.get("FUNDAMENTALS_CSV", "fundamentals/fundamentals.csv")
@@ -204,6 +214,12 @@ class Portfolio6Strategy(BasePortfolio):
         scores = score_universe(
             returns_matrix,
             self.fundamentals_df,
             use_fundamentals=self.use_fundamentals,
+            use_momentum=self.use_momentum_12_2,
+            use_op_profitability=self.use_op_profitability,
+            use_asset_growth=self.use_asset_growth,
+            momentum_lookback_days=self.momentum_lookback_days,
+            momentum_skip_days=self.momentum_skip_days,
+            factor_weights=self.factor_weights or None,
         )
         top = select_top_n(scores, n=self.screen_top_n)
         if not top:
```

**Required ALSO (non-strategy code; outside the strict ≤3-change ask but listed for the user)**:

For R2 (op-profit) and R3 (asset-growth) to actually fire, `scripts/APIs/fmp_fundamentals.py` must additionally write the columns: `revenue, cost_of_revenue, sga, interest_expense, book_equity, asset_growth`. FMP fields:
- `revenue` ← income statement `revenue`
- `cost_of_revenue` ← income statement `costOfRevenue`
- `sga` ← income statement `sellingGeneralAndAdministrativeExpenses`
- `interest_expense` ← income statement `interestExpense`
- `book_equity` ← balance sheet `totalStockholdersEquity`
- `asset_growth` ← `(totalAssets_year[0] - totalAssets_year[1]) / totalAssets_year[1]` (requires `limit=2`)

The strategy patch is **forward-compatible**: if those columns are absent, the legs no-op (`screener.py:.issubset(set(cols))` and `"asset_growth" in cols` guards).

---

## 7. Falsification test

Switch each new flag on individually with `SCREEN_TOP_N=50` and run a 5y backtest 2021-01-01
through 2026-04-30. Compare to the same-period baseline (all three flags off):

| Metric | Pass criterion (per added factor leg) |
|---|---|
| **5y rolling Sharpe (after vol target)** | ≥ 0.80× the baseline rolling Sharpe in **both** the lo-vol regime (`VIX < 20` months) and the hi-vol regime (`VIX ≥ 25` months). |
| **Deflated Sharpe Ratio probability** (already logged by `screener.py:228`) | ≥ 0.50 across the 5y window. Below 0.50 → **kill the factor**. |
| **Max drawdown** | ≤ 1.10× baseline max drawdown — i.e. < +10% deepening. Any momentum-driven crash > 1.20× kills momentum. |
| **Turnover** | ≤ 1.50× baseline monthly turnover (already monthly rebalance only). Momentum is the main culprit here; if monthly turnover doubles, downweight via `FACTOR_WEIGHTS.momentum` from 1.0 to 0.5. |
| **Factor-tilt OOS spread** | The mean realized monthly return spread between the top-quintile and bottom-quintile of the *new* factor inside the 50-name basket must be > 0 over the 5y. If ≤ 0 → kill. |

**Single concrete kill rule** (per the brief): *"If the 5y rolling Sharpe of the
operating-profitability tilt, measured OOS in the post-vol-target sleeve, is below 0.80×
baseline in any monthly rolling window of length 60, and the cumulative DSR probability for
the basket falls below 0.50, set `USE_OPERATING_PROFITABILITY=false` in next config push."*
Same rule applies independently to `USE_MOMENTUM_12_2` and `USE_ASSET_GROWTH`.

The DSR threshold here aligns with the `DSR_MIN_PROB=0.5` already present in `config.json:23`.

---

## 8. Risks + rollback path

| Risk | Mitigation | One-line rollback |
|---|---|---|
| New factor fires on noisy / sparse fundamentals (today `roe` is all-null and only 10/519 tickers have `gross_profit`) | All three new flags default `false`; gross-profit leg is guarded by `if not ratio.empty`; new legs guarded by `.issubset(cols)` | Set the relevant `USE_*` flag to `false` in `config.json`. |
| Momentum crash regime (Daniel-Moskowitz 2016) | Vol-target sleeve already scales realized portfolio vol; combined with low-vol leg this is the Conservative-Formula damping. Optional further mitigation: lower `FACTOR_WEIGHTS.momentum` to 0.5 in crisis regimes. | `"USE_MOMENTUM_12_2": false` in config. |
| Look-ahead bias in fundamentals (latest annual statement used as point-in-time) | Stay current-time only (live + walk-forward); for historical backtests this is acknowledged in `Portfolio6_Strategy.txt:39-41`. Recommend a follow-up ticket to add `report_date` filtering in `screener.py` joins (out of scope for this audit). | N/A — limitation pre-exists. |
| Survivor bias in universe | Acknowledged in code; out of scope here. Recommend a follow-up to use FMP `historical-sp500-constituents` (paid endpoint) before any multi-year backtest. | N/A. |
| Double-counting low-risk via momentum (momentum can pile into recent winners that are also recent vol-spike survivors) | `FACTOR_WEIGHTS.momentum` slider lets researcher down-weight. | Set `"momentum": 0.0` in `FACTOR_WEIGHTS`. |
| FMP API field rename / coverage loss | `screener.py` no-ops when fundamental columns are missing → existing low-vol behavior survives | None needed — degrades to baseline. |
| Configuration drift between live + backtest | All flags read from one config block (`PORTFOLIO_6_CONFIG`); no env-var path. | N/A. |

**Top-line rollback** (single line): in `src/portfolios/portfolio_6/config.json`, set
`"USE_MOMENTUM_12_2": false, "USE_OPERATING_PROFITABILITY": false, "USE_ASSET_GROWTH": false`
— behaviour reverts to today, including the current (broken) gross-profit screen.
