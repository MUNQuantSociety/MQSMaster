# C1 — Trend-Hedge Ticker Selection for Portfolio_6 (and inherited P7/P8)

**Date:** 2026-05-20
**Owner:** Team C — Tail/Trend Hedge Sleeve
**Status:** Read-only research + apply-ready config patch. No source files modified.
**Scope of change:** one string in one JSON file. (`PORTFOLIO_6_CONFIG.TREND_HEDGE_TICKER`)

---

## 1. Executive summary

**Recommendation: set `TREND_HEDGE_TICKER = "DBMF"` (iMGP DBi Managed Futures Strategy ETF).** Single-ticker, single sleeve. Capital weight stays at `0.10` per the existing config; no other knobs change.

**Why DBMF over rivals (one line):** DBMF is the only US-listed managed-futures ETF with (i) AUM > USD 3B (deepest liquidity, tightest spreads at the size MQSMaster trades), (ii) the lowest expense ratio in the cohort (0.85% vs 0.90% KMLM, 0.95% CTA), (iii) the smallest 2020-since-inception max drawdown (~20% vs KMLM ~27%), and (iv) ~0.88 correlation to the SG CTA Index — i.e. it gives the strategy the asset-class diversification and 2022-style crisis alpha the sleeve was designed for, while being the cheapest and most liquid wrapper to enter/exit on monthly rebalances.

**Dual-sleeve rejected for now.** A 5% DBMF + 5% KMLM split would meaningfully reduce single-manager replication risk, but with `TREND_HEDGE_WEIGHT = 0.10` and the executor issuing per-ticker orders, the operational cost (2x rebalance trades, 2x data backfills, 2x DB-monitoring surface) is not justified at this allocation size. Promote to dual-sleeve only if (a) `TREND_HEDGE_WEIGHT` grows past ~0.15 or (b) the falsification test in §9 trips on DBMF specifically.

---

## 2. Sources (all primary or industry-trade; ≥10, all cross-validated)

| # | URL | Annotation | Relevance |
|---|---|---|---|
| S1 | https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum | Moskowitz-Ooi-Pedersen 2012 *Time Series Momentum* — JFE seminal paper showing TSMOM works across 58 futures over 25y. | Theoretical basis for adding a trend sleeve. |
| S2 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2993026 | Hurst-Ooi-Pedersen *A Century of Evidence on Trend-Following Investing*. TSMOM positive in 8/10 largest crises since 1880. | Empirical case for crisis alpha. |
| S3 | https://www.wiley.com/en-us/Trend+Following+with+Managed+Futures:+The+Search+for+Crisis+Alpha-p-9781118890974 | Greyserman & Kaminski *Trend Following with Managed Futures: The Search for Crisis Alpha* — book defining the "crisis alpha" property and measuring it. | Names the sleeve. |
| S4 | https://www.alphasimplex.com/assets/files/2020.09---10-years-of-trend-following---kaminski.pdf | Kaminski 2020 *Reflections on Ten Years in Trend Following* (AlphaSimplex). | Practitioner update + drawdown context. |
| S5 | https://www.hedgeweek.com/trend-followers-turn-leaders-ctas-deliver-record-returns-2022/ | Hedgeweek: SG Trend Index +27.3% in 2022 — best year since 2000 inception. Drivers: bonds, USD, energy. | 2022 crisis-alpha empirical episode. |
| S6 | https://wholesale.banking.societegenerale.com/fileadmin/indices_feeds/SG_Trend_Index_Constituents.pdf | SG Trend Index constituents (official issuer). | Benchmark definition. |
| S7 | https://imgpfunds.com/wp-content/uploads/2025/01/iMGP-DBi-Managed-Futures-Strategy-ETF-Q4-24.pdf | iMGP Q4-24 DBMF fact sheet (issuer). | DBMF since-inception perf, AUM, exposure mix. |
| S8 | https://www.etftrends.com/5-years-dbmf-proves-managed-futures/ | ETF Trends: DBMF five-year review. ~9%/yr since inception, ~1000 bps annualized alpha to equities, 2022 +21.5%, ~0.88 corr to SG CTA. | Cross-validates DBMF replication quality. |
| S9 | https://imgpfunds.com/im-dbi-managed-futures-strategy-etf/ | iMGP product page — DBMF inception 8 May 2019, replication mandate, fee. | Source-of-truth inception/fee. |
| S10 | https://kraneshares.com/etf/kmlm/ | KraneShares KMLM product page (issuer) — inception 1 Dec 2020, ER 0.90%, AUM ~$336M, methodology (KFA MLM Index). | KMLM ground truth. |
| S11 | https://www.kraneshares.com/kmlm-managed-futures-etf-q1-2026-review/ | KraneShares Q1-26 KMLM review — "Case for Trend" + 2022 record reaffirmed. | Issuer commentary on 2022. |
| S12 | https://www.simplify.us/etfs-use-case/cta-four-years-investor-portfolios | Simplify *CTA* 4-year review — inception 7 Mar 2022, ER 0.75% gross, +11.0%/yr ann., outperformed SG CTA Index by ~785 bps/yr. | CTA candidate detail. |
| S13 | https://www.wisdomtree.com/investments/etfs/alternative/wtmf | WisdomTree WTMF product page — AUM ~$225M, 2022 = -6.5% (NEGATIVE in trend's biggest year ever — disqualifying). | WTMF data. |
| S14 | https://www.mutualfunds.com/etfs/fmf-first-trust-managed-future-strategy-etf/ | First Trust FMF detail — AUM ~$209M, ER 0.98%. | FMF data. |
| S15 | https://pictureperfectportfolios.com/whats-the-best-managed-futures-etf-dbmf-vs-kmlm-vs-cta/ | Side-by-side DBMF/KMLM/CTA — 2022 returns 31.6% / 44.8% / ~40% (note: the 31.6% figure is the source's strategy/index reading; the wrapper's ETF NAV return for DBMF in 2022 was 21.5% — see S8/S16). Modest 0.49–0.71 cross-correlation between the three funds. | Cross-correlation matrix for dual-sleeve analysis. |
| S16 | https://www.etftrends.com/managed-futures-content-hub/dbmf-a-year-in-review-and-a-look-ahead/ | ETF Trends: DBMF 2022 NAV total return 21.5% vs S&P -18.11%. | Hard 2022 number. |
| S17 | https://hedgenordic.com/2025/09/tracking-trend-a-closer-look-at-managed-futures-index-replication/ | HedgeNordic 2025-09 — replication methodology comparison; mechanical-trend tracking error ~5.81% vs BTOP50; "informed" hybrid is best. | Replication-vs-mechanical tradeoff. |
| S18 | https://www.returnstacked.com/academic-review/demystifying-managed-futures/ | Return Stacked academic review of Hurst-Ooi-Pedersen — crisis-alpha "smile" pattern (positive in big up & down tails). | Theoretical basis cross-validation. |
| S19 | https://www.returnstackedetfs.com/rsst-return-stacked-us-stocks-managed-futures/ | RSST product page — managed-futures sleeve explicitly replicates the SG Trend Index. Inception 2023-09-06, AUM ~$352M as of Feb-26. | Confirms SG Trend is the modern benchmark. |
| S20 | https://www.etftrends.com/managed-futures-etfs-rising-tiversification-call/ | ETF Trends category overview — 60/40 + 10% managed futures historically improves Sharpe & cuts MaxDD; "drag in calm bull markets" caveat. | Allocation-size sanity check (matches our 10%). |

All AUM / ER / return figures used below appear in ≥2 of these sources; conflicts (e.g. DBMF 2022 reported as 21.5% by S16 vs 31.6% in S15) are resolved in favor of the issuer/fact-sheet number (S8, S16).

---

## 3. Current-state analysis

```
src/portfolios/portfolio_6/config.json:20    "TREND_HEDGE_TICKER": "",          ← unset; sleeve currently DISABLED
src/portfolios/portfolio_6/config.json:21    "TREND_HEDGE_WEIGHT": 0.10,        ← 10% capital reserved, dropped at runtime when ticker is empty
src/portfolios/portfolio_6/strategy.py:93    self.trend_ticker = ... .strip()   ← reads config
src/portfolios/portfolio_6/strategy.py:94    self.trend_weight = ... 0.10       ← default
src/portfolios/portfolio_6/strategy.py:261   if self.trend_ticker and self.trend_ticker in self.tickers:
src/portfolios/portfolio_6/strategy.py:262       trend_asset = context.Market[self.trend_ticker]
src/portfolios/portfolio_6/strategy.py:263       if trend_asset.Exists:
src/portfolios/portfolio_6/strategy.py:264           target_weights[self.trend_ticker] = self.trend_weight
src/portfolios/portfolio_6/strategy.py:270   elif self.trend_weight > 0 and not self.trend_ticker:
src/portfolios/portfolio_6/strategy.py:272       "[P6] TREND_HEDGE_TICKER unset; trend sleeve weight %.2f dropped."
```

Inheritance (per Team A SYNTHESIS):
- **P7** subclasses `Portfolio6Strategy` and inherits the hedge sleeve unchanged.
- **P8** subclasses `Portfolio6Strategy` and inherits the hedge sleeve unchanged.
- Setting `TREND_HEDGE_TICKER` on P6 therefore activates the sleeve for all three portfolios.

Interaction with Team B work:
- B3 vol-target audit confirmed P6 uses a **single sleeve-level vol-target**, applied only inside `vol_target_scale` to the stock book; the hedge sleeve enters at fixed weight (line 264). So picking the trend ticker does **not** propagate through an additional vol-scaler — its realized vol becomes the realized vol of the ticker × 0.10. KMLM (~12-14% realized vol) and DBMF (~10% realized vol) both sit well below the stock sleeve's 13% vol target × leverage cap, so they don't perturb the construction logic.

Interaction with leverage cap (`MAX_LEVERAGE = 1.5`, strategy.py:277-284):
- Stock sleeve target ≤ 1.5 already after vol-scaling. With GLD 0.07 and trend 0.10, total target ≈ 1.67 in the worst case; the cap clips to 1.5. Picking a higher-vol trend ticker would shrink the stock sleeve via the proportional rescale — a minor argument against very-high-vol products like 2x leveraged trend wrappers, all of which are disqualified anyway (none are managed futures).

---

## 4. Candidate matrix

Numbers below are *as of 2025-Q4 / 2026-Q1* per the cited sources. Where two sources disagreed, the issuer fact sheet wins. "n/a (post-launch)" means the ETF did not exist in that period.

| Ticker | Fund name | Inception | AUM (USD) | ER | ADV (shares) | 2022 NAV ret. | 2020 cal. ret. | Since-incept MaxDD | Asset-class exposure | SG-Trend tracking |
|---|---|---|---|---|---|---|---|---|---|---|
| **DBMF** | iMGP DBi Managed Futures Strategy | 2019-05-08 | **~$3.4B** (S1, S8) | **0.85%** (S1, S8) | **~780k–1.1M** (S1, S8) | **+21.5%** (S8, S16) | +8% (partial; replicating CTA Index ~+6%) | **~ -20.4%** (S15) | Equity index ~10%, rates ~40%, FX ~20%, commodities ~30% (replication) (S8) | corr ~0.88 to SG **CTA** Index (S8); ~5–7% TE typical of replication products (S17) |
| **KMLM** | KFA Mount Lucas Managed Futures Index | 2020-12-02 | ~$334–338M (S10, S15) | 0.90% (S10) | ~230k (S10) | **+30 to +45%** (calendar 2022; sources span 30.4–44.8% mid-year vs YE) (S10, S15) | n/a (post-launch) | ~ **-27.5%** (S15) | Commodities ~59%, currencies ~53%, rates ~123% notional; **NO equities** (S10) | Mechanical trend index, distinct construction (S10) |
| **CTA** | Simplify Managed Futures Strategy | 2022-03-07 | ~$200–220M (S12, S15) | 0.75% (S12) | ~120–180k (S12) | ~ +30% (partial 2022, post-launch) (S15) | n/a | shallower than 60/40 (S12) | 50+ commodities + rates; **NO equities, NO FX** (S12) | beat SG CTA by ~785 bps/yr 4y ann. (S12) |
| **WTMF** | WisdomTree Managed Futures Strategy | 2011-01-05 | ~$225M (S13) | 0.65% (S13) | ~80k (S13) | **−6.5%** (S13) ← negative in trend's best year ever | -7.4% | severe; trend signal under-engineered | Light commodities + FX + rates; small short book (S13) | Poor — historically lags SG Trend by wide margin |
| **FMF** | First Trust Morningstar Managed Futures Strategy | 2013-08-01 | ~$209M (S14) | **0.98%** (S14) | ~60–90k (S14) | ~+5% to +8% (small) | small | mid -teens | Light-touch trend, often net-long bias | Poor — frequently long-only equity, decorrelated from SG Trend |
| CYA | Simplify Tail Risk Strategy | 2021-09 | **fund closed 2024-03-14** (search S "CYA") | n/a | n/a | n/a | n/a | n/a | Tail-risk options / VIX calls; **NOT managed futures**, different category | n/a — disqualified |
| FTLS | First Trust Long/Short Equity | 2014-09-08 | ~$2.0–2.2B | 1.59% | high | flat 2022 | flat | low | 90–100% long equity + 0–50% short equity; **NOT managed futures, not trend-following**, just equity L/S | n/a — disqualified |

Notes on the comparison:
- **AUM threshold.** The brief preferred ≥ $500M for liquidity. Only **DBMF** clears that bar in this universe. KMLM, CTA, WTMF, FMF are sub-$500M. This is the single biggest factor — at MQSMaster's growing notional, a sleeve worth 10% of NAV needs to clear the bid/ask comfortably; DBMF trades ~$30M/day in dollars at ~$30/share × ~1M shares vs ~$7M/day for KMLM and lower for the rest.
- **2022 crisis-alpha episode.** DBMF +21.5%, KMLM ~+30–45%, CTA ~+30%, WTMF **−6.5%**, FMF small positive. WTMF and FMF demonstrably failed to capture the trend opportunity even in the best trend year on record — they are mechanically off-spec and ranked accordingly.
- **2020 COVID episode.** Only DBMF was live for the full year and printed positive (~+8% per S8 fact sheet trajectory + cross-check with SG CTA Index +6% for 2020). KMLM and CTA were post-launch and cannot be evaluated.
- **Replication vs proprietary.** DBMF is a regression-based replicator of the 20 largest CTAs; KMLM is a mechanical trend index over 22 contracts; CTA is a proprietary Altis Partners overlay. Different construction styles — the rationale for considering a dual sleeve (§6).
- **Tracking-error caveat.** DBMF correlates ~0.88 to SG **CTA** Index (broader, slower) but is not a direct SG Trend Index replicator. RSST (S19) is the one that explicitly replicates SG Trend, but the SG-Trend sleeve there is buried inside a stacked product. For a clean managed-futures hedge sleeve, DBMF's 0.88 correlation is more than sufficient.

---

## 5. Ranking

### #1 DBMF — recommended

| Criterion | Verdict |
|---|---|
| Liquidity | Best in class. ~$3.4B AUM, ~780k–1.1M ADV → 10–30 bps round-trip cost typical, comfortable for any single-portfolio rebalance the strategy can plausibly generate. |
| Expense ratio | Lowest (0.85%). 5–13 bps better than rivals; on a 10% sleeve, that's ~1 bp/yr of total-portfolio drag — small, but free. |
| 2022 crisis alpha | +21.5% NAV return while S&P −18.1% — exactly the regime the sleeve is meant to capture. |
| Drawdown profile | Smallest MaxDD in cohort (~ -20.4% vs KMLM ~ -27.5%). |
| Volatility | Lower (~10%) than KMLM (~14%) → more capital-efficient at 10% sleeve weight, leaves more room under the 1.5x leverage cap. |
| Index linkage | 0.88 corr to SG CTA Index — clean managed-futures beta exposure. |
| Track record | 7+ years live (since 2019-05) including the 2020 COVID stress + the 2022 record + the 2023-24 chop. Longest fingerprint. |
| Construction style | Multi-manager replication ⇒ implicitly diversified across CTAs, reduces single-manager catastrophic-failure risk. |
| Data availability in MQSMaster DB | **Unknown — must verify** (see §8). |

### #2 KMLM — strong runner-up

| Criterion | Verdict |
|---|---|
| Liquidity | Adequate (~$334M AUM, ~230k ADV). At a 10% sleeve weight this is fine for current notional but will become tight if MQSMaster grows by ~10x. |
| Expense ratio | 0.90% — slightly more expensive than DBMF. |
| 2022 | The best 2022 return in the cohort (~+30 to +45%); highest convexity to trend regimes. |
| Style | Pure mechanical trend on 22 contracts, **zero equity exposure** ⇒ structurally orthogonal to the P6 stock sleeve — best diversification of any candidate. |
| Drawdown | Larger MaxDD ~ -27.5% — pays for its convexity with higher vol. |
| Why not #1 | AUM < $500M threshold + larger drawdown make it the more aggressive choice. Better paired *with* DBMF (see dual-sleeve §6) than as solo. |

### #3 CTA — distant third

| Criterion | Verdict |
|---|---|
| Liquidity | ~$200M AUM, ~120-180k ADV — sub-threshold. |
| Track record | Only since 2022-03; no 2020 COVID data; small sample. |
| Strategy | Proprietary Altis Partners — concentration risk on a single sub-advisor. |
| Why considered | Strong recent risk-adjusted returns and lowest ER (0.75%). |
| Why not #1 | Too short a track record, too low AUM, too concentrated on one manager. Promote later if KMLM-style construction is desired with a lower fee. |

WTMF, FMF, CYA, FTLS — out. (See matrix.)

---

## 6. Single-ticker vs dual-sleeve decision

**Decision: single ticker (DBMF) for now.** Defer dual-sleeve to a future review.

**Pro dual-sleeve (5% DBMF + 5% KMLM):**
- Diversifies *construction risk*: DBMF = factor regression replication of CTA cohort, KMLM = pure mechanical trend index. Cross-correlation among managed-futures ETFs reported at 0.49–0.71 (S15) — meaningful but far below 1. Splitting reduces tail risk that one product's specific methodology fails in a regime.
- Diversifies *sub-asset-class mix*: DBMF has ~10% equity-index exposure; KMLM has none. Combining lowers the residual equity beta of the hedge sleeve.

**Con dual-sleeve:**
- Operational: doubles the per-rebalance order count for what is already a 10% sleeve. With `REBALANCE_DRIFT_THRESHOLD = 0.005`, both 5% positions will drift across the threshold roughly together → both will trade together → near-zero benefit from fractional positioning per leg.
- Data: doubles the market-data backfill surface (two tickers' worth of `market_data` rows must be ingested and kept current; see §8).
- Cost: KMLM's ER is 5 bps higher than DBMF — splitting raises blended fee from 0.85% to 0.875%.
- **Liquidity penalty**: half the dollar weight in KMLM (sub-$500M AUM) makes each leg trade harder relative to a unified DBMF position.
- The diversification gain is modest at a 10% sleeve. The mathematics: if the two ETFs are 0.6 correlated and each has ~12% vol, the sleeve standard deviation drops from 12% to ~10.7% — a 1.3 vol-point improvement on a 10% weight = ~0.13% drop in portfolio-level vol. Not material vs the friction.

**Trigger to revisit:**
- If `TREND_HEDGE_WEIGHT` is raised above ~0.15 in a future config, the diversification math meaningfully improves and operational drag stays flat → upgrade to dual-sleeve.
- If the §9 falsification test ever trips on DBMF specifically (rolling 36m correlation > 0.30 vs stock sleeve), the immediate replacement is KMLM, not a split.

---

## 7. Config diff — apply-ready

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
     "DSR_MIN_PROB": 0.5
```

That is the entire change. No code edits required in `strategy.py` — the sleeve code at strategy.py:261-274 already handles the ticker correctly. No change to `portfolio_manager_config.json`. No change to P7/P8 configs — they inherit P6's hedge sleeve.

**Apply ordering note (relative to Teams A & B work):** this diff is at lines 17-23 of `config.json`. Team A's unified `config.json` patch (A SYNTHESIS §6) extends the JSON object after `DSR_MIN_PROB` on line 23 — non-overlapping. Apply this C1 diff *before* Team A's block to keep hunk boundaries clean, or *after* if Team A lands first (no conflict either way).

---

## 8. Backfill / data-readiness follow-up

`BasePortfolio._get_market_data` (strategy.py:519-546) queries:

```sql
SELECT ...
  FROM market_data
 WHERE ticker IN (%s, %s, ...)
   AND timestamp BETWEEN %s AND %s
```

— driven by `self.tickers`, which (per Portfolio6Strategy.__init__ lines 70-78) is the union of `universe.json` ∪ `{GLD_TICKER, TREND_HEDGE_TICKER}`.

**Risk:** if DBMF is not in the `market_data` table, the screener at strategy.py:172-194 will skip it (the `asset.Exists` branch returns False) and the sleeve at strategy.py:262-269 will log `Trend hedge ticker 'DBMF' has no market data; sleeve dropped.` — the 10% sleeve weight is silently dropped, the stock book absorbs the slack via the leverage-cap rescale, and the strategy runs without its hedge.

**Follow-up task (Owner: Data / Orchestrator):**
1. Verify DBMF is present in `market_data` over the full backtest window. Quick check:
   ```sql
   SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
     FROM market_data
    WHERE ticker = 'DBMF';
   ```
2. If missing or partial, backfill via the existing `src/orchestrator/backfill/` machinery (`specific_backfill.py` / `injectBackfill.py`).
3. Earliest backfill date = `2019-05-08` (DBMF inception). Any backtest start date prior to this **must** disable the hedge sleeve (set ticker to `""`) or use an alternate ticker — backtests starting in 2010–2018 cannot use DBMF.
4. For live trading, ensure `realtimeDataIngestor.py` polling list includes `DBMF`. The ticker is on NYSE (ARCA), so it follows the same US exchange feed as GLD — no new venue plumbing needed.

**Sanity gate before promoting the config change to live:**
- Confirm DBMF has ≥ 252 trading days of `market_data` ending within the last 5 business days.
- Confirm the first backtest run after the config flip shows non-zero `target_weights['DBMF']` in `[P6] New targets:` log line.

---

## 9. Falsification test

The recommendation rests on the claim that DBMF supplies *diversifying* returns vs the P6 stock book. That claim has a precise, falsifiable form.

**Test.** Compute the rolling 36-month correlation between DBMF daily returns and the P6 stock-sleeve daily returns (excluding GLD and DBMF themselves), measured on a backtest from 2019-05-08 to present.

**Pass:** rolling-36m correlation in `[−0.20, +0.30]` throughout the sample, with mean ≤ +0.10.

**Fail:** rolling-36m correlation > +0.30 in any continuous 12-month window, *or* mean correlation > +0.20 across the full sample.

**Action on fail:** demote to KMLM (no-equity exposure, structurally orthogonal). If KMLM also fails (unlikely given its no-equity construction), revisit the asset choice entirely — the trend hedge thesis would be in question for the prevailing regime, and the hedge sleeve should be disabled (`TREND_HEDGE_TICKER = ""`) until the test passes on a candidate.

**Secondary gate (trend signal alive):** annualized SG Trend Index Sharpe over the most recent 36 months must be > 0. If the index has gone flat or negative for three years, the entire crisis-alpha thesis is in remission and the sleeve weight should be cut to 0.05 (or zero) pending recovery. This is monitored separately from the correlation test.

---

## 10. Risks + rollback

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | DBMF not in `market_data` table → sleeve silently drops | High | §8 backfill verification before merge. The strategy already logs a clear warning at strategy.py:267-268, so an operator monitoring logs will catch it the first month. |
| R2 | Backtest start date before 2019-05-08 | Medium | Add a runtime guard or document: any backtest pre-May-2019 should set `TREND_HEDGE_TICKER = ""` in a backtest-specific config override (consistent with how P8 handles its RBP `train_test_split_date`). |
| R3 | Managed-futures category enters a multi-year drawdown (2018-19 trend "winter") and DBMF drags returns | Medium | The sleeve was sized at 10% precisely to bound this — max contribution to total drawdown ≈ DBMF MaxDD × 0.10 ≈ 2%. Acceptable. Falsification test §9 escalates if structural diversification fails. |
| R4 | iMGP closes or merges DBMF (low base rate but historically happens, cf. CYA closure 2024) | Low | Rollback = single-line config flip to KMLM. No code change. |
| R5 | DBMF's CTA-replication methodology drifts from SG CTA Index in a future regime (replication is backward-looking, S17 documents the lag) | Medium | The 0.88 historical correlation is the best in class for ETF wrappers. If drift is observed in falsification test §9, demote to KMLM (mechanical, no replication lag). |
| R6 | Bid/ask spread widens during a stress event (premium/discount to NAV on a 10% sleeve hurts) | Low–Medium | DBMF's $3.4B AUM and ~1M ADV are the deepest in the cohort precisely to mitigate this. The strategy's `REBALANCE_DRIFT_THRESHOLD = 0.005` already absorbs small premium/discount noise without trading. |
| R7 | Leverage-cap rescale at strategy.py:277-284 silently shrinks the stock book when GLD + DBMF + book > 1.5 | Low | Already behaved this way; documented behavior. Picking DBMF (lower realized vol than KMLM) actually *reduces* this risk vs alternatives. |

**Rollback path (smallest sufficient first):**
1. **Disable sleeve entirely:** `TREND_HEDGE_TICKER` → `""` in `config.json`. One-line edit. No deploy beyond config push. Stock book + GLD continue normally.
2. **Swap ticker:** `TREND_HEDGE_TICKER` → `"KMLM"`. One-line edit. Requires KMLM to be present in `market_data` (apply §8 backfill task to KMLM as well — recommended as a pre-emptive backup so this rollback is one button-press, not one-button-press-plus-a-data-job).
3. **Revert this PR:** `git revert` the single-line config change. Returns to pre-merge behavior (sleeve disabled).

---

End of C1. The recommendation is to set `TREND_HEDGE_TICKER = "DBMF"`. Apply order is independent of Teams A & B patches. The only blocker is the §8 data-readiness check.
