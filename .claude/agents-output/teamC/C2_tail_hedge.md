# C2 — Tail Hedge Sleeve Decision (P6 / P7 / P8)

**Date:** 2026-05-20
**Author:** Team C, agent C2
**Mandate:** Decide whether P6/P7/P8 should add an OTM-put or VIX-call overlay as a third hedge layer on top of GLD + trend-following, or skip with empirical justification.
**Scope:** Read-only repo audit. Single allowed write path: this file.

---

## 1. Executive summary

**Decision: SKIP.** Add no OTM-put or VIX-call overlay to P6/P7/P8 at this time.

Why: the empirical drag of every implementable systematic tail hedge (PPUT, TAIL ETF, VXTH) is large (-2% to -8% annualized in the post-1986 / post-2007 samples) and the crisis payoff is largely **redundant with the trend-following sleeve** already budgeted at 10% of leverage. AQR (Israelov 2015, 2017; Hurst-Ooi-Pedersen 2017), CBOE PPUT data, and live TAIL/VXTH returns all converge: rolling-put strategies systematically lose to "static de-risking" except in fast vol shocks, and trend-following has historically out-earned them across the largest 60/40 drawdowns. The honest Spitznagel/Universa counter-case requires deep-OTM, opportunistically sized puts that the MQSMaster equity executor cannot trade (no options pricing in scope per the mandate). Wrapping the exposure through TAIL is the only ETF-compatible path, and TAIL's 5-yr realized return (-8.0% annualized as of 2026-04-30) makes the cost prohibitive at any sleeve weight that would move portfolio max-DD by ≥ 2pp. We document a falsification test (§7) that would reverse the decision.

---

## 2. Sources fetched (≥10 primary)

| # | URL | Annotation | Relevance |
|---|---|---|---|
| 1 | https://www.aqr.com/Insights/Research/White-Papers/Tail-Risk-Hedging-Contrasting-Put-and-Trend-Strategies | AQR (Israelov, Klein) "Tail Risk Hedging: Contrasting Put and Trend Strategies" — landing page (PDF binary unreadable via fetch; abstract + search-result excerpts used). Argues puts are a more expensive but more reliable fast-crash hedge; trend is cheaper and better at slow drawdowns. | Core framing for §4–5 comparison |
| 2 | https://www.aqr.com/-/media/AQR/Documents/Journal-Articles/JPM-Still-Not-Cheap.pdf | Israelov & Nielsen 2015 "Still Not Cheap: Portfolio Protection in Calm Markets" (JPM, Summer 2015) — global cross-sectional study, March 1996 – June 2014. Finding: at average IV, a 1987-magnitude crash needs to occur once every 10 years for hedged-put strategy to break even; consistently positive volatility risk premium. | Cost-of-protection literature, §4 |
| 3 | https://images.aqr.com/-/media/AQR/Documents/Insights/Alternative-Thinking/Alternative-Thinking-Tail-Hedging-Strategies.pdf | AQR Alternative Thinking — tail-hedging strategies note. | Cross-validates trend-vs-put framing |
| 4 | https://www.cboe.com/insights/posts/benchmark-indices-series-hedging-downside-exposure-with-pput-cll-and-cllz-indices/ | Cboe PPUT/CLL/CLLZ benchmark note. Over ~35-year history, PPUT had 18 monthly declines ≥ 6% vs 35 for SPX. PPUT outperformed SPX by ≥ 10pp in 2008 and 2020. | Direct empirical for rolling-OTM-puts §4 |
| 5 | https://www.cambriafunds.com/tail | TAIL ETF (Cambria) — strategy page. Expense ratio 0.59%, OTM SPX puts + UST collateral. | TAIL ETF realized cost §4 |
| 6 | https://www.aaii.com/etf/ticker/TAIL | AAII metrics, as of 2026-04-30: 1-yr -11.2%, 3-yr -6.2% pa, 5-yr -8.0% pa, beta -0.31. | Live TAIL drag, §4 |
| 7 | https://www.cboe.com/insights/posts/cboe-vxth-index-had-5-year-gain-of-98-6-and-did-well-in-a-60-40-allocation/ | Cboe VXTH (VIX-call tail hedge) — 5-yr +98.6% (≈ 14.7% pa) through Oct 2023; in 2020Q1, VXTH +54.9% vs SPX-TR -19.6%. 60/40 SPX/VXTH = 9.5% pa, Sortino 0.91. | VIX-call ladder reference, §4 |
| 8 | https://www.universa.net/Universa_Spitznagel_SafeHaven_HedgeFunds.pdf | Universa "Why Do People Still Invest in Hedge Funds?" (Jan 2020) — life-to-date avg annual return on invested capital 105.2% (Jan 2008 – Dec 2019) on the put sleeve; 3.33% allocation recommendation. | Spitznagel counter-case, §4 |
| 9 | https://federicocarrone.com/series/leptokurtic/the-tail-hedge-debate-spitznagel-is-right/ | Independent backtest of Spitznagel-style deep-OTM puts vs SPY 2008–2025. 100%-stocks-plus-puts framing: +5.0% (0.5% budget), +10.0% (1.0%), +20.7% (2.0%) excess return vs SPY; max-DD reduced from -51.9% to -32.0% in the 2% case. | Strongest pro-put empirical for §4 |
| 10 | https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing | Hurst-Ooi-Pedersen 2017 "A Century of Evidence on Trend-Following Investing." TSMOM positive in 8 of 10 largest 60/40 drawdowns 1880–2016. | Trend-baseline §3, §5 |
| 11 | https://www.marketsentiment.co/p/tail-risk-hedging | Independent review: TAIL cumulative -44% since 2017, Universa 3,612% on put sleeve in Mar-2020, CAOS bond-like long-run. | Cross-validation §4 |
| 12 | https://bsic.it/tail-wind-for-tail-risk-hedge-funds/ | BSIC review: Universa 3,600% Mar-2020, 3.33%/96.67% allocation, CalPERS exit & forfeited ~$1bn. | Cross-validation §4 |
| 13 | https://www.cfm.com/wp-content/uploads/2022/12/266-2018-The-Convexity-of-trend-following.pdf | CFM "Convexity of Trend Following" — mechanical convexity of trend on the underlying. | Trend convexity context §5 |
| 14 | https://en.wikipedia.org/wiki/Variance_risk_premium | Variance-risk-premium summary; Carr-Wu (2009) — buyers of variance lose on average. | VRP literature §4 |
| 15 | https://wholesale.banking.societegenerale.com/en/prime-services-indices/ | SG CTA / SG Trend index — 2008 +13.1%, 2022 SG Trend +27% (SG CTA +18%). | Trend crisis empirics §5 |
| 16 | https://www.man.com/insights/trend-following-2022-review | Man Group 2022 trend-following review — confirms 2022 SG Trend +27% while 60/40 down double digits. | Cross-validation §5 |

Cross-validation rule satisfied: TAIL drag (sources 5, 6, 11), Universa numbers (8, 9, 12), VXTH numbers (7), trend 2008/2020/2022 (10, 13, 15, 16), put-protection drag literature (1, 2, 3, 4).

---

## 3. Current hedge composition + budget

File: `src/portfolios/portfolio_6/config.json` (lines 17–21):

```json
"GLD_TICKER": "GLD",
"GLD_WEIGHT": 0.07,
"TREND_HEDGE_TICKER": "",
"TREND_HEDGE_WEIGHT": 0.10,
```

File: `src/portfolios/portfolio_6/strategy.py` (hedge sleeve at lines 250–275):
- GLD sleeve: 7% of leverage, hard-coded ticker default `"GLD"`.
- Trend sleeve: 10% of leverage, ticker is **empty by default** — sleeve is disabled until Team C1 selects a ticker (DBMF / KMLM / managed-futures wrapper).
- Stock sleeve: vol-targeted to 13% annual, capped at 1.5× leverage.
- Total hedge budget today = 7% (GLD) + 10% (trend, when ticker set) = **17% of 1.5× leverage = 11.3% of NAV**.

P7 and P8 inherit this hedge sleeve verbatim through `Portfolio6Strategy` (Team A SYNTHESIS §1, D1/D2). Any tail-hedge sleeve added here automatically propagates to P7/P8.

The fixed leverage cap (`MAX_LEVERAGE=1.5`) is the binding constraint: any new sleeve trades **directly against the stock-sleeve allocation** after the proportional rescale at `strategy.py:276–284`.

---

## 4. Cost / payoff table — empirical

Numbers are taken from the sources in §2; columns labeled "drag" are average annualized returns relative to long-only S&P 500 over the sample.

| Candidate | Sample | Annualized drag (vs SPX) | 2008 (SPX -37%) payoff | 2020 Q1 (SPX -19.6%) payoff | 2022 (SPX -18%) payoff | Implementable in MQSMaster? |
|---|---|---|---|---|---|---|
| Rolling 1-mo 5% OTM SPX puts (PPUT) | 1986–2023 | ≈ –2.9% pa (PPUT 6.64% vs PUT 9.54% reference; PPUT also trails SPX-TR). Source 4 confirms ≥ 18 monthly drawdown reductions of 6%+ vs 35 for SPX over 35 yrs. | PPUT outperformed SPX by ≥ 10pp (Source 4). | PPUT loss capped — outperformed SPX by ≥ 10pp again (Source 4). | Small absolute drag, modest protection — puts on the way down only. | **No.** Options not in scope; we are not introducing pricing. |
| Rolling 3-mo 10-delta SPX puts (Universa-style) | 2008–2025 backtest (Source 9, independent) | At 1% annual put budget: **+10.0% excess vs SPY** in 100%-stocks-plus-puts framing; in no-leverage framing (sell stocks to fund puts) +3.07% at 1% budget; max-DD reduced from -51.9% to -32.0% at 2% budget. Counterexamples: AQR Israelov 2017 (near-ATM ~35-delta) finds systematic underperformance. | Sleeve returned multiples of premium spend. | Universa put sleeve **+3,612%** in Mar-2020 (Sources 8, 11, 12). | Whippy — fast vol-crush after Apr-2020; gives back gains. | **No.** Cannot trade 10-delta SPX puts in MQSMaster equity executor. |
| TAIL ETF (Cambria) | 2017-04 → 2026-04 | NAV cumulative ≈ **-44%** since inception (Source 11). Annualized: 1-yr -11.2%, 3-yr -6.2%, 5-yr **-8.0%** as of 2026-04 (Source 6). Expense 0.59%, beta -0.31. Strategy buys laddered OTM puts (~1% budget/mo). | Pre-inception. | Strong positive in 2020 (1-yr return 18.4% as of fact sheet snapshot, Source 5 derived). | Negative — puts gave back. | **Yes** (ETF wrapper). But realized drag is the worst of the candidates. |
| VXTH (CBOE VIX Tail Hedge — 30-delta VIX calls, dynamic ≤1% budget) | 2007 → 2023 | 5-yr +98.6% (~14.7% pa) through Oct 2023 (Source 7) — outsized because of 2020. Long-history drag is small but non-zero in calm regimes. | Q4 2008 month VXTH +1.48% vs SPX -21.4% (Source 7). | VXTH **+54.9%** vs SPX-TR -19.6% (Source 7). | Modest positive (no fast vol spike beyond Feb-Mar). | **No tradable single-ticker ETF wrapper** that cleanly replicates VXTH; VXX/VIXY are continuous vol-products with even worse roll drag. |
| Universa-style deep-OTM (proxy: 36500 LP, undisclosed) | 2008–2019 | LTD avg ann. **+105.2%** on invested put capital (Source 8) — but that is **on the put sleeve**, not the total portfolio (Source 9). 3.33% allocation. | 20 years of premia recovered in one year (Source 8/12). | **+3,612%** on the put sleeve in Mar-2020 (Sources 8, 11). | Premium spend continues, no big drawdown to capture. | **No.** Closed fund, not investable; the deep-OTM machinery is proprietary. |
| **Baseline — Trend sleeve (already in P6 config)** | 1880–2016 (Hurst-Ooi-Pedersen) + live SG Trend | Long-run TSMOM ≈ **+11% pa** gross, +4–7% pa net (Source 10, 13). SG CTA: 2008 +13.1%, 2022 SG Trend +27% (Sources 15, 16). | SG CTA **+13.1%** vs SPX -37% (Source 13/15). | Mixed: trend stumbled in Mar-2020 fast V-shape, then recovered. | SG Trend **+27%** while SPX -18% (Source 16). | **Yes** — `TREND_HEDGE_TICKER` wiring exists; ticker pick is the C1 deliverable. |

**Key tradeoffs distilled:**
- **Pure cost (no crisis):** rolling 5% OTM puts ≈ −3% pa, TAIL ETF realized −8% pa, VXTH near-zero / mildly negative in calm regimes, trend −2 to +5% pa depending on regime.
- **Crisis payoff per unit of drag:** Universa deep-OTM dominates **only** when (a) the crash is fast (vol explodes before equities mean-revert) **and** (b) you can rebalance the gains back into equity at the bottom. TAIL captures part of that but bleeds heavily between crises. Trend captures (b) without the option premium.
- **Drag concentration risk:** TAIL/VXTH compound the drag inside the wrapper. Trend compound the return inside the wrapper.

---

## 5. Decision + literature-backed reasoning

**Decision: SKIP** — do not add a third hedge sleeve to P6/P7/P8 at this time.

Five empirical reasons:

1. **Redundancy with the trend sleeve.** Hurst-Ooi-Pedersen (2017, Source 10) show TSMOM was positive in 8 of the 10 largest 60/40 drawdowns 1880–2016. SG CTA / SG Trend posted +13.1% in 2008 and +27% in 2022 (Sources 13, 15, 16) — matching or exceeding what a small put sleeve would have generated, **without** the persistent option premium drag. AQR's own conclusion in the Israelov tail-hedge paper (Source 1) is that put and trend are complementary, with trend dominating slow drawdowns. The MQSMaster trend sleeve at 10% of leverage already absorbs the 2008 / 2022 use-case.

2. **The 2020 fast-crash gap is the only place a tail hedge clearly beats trend.** Universa's put sleeve returned 3,612% in Mar-2020 (Sources 8, 11, 12) while trend was whipped by the V-shape. But the only ETF wrapper available — TAIL — captured ≈ +18% in the 1-yr ending crisis (Source 6 derived) at the cost of **−8% pa drag over the surrounding 5 yrs**. Spitznagel's own 3.33% allocation rule assumes a **proprietary, deep-OTM, opportunistically sized** book that the MQSMaster executor cannot replicate without an options pricing layer (explicitly out of scope per the mandate).

3. **Israelov / AQR direct evidence on rolling-OTM puts.** Israelov & Nielsen (2015, Source 2) document that across 10 global equity indices 1996–2014, put-based protection has consistently positive volatility-risk-premium drag; at average IV, a 1987-magnitude crash must recur every 10 years just to break even on the hedge. Carr-Wu (2009, Source 14) cement the variance-risk-premium evidence: variance buyers lose on average. The implication for a long-only US equity book is unambiguous.

4. **TAIL ETF live track record is disqualifying.** 5-yr trailing return −8.0% pa (Source 6), 3-yr −6.2% pa, beta −0.31. Even a small 5% sleeve in TAIL would cost the portfolio ~40 bps/yr in static drag — a quarter to a third of the entire trend-sleeve's expected positive return. The wrapper that is actually trade-able in MQSMaster is also the worst-performing one.

5. **Leverage budget is binding.** P6 caps leverage at 1.5× (`MAX_LEVERAGE`) and rescales proportionally at `strategy.py:277–284`. Any tail sleeve must come out of the stock sleeve. The stock sleeve (vol-targeted to 13% annual) is the portfolio's main return engine; trading 5% of it for TAIL costs ≈ 13%·0.05 = 65 bps of expected return for ≈ 40 bps of drag, **plus** the opportunity cost of redundant crisis coverage already paid for by the trend sleeve.

Counterweight considered and rejected: the Spitznagel/leptokurtic counter-case (Source 9) is **directionally correct for deep-OTM, opportunistically-sized 100%-stock+puts portfolios**, not for a constrained ETF wrapper. We agree it is the best path to crisis-alpha-per-premium in principle; we cannot implement it without an options pricing module, which is out of scope per the hard constraint.

---

## 6. Recommended artifact — doc patch (SKIP path)

Single comment update inside `src/portfolios/portfolio_6/strategy.py`, at the top-of-file docstring (lines 1–17). **NO source-code change** — read-only constraint respected. This is the artifact to apply in a follow-up PR.

```diff
--- a/src/portfolios/portfolio_6/strategy.py
+++ b/src/portfolios/portfolio_6/strategy.py
@@ -10,10 +10,29 @@
   4. Stock-sleeve weights scaled so realized portfolio vol ~= VOL_TARGET_ANNUAL,
      bounded by MAX_LEVERAGE.
   5. Hedge sleeve adds GLD_WEIGHT in GLD_TICKER and TREND_HEDGE_WEIGHT in
      TREND_HEDGE_TICKER (config; empty disables that sleeve).
   6. Deflated Sharpe Ratio is logged as a diagnostic of multiple-testing bias.
   7. OnData runs the screen on the first tick of each calendar month and then
      issues rebalance orders toward the stored target weights.
+
+ Tail-hedge decision (Team C / C2, 2026-05-20):
+   We deliberately do NOT add a third "tail" sleeve (rolling OTM SPX puts, VIX
+   calls, or the TAIL / VXTH ETF wrappers) on top of GLD + trend. Reasons:
+     - TAIL ETF live 5-yr return -8.0% pa (AAII/Cambria, 2026-04-30); even a
+       small 5% sleeve costs ~40 bps/yr static drag.
+     - PPUT / Israelov 2015 ("Still Not Cheap", JPM): rolling protective puts
+       carry a persistent volatility-risk-premium drag.
+     - Crisis coverage that a tail sleeve provides is already largely supplied
+       by the trend sleeve (Hurst-Ooi-Pedersen 2017: TSMOM positive in 8/10
+       largest 60/40 drawdowns; SG Trend +13.1% in 2008, +27% in 2022).
+     - The only path that genuinely improves crisis-alpha per premium spent is
+       deep-OTM, opportunistically sized options (Universa / Spitznagel). That
+       requires an options-pricing layer that is out of scope for the equity
+       executor; we will not approximate it via ETF wrappers because the live
+       wrapper drag is empirically worse.
+   Falsification criterion (would reverse this decision):
+     in a walk-forward 2008-2026 backtest, adding TAIL @ 5% sleeve weight
+     (paid from the stock sleeve) raises portfolio CAGR by >= 0 bps AND
+     reduces realized max-DD by >= 4pp vs the GLD+trend baseline.
+   See /Users/abhinav/Desktop/MQSMaster/.claude/agents-output/teamC/C2_tail_hedge.md
 """
```

(No code change; comment-only. The diff is left to the operator to apply in a separate PR since this agent is read-only on source.)

**Optional placeholder config knobs** (back-compat, default disabled — for if/when the falsification test passes):

```diff
--- a/src/portfolios/portfolio_6/config.json
+++ b/src/portfolios/portfolio_6/config.json
@@ -20,7 +20,10 @@
     "TREND_HEDGE_TICKER": "",
     "TREND_HEDGE_WEIGHT": 0.10,
+    "TAIL_HEDGE_TICKER": "",
+    "TAIL_HEDGE_WEIGHT": 0.0,
     "REBALANCE_DRIFT_THRESHOLD": 0.005,
```

These knobs are inert today (`""` ticker + `0.0` weight) and require zero code change to thread through — the existing GLD/trend pattern in `strategy.py:251–274` is a direct template. They are listed here only so a future PR can flip the decision without touching code; not strictly required for the SKIP path.

---

## 7. Falsification test (concrete numeric)

**Falsifier — would reverse the SKIP decision:**

Run a walk-forward, purged-k-fold OOS backtest of P6 (and P7, P8 inheriting) covering **2008-01-01 → 2026-04-30** with two configurations:

- **Baseline:** current config (GLD 7%, trend 10%, ticker selected per C1, stock sleeve fills the rest).
- **Candidate:** Baseline minus 5% stock-sleeve weight, plus `TAIL_HEDGE_TICKER="TAIL"` at `TAIL_HEDGE_WEIGHT=0.05`. (Equivalent constructions: VXTH replicator if available; do **not** test raw VXX/VIXY because of structurally worse roll drag.)

Decision flips to ADD iff **both** hold on the OOS aggregate (means across folds, with mid-period rebalancing):

1. ΔCAGR ≥ 0 bps (i.e., adding TAIL must not reduce return at all — a strict zero-tolerance on drag because the trend sleeve already prices the crisis insurance).
2. Δmax-DD ≤ −4pp (tail sleeve must shave at least 4 percentage points off the worst peak-to-trough drawdown).

Why these thresholds:
- The −8% pa published TAIL drag (Source 6) implies a 5% sleeve drag of ≈ 40 bps/yr — so a positive ΔCAGR on backtest requires the crisis bursts to more than offset 40 bps/yr in our specific universe and rebalance cadence. If they do, the case for ADD is empirical, not theoretical.
- 4pp DD reduction is the smallest improvement that materially changes the Sharpe-to-Calmar ratio at the portfolio level given P6's 13% vol target and 1.5× leverage cap.

If only #2 holds (DD improves but CAGR drops by < 50 bps), promote to a **discussion-required** state — not auto-add. If only #1 holds, do not promote.

---

## 8. Risks + rollback path

Even with the SKIP decision, some residual risks remain:

| # | Risk | Mitigation |
|---|---|---|
| R1 | **Slow-trending drawdown is fully on the trend sleeve** — if trend whipsaws (e.g., Feb-2018, Mar-2020 V-shape), the portfolio has only GLD as the residual hedge. | Acceptable: A1/SYNTHESIS keeps DSR diagnostics live; if trend underperforms over 3+ years, demote the trend sleeve weight in `portfolio_manager_config.json` and re-run the falsification test on TAIL. |
| R2 | **Trend ticker is empty** (`TREND_HEDGE_TICKER=""`). Until C1 picks a ticker, the only crisis hedge is GLD (7%), which **failed in 2020** (GLD held in but didn't rally during the fast-crash week). | Treat this as a C1 blocker, not a C2 problem. C1's deliverable resolves it. The 17% nominal hedge budget remains earmarked. |
| R3 | **Regime change** — if implied vol structurally falls (e.g., a structurally lower-vol equity regime), the calculus could shift: cheaper puts could become attractive at exactly the moment they're needed. | The falsification test in §7 is run quarterly going forward. If TAIL's trailing-3-yr drag improves to ≤ −2% pa, re-run the test. |
| R4 | **Option-pricing module remains absent** — Universa-style deep-OTM construction is permanently inaccessible while that is true. | Out of scope for this agent. Flag for backlog. |

**Rollback path if a future PR flips the decision (ADD):**
- Reverse step 1: set `TAIL_HEDGE_WEIGHT=0.0` in `src/portfolios/portfolio_6/config.json`. The placeholder ticker `""` short-circuits at `strategy.py:261–274` (existing trend pattern), so a single one-line config edit undoes the sleeve with zero code change.
- Reverse step 2 (if also registering at master): set capital weight back to 0 in `portfolio_manager_config.json`.

Both rollbacks are reversible in < 1 minute with no schema migration, no DB writes, no historical-data re-pull.

---

## 9. Coordination notes

- **C1 dependency:** the trend-sleeve ticker pick by C1 is the **prerequisite** for this decision to hold; if C1 also SKIPs (no trend ticker selected), then the hedge budget collapses to GLD 7% only, and §5 reason #1 weakens. In that case, the falsification test in §7 should be **rerun immediately** before P6 promotion. Treat the two decisions as joint.
- **Team A SYNTHESIS:** the new score-method / RBP / sentiment overlays in P7/P8 do not interact with this hedge decision (the hedge sleeve runs post-screening at `strategy.py:251–274`, after vol-target rescale).
- **No source code modified** by this agent — every recommendation above is captured as a patch description for the operator to apply through the normal PR process.

---

End of C2 deliverable.
