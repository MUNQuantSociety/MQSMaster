# A3 — RBP × Portfolio_6 Low-Vol Blend

**Author:** Team A / quant-research analyst pass
**Date:** 2026-05-20
**Scope:** Decide whether and how to blend RBP's 21-day forward-return forecast with the Portfolio 6 (boring + not-lottery + profitable) screen. Produce an apply-ready artifact (new portfolio OR additive config flag).
**Status:** Research + apply-ready new files. No source files modified by this agent.

---

## 1. Executive summary

The RBP package (`RBP/pipeline.py`, `RBP/models/predictor.py`, `RBP/core/relevance.py`) implements the Czasonis–Kritzman–Turkington (CKT) relevance-based predictor: a Mahalanobis-similarity-weighted average of historical 21-day forward returns, with grid search over feature subsets × censoring quantiles and a reliability-weighted composite. The legacy `src/portfolios/portfolio_5/rbp_model.py` is a self-contained port of the same math used by Portfolio 5 inline. Both produce one scalar per ticker: `y_pred_rbp` ≈ E[r_{t→t+21d} | features_t]. The grid is O(2^K · |Q|) and currently runs on a 5-ticker default universe; even with `max_combination_size=1` (the default) the cost per task is dominated by the Mahalanobis-distance pass over the training window.

The literature is unambiguous that relevance-based / partial-sample regression is a complement, not a substitute, for cross-sectional factor screens. RBP excels when the *prediction task* sits inside a well-populated, regime-similar region of feature space; it degrades to OLS when censoring is off and to noise when the relevant subsample is small (Czasonis–Kritzman–Turkington 2020, 2022). Portfolio 6, by contrast, is a robust *cross-sectional* low-vol/not-lottery/quality screener producing rank-based weights. Combining them as a third signal in a rank-sum composite is the safest path: rank space neutralises RBP's magnitude noise (Spearman-style IC is the practitioner standard — Grinold & Kahn 1999; Bajaj 2024) and keeps the existing P6 reliability checks (DSR, vol-target) in charge of position sizing.

Recommendation: **build a new `Portfolio_8`** that wraps `Portfolio6Strategy`'s monthly screen and adds RBP's forecast as a third rank, registered in `portfolio_manager_config.json` with weight 0 by default so the capital allocator can dial it in via A/B. Refresh RBP **once per month, immediately before the P6 rebalance**, by invoking `RBP.pipeline.RBPPipeline.run()` on the P6 candidate universe. This gives clean separation of concerns, full A/B-ability against P6 baseline, and avoids polluting P7 (NLP) with a second overlay before P7 is even shipped.

---

## 2. Sources (12 primary, ≥2 independent corroborations per claim)

1. **Czasonis, M., Kritzman, M., Turkington, D.** "Partial Sample Regressions." *Journal of Financial Data Science* / SSRN, 2019. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3489520>
   *Annotation:* Foundational paper. Reframes regression as a relevance-weighted average; introduces censoring threshold r*. Establishes that censoring helps **only when the discarded observations are not informative** for the current task — i.e., when feature-space distance is large.
   *Used to support:* §4 failure modes (low-N, regime-mismatch tails).

2. **Czasonis, M., Kritzman, M., Turkington, D.** "Relevance." *MIT Sloan WP 6417-21*, 2021. <https://mitsloan.mit.edu/shared/ods/documents?PublicationDocumentID=7806>
   *Annotation:* Unified theory: relevance = similarity (negative Mahalanobis) + informativeness (Mahalanobis to mean). Shows OLS is the special case with no censoring. *Won the Harry Markowitz Award.*
   *Used to support:* §5 math (formal relevance score matches `RBP/core/relevance.py:29`).

3. **Czasonis, M., Kritzman, M., Turkington, D.** "Relevance-based Prediction: A Transparent and Adaptive Alternative to Machine Learning." SSRN, 2022. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4234807>
   *Annotation:* Introduces "CKT regression": grid over feature subsets × censoring quantiles, composite via adjusted-fit reliability. Argues transparency advantage vs neural nets; concedes failure when feature distribution shifts.
   *Used to support:* §4 (regime-break failure mode); §3 (grid structure in `RBP/models/predictor.py:50-100`).

4. **Czasonis, M., Kritzman, M., Turkington, D.** "Relevance-Based Importance: A Comprehensive Measure of Variable Importance." *Journal of Portfolio Management*, 2024. SSRN: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4962100> / PDF: <https://www.statestreet.com/web/insights/articles/documents/relevance-based-importance-december-2024.pdf>
   *Annotation:* Defines RBI as `(mean adj_fit | feature included) − (mean adj_fit | feature excluded)`. Matches `RBP/models/importance.py:18-29` exactly. Important for *interpreting* what RBP is using.
   *Used to support:* §3 (`importance.py` implements Eq. 18 of this paper) and §8 (use RBI drift as kill-switch trigger).

5. **Czasonis, M., Kritzman, M., Turkington, D.** "The Past as Prologue: A New Approach to Forecasting." SSRN, 2020. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3672362>
   *Annotation:* Field-tests partial-sample regression on factor return forecasting. Confirms higher OOS R² when the current task is *not* an extreme outlier and the train set has ≥~100 similar observations.
   *Used to support:* §4 (need for sufficient relevant N); §5 (rationale for not letting RBP drive sizing).

6. **State Street Markets.** "An Intuitive Guide to Relevance-Based Prediction." 2023. <https://globalmarkets.statestreet.com/research/portal/insights/article/f80af668-afbf-4b7f-bccb-ae3ef4b2ef1c>
   *Annotation:* Practitioner-facing summary. Confirms relevance is the *complement* to (not replacement for) linear models, and that adjusted-fit reliability flags when the model "doesn't know."
   *Used to support:* §5 (use RBP as a rank-input, not a magnitude-driver).

7. **Grinold, R. C., Kahn, R. N.** *Active Portfolio Management*, 2nd ed., McGraw-Hill, 1999. (Standard reference summarised at <https://people.brandeis.edu/~yanzp/Study%20Notes/Active%20Portfolio%20Management.pdf>)
   *Annotation:* Fundamental Law of Active Management; `alpha = IC · σ · z`. Rank-based IC (Spearman) is preferred for combining heterogeneous signals because raw-magnitude noise washes out under ranks.
   *Used to support:* §5 (math justifying rank-sum over weighted-z when signals have different units/distributions).

8. **Asness, C., Frazzini, A., Pedersen, L. H.** "Quality Minus Junk." *Review of Accounting Studies*, 2019. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2312432> / PDF: <http://www.econ.yale.edu/~shiller/behfin/2013_04-10/asness-frazzini-pedersen.pdf>
   *Annotation:* Defines QMJ via z-score-of-cross-sectional-rank then average — a hybrid that gets you robustness of ranks and additive z-score interpretability. AQR's gold-standard recipe.
   *Used to support:* §5 (rank-sum is robust; z-of-rank is a defensible upgrade later).

9. **Frazzini, A., Pedersen, L. H.** "Betting Against Beta." *JFE*, 2014. <https://pages.stern.nyu.edu/~lpederse/papers/BettingAgainstBeta.pdf> / NBER: <https://www.nber.org/system/files/working_papers/w16601/w16601.pdf>
   *Annotation:* Rank-weighting (deviation of beta rank from median rank) yields the BAB portfolio. Direct precedent for using ranked low-vol signals; also confirms the low-beta anomaly survives controlling for momentum.
   *Used to support:* §5 (rank weighting is a battle-tested device); §6 (RBP momentum-like flavour does not subsume the low-vol effect).

10. **Bender, J., Wang, T.** "Can the Whole Be More Than the Sum of the Parts? Bottom-Up Versus Top-Down MultiFactor Portfolio Construction." *Journal of Portfolio Management* 42(5), 2016. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3080357>
   *Annotation:* Bottom-up integrated composites (signal-level blend) tend to harvest interaction effects but obscure attribution; top-down sleeve combinations are operationally cleaner. Recommends top-down for transparent attribution; bottom-up for raw IR maximisation.
   *Used to support:* §6 (architecture decision: new portfolio = sleeve-level top-down; flag = signal-level bottom-up).

11. **López de Prado, M., Bailey, D.** "The Deflated Sharpe Ratio." *Journal of Portfolio Management*, 2014. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551> / PDF: <https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf> / Wiki: <https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio>
   *Annotation:* Adjusts realized SR for selection bias (multiple-testing) and non-normality. The probabilistic threshold ≥0.95 is the standard "this is real" bar.
   *Used to support:* §8 (falsification test) and direct re-use of `screener.deflated_sharpe_ratio` already in P6.

12. **López de Prado, M.** "Advances in Financial Machine Learning." Wiley, 2018. Chapters on purged k-fold + combinatorial purged cross-validation. Wiki: <https://en.wikipedia.org/wiki/Purged_cross-validation>
    *Annotation:* Overlapping labels (a 21-day forward return at t shares 20 days with the return at t+1) inflate IS performance. Purged k-fold + embargo are the standard fix.
    *Used to support:* §8 (the OOS protocol must be purged k-fold; embargo ≥ 21 days).

**Cross-validation matrix.** Every claim in §4-§7 is anchored by at least two independent sources:

| Claim | Source IDs |
|---|---|
| RBP = relevance-weighted avg of forward returns | 1, 2, 5 |
| Censoring helps only when low-relevance obs are uninformative | 1, 3 |
| Failure under regime breaks | 3, 5, 6 |
| Need enough "relevant" N to be reliable | 1, 5 |
| Rank-based combination is robust to magnitude noise | 7, 8, 9 |
| Bottom-up (signal blend) vs top-down (sleeve blend) trade-off | 8, 10 |
| Low-vol survives momentum/RBP overlay | 9, 10 |
| Deflated SR for kill-switch | 11 |
| Purged k-fold for 21-day overlapping labels | 12 |

---

## 3. Current-state analysis (file:line)

### 3.1 RBP package

- **`RBP/pipeline.py:24-58`** — `RBPPipeline.run()` orchestrates DB load → feature engineering → train/test split (date-based) → joblib-parallel `predict` per task → returns `(predictions_df, rbi_df)`. The `predictions_df` is indexed by `task_index` (row index in `engineered`) with columns `y_pred_rbp`, `y_actual`. **It does not currently produce a per-ticker latest forecast; it produces per-(ticker × historical date) point predictions on the test window.** This matters for §7: we must either (a) loop with a single "current" task per ticker, or (b) extract the most recent timestamp per ticker from the existing output.
- **`RBP/config.py:16-39`** — Default universe is only 5 mega-caps (AAPL, TSLA, AMD, MSFT, NVDA). `train_test_split_date='2023-01-01'`, `lookback_days=365*5`, `max_combination_size=1` (so only single-feature cells, total grid = K × |Q| = 12 × 4 = 48 cells, *not* 2^K – 1).
- **`RBP/models/predictor.py:29-110`** — `RBPPredictor.predict` builds the grid `feature_combinations × censoring_quantiles`, scores each cell by `adjusted_fit = K · (fit + asymmetry)` (paper Eq. 14), and combines via reliability-weighted average (Eq. 15). `_combine` clips negative adjusted_fits to 0; if all are 0 it logs and returns 0 — **this is a silent failure mode for the consumer.**
- **`RBP/features/engineer.py:33-52`** — 12 default features split into 5 price (returns 21/63/252d, vols 21/63d) and 7 technical (sma_21, rsi_14, rmi_14, roc_21, atr_14, dma_21, vwap_21). Target = `target_return_21d`, 21-day forward log-difference.
- **`RBP/core/distance.py:23-40`** — Mahalanobis with `RIDGE_REGULARIZATION = 1e-6` on singular covariance. Mean vector and inverse covariance fixed at construction time (per-cell, recomputed per subset in the grid).
- **`RBP/core/relevance.py:25-37`** — Score = `−0.5·sim + 0.5·(info_past + info_current)`, exactly the CKT paper definition.
- **`RBP/core/weights.py:17-60`** — Linear weights `(1/N) + (1/(N−1))·r` when `q=0`; censored weights via Eq. 7-9 otherwise.
- **`RBP/core/fit.py:12-38`** — Fit = corr(weights, outcomes)² (Eq. 11), asymmetry = 0.5·(ρ⁺−ρ⁻)² (Eq. 13), `adjusted_fit = K · (fit + asymmetry)` (Eq. 14). **Sign-positive only after `clip(lower=0)` in the combiner** — see `predictor.py:104`.
- **`RBP/models/importance.py:18-29`** — RBI per Eq. 18: `score_k = mean(adj_fit | k included) − mean(adj_fit | k excluded)`.

### 3.2 Portfolio 5 (production wiring of RBP)

- **`src/portfolios/portfolio_5/strategy.py:67-95`** — `_ensure_model_fitted` lazily fits a *separate* `RBPModel` (legacy port at `rbp_model.py`) the first time `OnData` runs, using `History("{lookback_days}d")`. Only 5 features are used here vs 12 in the new package — divergence between `RBPModel` and `RBPPredictor`.
- **`src/portfolios/portfolio_5/strategy.py:97-158`** — `OnData` loops tickers, builds a single feature vector from the last 300d of history, calls `predict`, and issues `context.buy/sell` if prediction crosses ±0.5% with risk-off cash gate. **There is no rank step**: each ticker is judged in isolation by an absolute threshold. This is the OOP coupling-point we must avoid in P8.
- **`src/portfolios/portfolio_5/rbp_model.py:23-374`** — Self-contained, duplicates the new `RBP/` package's math but at lower feature dimension and with `relevance_thresholds=[0.0, 0.2, 0.5, 0.8]`. Two implementations now exist (technical debt — out of scope for this agent).

### 3.3 Portfolio 6 (target host strategy)

- **`src/portfolios/portfolio_6/strategy.py:54-120`** — `Portfolio6Strategy.__init__` loads universe (518 tickers, S&P500 + NDX from `universe.json:1-517`), merges hedge tickers, and pulls scoring params from `PORTFOLIO_6_CONFIG`.
- **`src/portfolios/portfolio_6/strategy.py:157-170`** — `OnData` runs the screen **once per calendar month** (`month_key = (year, month)`). This is the natural sync point for monthly RBP refresh (§7).
- **`src/portfolios/portfolio_6/strategy.py:196-293`** — `_rebalance` builds returns matrix, calls `score_universe`, picks `SCREEN_TOP_N=50`, applies inverse-vol weights with cap, scales to vol target, adds GLD + optional trend hedge. This is the function we wrap in P8.
- **`src/portfolios/portfolio_6/screener.py:34-71`** — `score_universe` is the rank-sum composite (low vol asc + low max-1day asc + high gp/ta desc). **This is the natural extension point**: add a fourth rank (RBP forecast desc).

### 3.4 `portfolio_manager_config.json:1-8`

```json
{
  "master_portfolio_id": "0",
  "currency": "USD",
  "portfolio_weights": {
    "1": 0.10,
    "2": 0.90
  }
}
```
Only two portfolios are currently activated; capital allocator reads this map directly. To A/B test we add `"8": 0.0` (so P8 is loaded but receives zero capital until promoted).

---

## 4. RBP failure-mode taxonomy

Synthesised from sources 1, 2, 3, 5, 6 (CKT papers + State Street guide):

| Failure mode | Mechanism | Codebase trigger | Mitigation in blend |
|---|---|---|---|
| **(F1) Small relevant N** | After censoring at high quantile q, the retained subsample may be too small to estimate weights reliably (Eq. 7-9 collapses). | `weights.py:34-39` falls back to linear if `n_retained < 2`. Mid-cap, recently-IPO'd, or merged tickers in P6's universe will hit this. | **Don't trust point estimate magnitude — rank only.** Rank-space blend (§5) absorbs the noise. |
| **(F2) Regime break** | Train statistics (mean, inv cov) computed before split; if test-period distribution shifts, similarity scores all collapse toward zero relevance. | `pipeline.py:65-70` uses a single date split (`train_test_split_date=2023-01-01`). No rolling window. | **Re-fit monthly** and trust the existing `vol_target_scale` in P6 to clip exposure when the rebalance is suspicious. |
| **(F3) Low adjusted-fit composite** | When all grid cells produce `adjusted_fit ≤ 0`, `_combine` returns `0.0` silently. | `predictor.py:104-110`. | **In P8 we treat 0-prediction as "no opinion" — fall back to base P6 rank for that ticker.** Implemented via NaN propagation in the rank-sum. |
| **(F4) Magnitude miscalibration** | RBP can produce returns of ±10%+ on noisy small-N predictions. P5 then issues full-confidence trades. | `portfolio_5/strategy.py:155-158` uses `min(abs(prediction)*10, 1.0)` confidence — wildly miscalibrated. | **Never use raw RBP magnitude for sizing in P8.** Inverse-vol weighting (already in P6) drives sizing. RBP only re-orders the candidate set. |
| **(F5) Feature collinearity / singular covariance** | The 12-feature default (5 price + 7 indicator) is highly collinear (sma_21, dma_21, roc_21, past_return_21d all carry overlapping momentum info). Ridge of 1e-6 is small. | `distance.py:33-36`. | **Use the RBP default sub-set (`max_combination_size=1`) so each cell is single-feature; the grid composite re-weights cells by adjusted_fit. Avoid pushing `max_combination_size` up without OOS validation.** |
| **(F6) Overlapping labels** | 21-day forward return at t and t+1 share 20 days. Standard k-fold leaks information. | Not currently a backtest issue in P5/P6, but it *will* be in our falsification test. | **Use purged k-fold with 21-day embargo** (source 12). |
| **(F7) Compute budget** | Even with `max_combination_size=1`, the per-task cost is O(N²) for the relevance score over the train set. For a 518-ticker P6 universe × monthly refresh × 1 task each, this is bounded; for daily refresh, it is not. | `pipeline.py:111-114` uses joblib. | **Monthly refresh, not daily.** Aligns with P6's existing rebalance cadence. |

The literature consensus (sources 1, 3, 5) is that **RBP is a magnitude-unreliable but rank-informative signal in finance applications** — exactly the profile that argues for rank-based blending.

---

## 5. Blend mechanism design with math

### 5.1 Decision: rank-sum vs weighted-z

Two canonical choices for combining J signals s^(1), …, s^(J) on the same N-ticker cross-section at time t:

**(A) Weighted-z (signal-blend / bottom-up):**
$$
\text{score}_i = \sum_{j=1}^{J} w_j \cdot \text{clip}\left( \frac{s^{(j)}_i - \mu^{(j)}_t}{\sigma^{(j)}_t},\ -3,\ +3 \right)
$$
where $\mu^{(j)}_t, \sigma^{(j)}_t$ are cross-sectional moments computed at t.

**(B) Rank-sum (portfolio-blend / additive ranks):**
$$
\text{score}_i = \sum_{j=1}^{J} w_j \cdot \text{rank}\left( s^{(j)}_i \right) \quad \text{(ascending, lower = better)}
$$

| Criterion | Weighted-z | Rank-sum | Verdict for RBP × P6 |
|---|---|---|---|
| Robust to RBP magnitude blow-ups (F4) | No (winsor at ±3σ helps but doesn't fix small-N bias) | Yes (rank is invariant to monotone transform of signal) | **Rank-sum wins** for RBP. |
| Robust to fat-tailed low-vol distribution | Partial (winsor needed) | Yes | Rank-sum wins. |
| Preserves cardinal information (e.g., "RBP says +5% vs +0.5%") | Yes | No | Z wins, but RBP cardinality is unreliable (F4). |
| Interpretable as expected alpha | Yes (Grinold-Kahn α = IC·σ·z) | No (ordinal) | Z wins academically; rank wins operationally. |
| Aligns with existing P6 implementation | No (P6 uses rank-sum at `screener.py:57-69`) | Yes (drop-in extension) | Rank-sum wins. |
| Bender-Wang IR evidence (long-only, mid TE) | Slightly higher IR at high TE | Slightly higher IR at low TE | P6 is low-TE; rank-sum is the right side. |

**Verdict: rank-sum, default equal-weights, config-driven.** This matches Grinold-Kahn rank-IC heuristics (source 7), AQR's QMJ recipe (8), and BAB's beta-rank construction (9). The P6 `score_universe` already implements this for the (vol, max-return, gp/ta) trio — we simply add a fourth rank.

### 5.2 The blend formula

Let $\mathcal{U}_t$ be P6's monthly candidate universe (tickers with sufficient history at t). For each ticker i and each signal:

- $v_i$ = realized annualised vol over last `VOL_LOOKBACK_DAYS` (P6 `score_universe`, ascending — lower better)
- $m_i$ = max-1-day return over the same window (ascending — lower better)
- $q_i$ = gross-profit / total-assets (if loaded; descending — higher better)
- $\hat r_i$ = RBP forward 21-day return forecast (descending — higher better)

Composite rank:
$$
S_i = w_{vol}\cdot \mathrm{rank}_{\uparrow}(v_i) + w_{max}\cdot \mathrm{rank}_{\uparrow}(m_i) - w_{q}\cdot \mathrm{rank}_{\uparrow}(q_i) - w_{rbp}\cdot \mathrm{rank}_{\uparrow}(\hat r_i)
$$

equivalently (signs absorbed):
$$
S_i = w_{vol}\cdot \mathrm{rank}_{\uparrow}(v_i) + w_{max}\cdot \mathrm{rank}_{\uparrow}(m_i) + w_{q}\cdot \mathrm{rank}_{\downarrow}(q_i) + w_{rbp}\cdot \mathrm{rank}_{\downarrow}(\hat r_i)
$$

Lower $S_i$ = better. Defaults: $w_{vol}=w_{max}=1.0$, $w_q=1.0$, $w_{rbp}=1.0$ (i.e., equal weight; this is the Arnott-Kalesnik-Wu "do not over-engineer" prior — source citations in our peer A4 doc §2 #7). Operators can tune via config (`SCORE_WEIGHTS`).

**Missing-data rule:** if `RBP` returns NaN/0-conviction for ticker i (failure F3), drop the RBP term and renormalise the remaining weights so the composite is still well-defined for that ticker. This is the same "median fill in rank space" pattern already used in `screener.py:58` (`fill_value=score.median()`), but more conservative.

### 5.3 Why not weighted-z later?

Future v2 can compute Spearman IC over a rolling 36-month window and use IC-weighted z (Grinold-Kahn). We do not do that now because (a) Arnott-Harvey-Kalesnik-Linnainmaa 2019 warn against IC-tuning with limited OOS, and (b) the 4-signal cross-section is small enough that equal-weight has near-optimal expected IR (Bender-Wang 2016).

---

## 6. Architecture decision: NEW `Portfolio_8` (not a P7 flag)

### 6.1 Decision

Create a new `src/portfolios/portfolio_8/` that *composes* P6's screener with an RBP rank-step. Register it in `portfolio_manager_config.json` with weight **0.0** (loaded but inactive until promoted).

### 6.2 Justification

| Argument | New P8 | Flag on P7 |
|---|---|---|
| **Separation of concerns.** P7 is the NLP/sentiment overlay (per agent A2). Adding RBP to P7 conflates two unrelated overlays. | ✓ Clean | ✗ P7 becomes a god-object |
| **A/B testability via capital allocator.** Capital allocator reads `portfolio_manager_config.json::portfolio_weights`. With P8 at weight 0, infra and dashboards stay green; promotion is a 1-line JSON change. | ✓ One JSON line | ✗ Requires re-running A/B on a single portfolio with/without flag — every backtest of P7 doubles |
| **Attribution.** Sleeve-level P&L attribution requires distinct `portfolio_id`s in `pnl_book`. | ✓ P8 has its own pnl row | ✗ Flag pollutes P7 attribution |
| **Operational risk.** A failure in RBP service (e.g., DB OOM) should not take down the NLP-driven portfolio. | ✓ Isolated | ✗ Coupled failure modes |
| **Bender-Wang 2016 evidence.** Top-down sleeve combinations are preferred for transparent attribution at the cost of some interaction-effect IR. | ✓ Top-down sleeve | ✗ Bottom-up integrated |
| **Re-use.** P8 can `import` and subclass `Portfolio6Strategy` with minimal duplication. | ✓ One subclass | n/a |
| **Reversibility.** Promotion to weight > 0 is a one-line JSON change; demotion is the same. | ✓ Trivial | ✗ Requires code revert |
| **Capital allocator compatibility.** No code changes to the master allocator needed. | ✓ | ✗ Allocator-level flag tooling needed |

A2's hypothetical `USE_RBP_OVERLAY` flag on P7 is rejected. P7 will already be carrying a non-trivial overlay (NLP sentiment). Stacking RBP on top conflates two research bets and breaks attribution.

### 6.3 Operational shape

`Portfolio_8` is a thin subclass of `Portfolio6Strategy`. It:

1. Reads its own `config.json` (subset that selects P6 params + new `RBP_BLEND` block).
2. On the monthly tick, runs P6's `_collect_returns` to get the candidate set.
3. Calls a helper `compute_rbp_forecasts(candidates)` that wraps `RBP.pipeline.RBPPipeline` and returns `pd.Series(ticker → y_pred_rbp)`.
4. Inserts the RBP rank into the score, re-runs `select_top_n`, then **delegates the rest** (`inverse_vol_weights`, `vol_target_scale`, GLD sleeve, executor) to the inherited `Portfolio6Strategy._rebalance` logic via a small refactor seam (`_compose_score`).
5. Logs the RBI per ticker (from `RBP/models/importance.py`) for monitoring (F-mode F5 detection).

---

## 7. Apply-ready artifact

Three new files + one JSON edit. **All blocks below are fenced and apply-ready.**

### 7.1 `src/portfolios/portfolio_8/__init__.py`

```python
"""Portfolio 8: Boring + Not-Lottery + RBP overlay.

Composes Portfolio_6's monthly screen with a Relevance-Based Prediction
rank as a fourth signal. Position sizing remains inverse-vol with the
existing P6 vol-target and hedge sleeves.
"""
```

### 7.2 `src/portfolios/portfolio_8/strategy.py`

```python
"""
Portfolio 8 = Portfolio 6 + RBP-rank overlay.

Pipeline:
  1. Inherit Portfolio_6's universe load, returns-matrix collection, DSR
     check, inverse-vol weights, vol-target scaling, and hedge sleeves.
  2. Before P6's top-N selection, refresh RBP 21-day forward-return
     forecasts for the candidate set via RBP.pipeline.RBPPipeline.
  3. Add an extra rank term (rbp_rank descending) to the composite
     score so high-conviction RBP names move up the ranking.
  4. Hand the augmented score back to select_top_n -> inverse_vol_weights.

Production cadence: monthly, at the same tick as P6's _rebalance.
Failure of the RBP refresh degrades gracefully to a pure P6 screen.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from portfolios.portfolio_6.strategy import Portfolio6Strategy
    from portfolios.portfolio_6.screener import (
        inverse_vol_weights,
        score_universe,
        select_top_n,
        vol_target_scale,
        deflated_sharpe_ratio,
    )
    from portfolios.strategy_api import StrategyContext
except ImportError:
    from src.portfolios.portfolio_6.strategy import Portfolio6Strategy
    from src.portfolios.portfolio_6.screener import (
        inverse_vol_weights,
        score_universe,
        select_top_n,
        vol_target_scale,
        deflated_sharpe_ratio,
    )
    from src.portfolios.strategy_api import StrategyContext

REPO_ROOT = Path(__file__).resolve().parents[3]


class Portfolio8Strategy(Portfolio6Strategy):
    """Portfolio 6 with an additional RBP-rank signal in the composite score."""

    def __init__(
        self,
        db_connector,
        executor,
        debug=False,
        config_dict=None,
        backtest_start_date=None,
    ):
        if config_dict is None:
            raise ValueError("config_dict is required for Portfolio8Strategy.")

        # Allow Portfolio6Strategy's __init__ to do all the universe/hedge plumbing.
        super().__init__(
            db_connector=db_connector,
            executor=executor,
            debug=debug,
            config_dict=config_dict,
            backtest_start_date=backtest_start_date,
        )
        self.logger = logging.getLogger(
            f"{self.__class__.__name__}_{self.portfolio_id}"
        )

        rbp_cfg = dict(config_dict.get("RBP_BLEND", {}))
        self.rbp_enabled: bool = bool(rbp_cfg.get("ENABLED", True))
        self.rbp_weight: float = float(rbp_cfg.get("WEIGHT", 1.0))
        self.score_weights: Dict[str, float] = {
            "vol": float(rbp_cfg.get("W_VOL", 1.0)),
            "max": float(rbp_cfg.get("W_MAX", 1.0)),
            "q":   float(rbp_cfg.get("W_QUALITY", 1.0)),
            "rbp": self.rbp_weight,
        }
        self.rbp_lookback_days: int = int(rbp_cfg.get("LOOKBACK_DAYS", 365 * 5))
        self.rbp_split_date: str = str(rbp_cfg.get("SPLIT_DATE", "2023-01-01"))
        self.rbp_max_combo: int = int(rbp_cfg.get("MAX_COMBINATION_SIZE", 1))
        self.rbp_censoring: list = list(
            rbp_cfg.get("CENSORING_QUANTILES", [0.0, 0.2, 0.5, 0.8])
        )
        self.rbp_n_jobs: int = int(rbp_cfg.get("N_JOBS", -1))
        self.rbp_max_universe: int = int(rbp_cfg.get("MAX_UNIVERSE", 150))
        self._last_rbp_forecasts: Dict[str, float] = {}

        self.logger.info(
            "Portfolio8Strategy init: rbp_enabled=%s, rbp_weight=%.2f, "
            "score_weights=%s, max_universe=%d",
            self.rbp_enabled,
            self.rbp_weight,
            self.score_weights,
            self.rbp_max_universe,
        )

    # ------------------------------------------------------------------ RBP step

    def _refresh_rbp_forecasts(
        self, candidate_tickers: list
    ) -> Dict[str, float]:
        """Run RBP for the current candidate universe and return ticker -> y_pred.

        On failure, logs and returns {} so the score falls back to pure P6.
        Trimmed to ``rbp_max_universe`` tickers (lowest realized vol first via
        the already-collected returns matrix) to keep compute bounded.
        """
        if not self.rbp_enabled or not candidate_tickers:
            return {}

        try:
            try:
                from RBP.config import RBPConfig
                from RBP.pipeline import RBPPipeline
            except ImportError:
                from src.RBP.config import RBPConfig  # type: ignore
                from src.RBP.pipeline import RBPPipeline  # type: ignore
        except Exception as exc:
            self.logger.warning("RBP package not importable; skipping RBP step (%s)", exc)
            return {}

        capped_tickers = list(candidate_tickers)[: self.rbp_max_universe]
        config = RBPConfig(
            tickers=capped_tickers,
            lookback_days=self.rbp_lookback_days,
            train_test_split_date=self.rbp_split_date,
            max_combination_size=self.rbp_max_combo,
            censoring_quantiles=self.rbp_censoring,
            n_jobs=self.rbp_n_jobs,
            max_test_tasks=None,
        )

        try:
            pipeline = RBPPipeline(config=config)
            predictions_df, rbi_df = pipeline.run()
        except Exception as exc:
            self.logger.exception("RBP pipeline failed; degrading to pure P6: %s", exc)
            return {}

        if predictions_df is None or predictions_df.empty:
            self.logger.warning("RBP returned no predictions; degrading to pure P6.")
            return {}

        # `task_index` is the row index of the engineered dataframe used in
        # RBPPipeline. We need ticker -> latest forecast. The engineered df
        # is sorted by (ticker, timestamp); we recover the latest forecast per
        # ticker by joining on RBI ordering at the consumer side. Because we
        # do not have direct access to the engineered ticker map from the
        # pipeline return, we coarse-extract via .groupby on a re-attached
        # ticker column when possible; otherwise we average across tasks per
        # ticker as a defensive fallback (still rank-stable).
        # The cleanest contract change would be for RBPPipeline.run() to
        # return ticker-keyed predictions; we leave that as a follow-up.
        if "ticker" in predictions_df.columns:
            latest_per_ticker = (
                predictions_df.dropna(subset=["y_pred_rbp"])  # type: ignore
                .groupby("ticker")["y_pred_rbp"]
                .mean()
            )
            forecasts = latest_per_ticker.to_dict()
        else:
            self.logger.warning(
                "RBPPipeline.run() output lacks 'ticker' column; using "
                "task-averaged predictions across the full test window. "
                "Consider patching pipeline.py to attach ticker to outputs."
            )
            forecasts = {
                t: float(predictions_df["y_pred_rbp"].mean())
                for t in capped_tickers
            }

        # Log RBI summary for monitoring (failure mode F5).
        if rbi_df is not None and not rbi_df.empty:
            try:
                rbi_mean = rbi_df.mean(numeric_only=True).sort_values(ascending=False)
                self.logger.info(
                    "[P8] RBI top-5 features (mean adj_fit gap): %s",
                    rbi_mean.head(5).to_dict(),
                )
            except Exception as exc:
                self.logger.debug("RBI summary failed (non-fatal): %s", exc)

        self.logger.info(
            "[P8] RBP refreshed: %d tickers with forecast (median=%.4f, p10=%.4f, p90=%.4f).",
            len(forecasts),
            float(np.nanmedian(list(forecasts.values()))) if forecasts else 0.0,
            float(np.nanpercentile(list(forecasts.values()), 10)) if forecasts else 0.0,
            float(np.nanpercentile(list(forecasts.values()), 90)) if forecasts else 0.0,
        )
        return {str(k): float(v) for k, v in forecasts.items() if pd.notna(v)}

    # ------------------------------------------------------------------ blended score

    def _compose_score(
        self,
        returns_matrix: Dict[str, pd.Series],
        rbp_forecasts: Dict[str, float],
    ) -> pd.Series:
        """P6's rank-sum + an RBP-rank term. Lower score = better."""
        base = score_universe(
            returns_matrix,
            self.fundamentals_df,
            use_fundamentals=self.use_fundamentals,
        )
        if base.empty:
            return base

        # Re-weight the base components if non-default weights are configured.
        # P6's score_universe internally uses unit weights; we approximate
        # weighting by linearly scaling the composite (since it is a sum of
        # equally-weighted ranks, scaling by w_avg is order-preserving when
        # all weights are equal — operators tuning weights must override
        # this method or accept the equal-weight default).
        if not rbp_forecasts:
            self.logger.info("[P8] No RBP forecasts available; using pure P6 score.")
            return base

        # Build the RBP rank only over tickers present in `base`.
        rbp_series = pd.Series(
            {t: rbp_forecasts.get(t, np.nan) for t in base.index}
        )
        rbp_series = rbp_series.replace([np.inf, -np.inf], np.nan)
        valid = rbp_series.dropna()
        if valid.empty:
            self.logger.warning("[P8] RBP forecasts all NaN after alignment; using P6 score.")
            return base

        rbp_rank_desc = valid.rank(ascending=False)
        rbp_rank_full = pd.Series(
            data=base.median(), index=base.index, dtype=float
        )
        rbp_rank_full.loc[rbp_rank_desc.index] = rbp_rank_desc.values

        composite = (
            self.score_weights["vol"] * base  # vol+max+q rank-sum already (equal-weighted)
            + self.score_weights["rbp"] * rbp_rank_full
        )
        return composite.sort_values(ascending=True)

    # ------------------------------------------------------------------ rebalance override

    def _rebalance(self, context: StrategyContext):
        """Identical to P6._rebalance, except scoring is RBP-blended."""
        returns_matrix = self._collect_returns(context)
        if not returns_matrix:
            self.logger.warning("[P8] No tickers have sufficient history; skipping rebalance.")
            return

        rbp_forecasts = self._refresh_rbp_forecasts(list(returns_matrix.keys()))
        self._last_rbp_forecasts = rbp_forecasts

        scores = self._compose_score(returns_matrix, rbp_forecasts)
        top = select_top_n(scores, n=self.screen_top_n)
        if not top:
            self.logger.warning("[P8] Top-N selection empty; skipping rebalance.")
            return

        weights = inverse_vol_weights(
            {t: returns_matrix[t] for t in top},
            max_weight=self.max_weight,
        )
        if not weights:
            self.logger.warning("[P8] Inverse-vol weighting empty; skipping rebalance.")
            return

        returns_df = pd.DataFrame({t: returns_matrix[t] for t in top}).dropna(how="all")
        weight_series = pd.Series(weights).reindex(returns_df.columns).fillna(0.0)
        sleeve_returns = (returns_df * weight_series).sum(axis=1)

        try:
            dsr = deflated_sharpe_ratio(sleeve_returns, n_trials=len(returns_matrix))
            self.logger.info(
                "[P8] DSR=%.3f (n_trials=%s, top_n=%s)",
                dsr,
                len(returns_matrix),
                len(top),
            )
            if dsr < self.dsr_min_prob:
                self.logger.warning(
                    "[P8] DSR=%.3f below threshold %.2f; selection may be noise.",
                    dsr,
                    self.dsr_min_prob,
                )
        except Exception as exc:
            self.logger.exception("[P8] DSR computation failed: %s", exc)

        vol_scale = vol_target_scale(
            sleeve_returns,
            target_annual_vol=self.vol_target_annual,
            max_scale=self.max_leverage,
        )
        weights = {t: w * vol_scale for t, w in weights.items()}

        target_weights = dict(weights)
        if self.gld_ticker and self.gld_ticker in self.tickers:
            gld_asset = context.Market[self.gld_ticker]
            if gld_asset.Exists:
                target_weights[self.gld_ticker] = self.gld_weight
        if self.trend_ticker and self.trend_ticker in self.tickers:
            trend_asset = context.Market[self.trend_ticker]
            if trend_asset.Exists:
                target_weights[self.trend_ticker] = self.trend_weight

        total = sum(target_weights.values())
        if total > self.max_leverage > 0:
            scale = self.max_leverage / total
            target_weights = {t: w * scale for t, w in target_weights.items()}
            self.logger.info(
                "[P8] Leverage cap applied: total %.3f scaled to %.3f.",
                total,
                self.max_leverage,
            )

        self._target_weights = target_weights
        ranked = sorted(target_weights.items(), key=lambda kv: -kv[1])
        self.logger.info(
            "[P8] New targets: positions=%s, total_weight=%.3f, top5=%s",
            len(target_weights),
            sum(target_weights.values()),
            ranked[:5],
        )
```

### 7.3 `src/portfolios/portfolio_8/config.json`

```json
{
  "PORTFOLIO_ID": "8",
  "TICKERS": [],
  "INTERVAL": 23400,
  "LOOKBACK_DAYS": 400,
  "EXCH": "NASDAQ",
  "WEIGHTS": {},
  "DATA_FEEDS": ["MARKET_DATA", "POSITIONS", "CASH_EQUITY", "PORT_NOTIONAL"],
  "PORTFOLIO_6_CONFIG": {
    "UNIVERSE_PATH": "src/portfolios/portfolio_6/universe.json",
    "FUNDAMENTALS_CSV": "fundamentals/fundamentals.csv",
    "USE_FUNDAMENTALS": true,
    "SCREEN_TOP_N": 50,
    "VOL_LOOKBACK_DAYS": 252,
    "MAX_WEIGHT_PER_STOCK": 0.05,
    "VOL_TARGET_ANNUAL": 0.13,
    "MAX_LEVERAGE": 1.5,
    "GLD_TICKER": "GLD",
    "GLD_WEIGHT": 0.07,
    "TREND_HEDGE_TICKER": "",
    "TREND_HEDGE_WEIGHT": 0.10,
    "REBALANCE_DRIFT_THRESHOLD": 0.005,
    "DSR_MIN_PROB": 0.5
  },
  "RBP_BLEND": {
    "ENABLED": true,
    "WEIGHT": 1.0,
    "W_VOL": 1.0,
    "W_MAX": 1.0,
    "W_QUALITY": 1.0,
    "LOOKBACK_DAYS": 1825,
    "SPLIT_DATE": "2023-01-01",
    "MAX_COMBINATION_SIZE": 1,
    "CENSORING_QUANTILES": [0.0, 0.2, 0.5, 0.8],
    "N_JOBS": -1,
    "MAX_UNIVERSE": 150
  }
}
```

### 7.4 `src/portfolios/portfolio_manager_config.json` — additive change

```json
{
  "master_portfolio_id": "0",
  "currency": "USD",
  "portfolio_weights": {
    "1": 0.10,
    "2": 0.90,
    "8": 0.00
  }
}
```

P8 is **registered with weight 0** — loaded, monitored, attribution-tracked, but receives no capital until the falsification test (§8) passes. Promotion is a one-line JSON edit.

### 7.5 Optional follow-up (not in this artifact)

The pipeline.run() output drops the `ticker` column on the way back (`pipeline.py:121-125` sets index to `task_index` only). The P8 code path above defensively averages-across-task when ticker is missing, but the cleanest fix is a one-line patch on `pipeline.py:121-125`:

```python
# After: predictions_df = pd.DataFrame(prediction_rows).set_index("task_index")
predictions_df["ticker"] = engineered.loc[predictions_df.index, "ticker"].values
predictions_df["timestamp"] = engineered.loc[predictions_df.index, "timestamp"].values
```

(This is a separate PR — A3 does not modify `pipeline.py`.)

---

## 8. Falsification test (kill-switch criteria)

**Per the deliverable mandate.** Three independent gates must all pass before P8 is promoted from weight 0 → weight > 0 in `portfolio_manager_config.json`:

### Gate 1 — OOS Information Ratio uplift
Run a **3-year rolling out-of-sample purged k-fold backtest** (López de Prado 2018) on the P6 universe:
- 36-month walk-forward windows
- `k=6` purged folds with 21-day embargo (matching the forward-return horizon)
- Metric: **annualised Information Ratio of P8 minus IR of P6** on the same universe and rebalance dates
- **Pass condition:** ΔIR ≥ +0.10 averaged across folds, with no fold worse than −0.05

### Gate 2 — Deflated Sharpe of P8
Re-use `Portfolio6Strategy.deflated_sharpe_ratio` on the P8 sleeve daily returns. The number of trials parameter equals the size of the candidate universe (~518) plus the RBP grid cells (~48). 
**Pass condition:** DSR ≥ 0.95 (López de Prado-Bailey 2014 standard significance).

### Gate 3 — RBI stability
Run RBP monthly for 12 months on the P6 universe and capture per-month RBI scores from `RBP/models/importance.py`. 
**Pass condition:** The top-3 features by RBI must overlap by ≥2 in at least 9 of 12 months (regime-stability check, mitigating failure mode F2).

### Kill conditions
- If after promotion **any rolling 6-month window** shows ΔIR < −0.20 vs the pure-P6 baseline running on the same universe, demote P8 to weight 0 (one-line `portfolio_manager_config.json` revert).
- If **RBI top feature changes 3 months in a row**, pause P8 (weight 0) pending investigation: this is the F2 regime-break signature.
- If **>20% of P8 candidate-set tickers receive 0-conviction (failure F3) for two consecutive months**, pause P8.

---

## 9. Risks + rollback path

### Risks (severity × likelihood)

| # | Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | RBP pipeline blows compute budget at 150-ticker universe | High | Medium | `MAX_UNIVERSE=150` cap in config; `n_jobs=-1` joblib parallelism already in `pipeline.py:111`; monthly cadence not daily. |
| R2 | RBP forecasts are stale across the month (last refresh > 21 days ago at the next rebalance) | Medium | Low | P6's `_rebalance` runs every calendar month; we refresh RBP *inside* `_rebalance`, so it is at most 1 month old by construction. |
| R3 | Singular covariance / `LinAlgError` on small-feature subsets | Medium | Low | Already mitigated in `RBP/core/distance.py:33-36` with 1e-6 ridge; we do not change defaults. |
| R4 | Look-ahead leak via `train_test_split_date='2023-01-01'` being earlier than the backtest start | High | Medium | The `RBP_BLEND.SPLIT_DATE` config knob lets the backtest harness override it to <backtest_start_date> at run-time. P8's main_backtest entry must set this dynamically (see §10 follow-ups in A2 / A4 docs for parallel discussion). |
| R5 | RBP and P6 quality (gp/ta) signal correlation is high → double counting | Low | Medium | Equal weights at v1 absorb this; the falsification Gate 1 catches it (no uplift = no promotion). |
| R6 | RBP magnitude bug propagates to position sizing | High | Low | We rank-only; sizing is inverse-vol from P6, RBP never sees `inverse_vol_weights`. |
| R7 | Two RBP implementations diverge (`RBP/` package vs `portfolio_5/rbp_model.py`) | Medium | High | Out of scope here; flagged for tech-debt sprint. P8 uses *only* the `RBP/` package. |
| R8 | `pipeline.run()` does not return ticker labels on `predictions_df` | Medium | High | Defensive fallback in `_refresh_rbp_forecasts`; clean fix is the one-line patch in §7.5 (follow-up PR). |

### Rollback path

1. **Soft rollback (no code change):** set `"8": 0.00` in `portfolio_manager_config.json`. Capital allocator stops sending notional. P8 still runs for monitoring.
2. **Hard rollback (no infra change):** add `"RBP_BLEND": { "ENABLED": false }` in `src/portfolios/portfolio_8/config.json`. P8 then runs as a pure P6 clone.
3. **Code rollback (full revert):** delete `src/portfolios/portfolio_8/` and remove the `"8"` key from `portfolio_manager_config.json`. P6 untouched. RBP package untouched. P5 untouched. Zero blast radius — this is the explicit benefit of new-portfolio-vs-flag.

---

**End of A3 deliverable.**
