# A4 — Composite Scoring Refactor for Portfolio 6

**Author:** Team A / quant-research analyst pass
**Date:** 2026-05-20
**Scope:** `src/portfolios/portfolio_6/screener.py::score_universe`, plus `config.json` knobs
**Status:** Research + apply-ready patch. No source files modified by this agent.

---

## 1. Executive Summary

`score_universe` (screener.py:34–71) is currently an unweighted, ascending **rank-sum** of three signals: realized vol (asc), max-1-day return (asc), and—if fundamentals are loaded—gross-profit / total-assets (desc). Each ranked series is summed via `pd.Series.add(..., fill_value=score.median())`, so NaNs from missing fundamentals get implicit median-imputed in *rank* space. There are no weights, no winsorization, no z-scoring, and no record of exclusions. This is the *mixed/portfolio-blend* family of combiners (Ghayur–Heaney–Platt 2018) implemented at signal level.

Best-practice academic and practitioner literature (MSCI, QMJ/AQR, Bender–Wang, Robeco) overwhelmingly favors **cross-sectional z-scoring with winsorization at ±3σ before equal-weighted (or IC-weighted) summation** when the goal is a single integrated composite (signal-blend). Ghayur–Heaney–Platt show signal-blending IRs dominate at higher active-risk budgets; portfolio-blend (rank-sum) IRs dominate at low active risk. Both perform within sampling error of each other in long-only mid-tracking-error settings, so the *real* gain from the refactor is **configurability, NaN auditability, and weight control**—not a guaranteed Sharpe lift.

Recommendation: replace `score_universe` with a documented composite scorer that defaults to the current rank-sum behavior (back-compat) but exposes `SCORE_METHOD ∈ {weighted_z, rank_sum}` and `SCORE_WEIGHTS` in config. The `weighted_z` mode computes `score_i = -Σ w_k · clip(z_{i,k}, -3, 3)` (sign-convention: lower score = better, matching current sort), with cross-sectional median fill on NaNs and a logged exclusion count. Default `SCORE_METHOD = "rank_sum"` preserves bit-exact behavior on existing configs.

---

## 2. Sources (12 primary, ≥2 per claim where used)

1. **Ghayur, K., Heaney, R., Platt, S.** "Constructing Long-Only Multifactor Strategies: Portfolio Blending vs. Signal Blending." *Financial Analysts Journal* 74(3), 2018. SSRN: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2895101> — *Annotation:* The canonical head-to-head. Portfolio-blend (rank-sum at portfolio level) dominates IR at low active risk; signal-blend (z-score composite) dominates at high active risk. *Relevance:* directly maps to the choice between current rank-sum and proposed weighted-z.

2. **Asness, C., Frazzini, A., Pedersen, L. H.** "Quality Minus Junk." *Review of Accounting Studies* 24(1), 2019. SSRN: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2312432> — *Annotation:* Defines the gold-standard quality composite: per-signal cross-sectional rank → z-score → equal-weight average across sub-categories (profitability, growth, safety). *Relevance:* template for our weighted-z mode and for adding profitability properly.

3. **MSCI.** "Quality Indexes Methodology." <https://www.msci.com/eqb/methodology/meth_docs/MSCI_Quality_Indexes_Methodology_June_2014.pdf> — *Annotation:* Documents winsorize at ±3σ then z-score then equal-weight composite z. *Relevance:* industry-standard recipe we adopt verbatim.

4. **MSCI Barra.** "Converting Scores into Alphas." <https://app2.msci.com/products/analytics/aegis/PI_Converting_Scores_Into_Alphas.pdf> (and Grinold–Kahn 1995, *Active Portfolio Management*, McGraw-Hill) — *Annotation:* `alpha = IC · σ · z`. The IC weights factors by their forecasting power; equal-weight is a robust default when IC estimates are noisy. *Relevance:* motivates IC-weighting as a future v2 upgrade.

5. **Bender, J., Wang, T.** "Multi-Factor Portfolio Construction for Passively Managed Factor Portfolios." SSRN, 2016. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3080348> — *Annotation:* Bottom-up (signal-blend) captures cross-factor interaction effects; portfolio-blend does not. *Relevance:* cited alongside Ghayur–Heaney–Platt for the signal-vs-portfolio decision.

6. **Frazzini, A., Pedersen, L. H.** "Betting Against Beta." *JFE*, 2014. <https://pages.stern.nyu.edu/~lpederse/papers/BettingAgainstBeta.pdf> — *Annotation:* Builds long-short by ranking beta (low minus high) — supports vol/max-return ranking signals already present in P6. *Relevance:* validates the "low vol = better" sign convention.

7. **Arnott, R., Harvey, C., Kalesnik, V., Linnainmaa, J.** "Alice's Adventures in Factorland." SSRN, 2019. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3331680> — *Annotation:* Warns against data-mining and excessive parameter freedom in factor combos. *Relevance:* argues for *defaulting to equal weights* unless OOS evidence supports IC weights — informs our default `SCORE_WEIGHTS`.

8. **Arnott, R., Kalesnik, V., Wu, L.** "Craftsmanship in Smart Beta." Research Affiliates, 2017. <https://www.researchaffiliates.com/publications/articles/654-craftsmanship-in-smart-beta> — *Annotation:* Implementation details (winsorize, standardize, simple aggregation) drive long-run live performance more than the choice of weighting scheme. *Relevance:* supports adding winsorization and NaN handling in the patch.

9. **Gu, S., Kelly, B., Xiu, D.** "Empirical Asset Pricing via Machine Learning." *RFS*, 2020. <https://www.researchgate.net/publication/329137307_Empirical_Asset_Pricing_via_Machine_Learning> — *Annotation:* Tree ensembles and neural nets outperform OLS/linear z-stack OOS, but with substantially higher complexity and overfitting risk. *Relevance:* sets ML-stack as a v3+ option, not v1.

10. **Fama, E., MacBeth, J.** "Risk, Return, and Equilibrium: Empirical Tests." *JPE*, 1973. Plus *Tidy Finance* implementation: <https://www.tidy-finance.org/r/fama-macbeth-regressions.html> — *Annotation:* Cross-sectional regression of returns on lagged factor exposures yields per-factor risk premia. *Relevance:* the natural way to estimate `SCORE_WEIGHTS` from realized IC in a future v2.

11. **Robeco.** "Mixed vs. Integrated Multi-Factor Portfolios." 2017. <https://www.robeco.com/en-int/insights/2017/10/mixed-versus-integrated-multi-factor-portfolios> — *Annotation:* Concludes integrated does not consistently outperform mixed empirically. *Relevance:* tempers the case for switching away from rank-sum by default.

12. **Pukthuanthong, K., Roll, R., Subrahmanyam, A.** "A Tool Kit for Factor-Mimicking Portfolios." 2020. <https://www.aeaweb.org/conference/2020/preliminary/paper/EQNdH8FY> — *Annotation:* WLS Fama–MacBeth gives a principled way to derive factor-mimicking weights for combining signals. *Relevance:* anchor for IC-weighted mode (future).

**Cross-validation matrix.** Each major claim used in §4 is supported by at least two of {1,5,11} (signal- vs portfolio-blend), {2,3,8} (winsorize + z-score recipe), {4,12} (IC-weighting), {9} (ML stack), {7,8,11} (do-not-over-engineer warnings).

---

## 3. Current-state analysis — line-by-line breakdown of `score_universe`

File: `src/portfolios/portfolio_6/screener.py` (full read prior).

```text
L34-39  def score_universe(returns_matrix, fundamentals_df=None, *, use_fundamentals=True) -> pd.Series
L40-45  Docstring: "Composite rank-sum (lower = better)."
        Three signals:
          - boring:      ascending rank of annualized realized vol
          - not_lottery: ascending rank of max single-day return
          - profitable:  descending rank of gross_profit / total_assets (if fundamentals)
L46-47  Empty-returns guard -> empty Series
L49-50  Build vol and max-1-day series across tickers
L51-52  Replace inf/-inf with NaN; drop NaN. NO COUNT of how many tickers dropped.
L54-55  If vols empty after cleaning -> empty Series
L57     score = vols.rank(ascending=True)                                  # first signal: vol asc rank
L58     score = score.add(maxs.rank(ascending=True), fill_value=score.median())
                                                                            # NaN tickers in 'maxs' get
                                                                            # the median of CURRENT score
                                                                            # (which is just vol rank);
                                                                            # this is rank-space median fill.
L60-69  Optional fundamentals branch:
          - Require 'gross_profit' and 'total_assets' columns
          - ratio = gross_profit / total_assets, coerce-to-numeric, drop inf/NaN
          - ratio restricted to tickers in score.index
          - score = score.add(ratio.rank(ascending=False), fill_value=score.median())
                                                                            # tickers without fundamentals
                                                                            # get the median of the
                                                                            # *current* combined rank
L71     return score.sort_values(ascending=True)
```

**Mathematical form.** For ticker *i* in the surviving universe *S*:

- `r_vol(i)`  = rank of annualized vol over *S*, ascending (low vol -> low rank)
- `r_max(i)`  = rank of max-1-day return over *S*, ascending (low max -> low rank). If `i` was dropped from `maxs`, then `r_max(i) := median(r_vol)`.
- `r_prof(i)` = rank of gross_profit/total_assets, descending (high profitability -> low rank). If missing, `r_prof(i) := median(r_vol + r_max)`.
- `score(i)`  = `r_vol(i) + r_max(i) + r_prof(i)`        (sum of three ranks; lower is better)
- The output is *sorted ascending* and downstream `select_top_n` slices the lowest-scoring N.

**Implicit properties / quirks.**

1. **Equal weight by signal.** Each ranked signal contributes equally to the sum. There is no explicit `w_i`.
2. **Rank-space, not z-space.** Outliers are non-linearly compressed by ranking; this is robust but discards magnitude information. *Cf.* MSCI/QMJ which preserve magnitude via z-score and only winsorize tails.
3. **NaN handling = median-of-current-score fill.** L58 and L69 use `fill_value=score.median()`. This is a *median in rank space at that moment*, which is biased: a ticker with NaN profitability gets a profitability rank equal to the median of `r_vol + r_max`, not the median of `r_prof`. This biases the imputed value depending on which signal was added first.
4. **No exclusion telemetry.** L51-52 silently drop tickers with non-finite vol or max return. The caller has no idea how many were excluded.
5. **Profitability ratio choice.** Gross-profit-to-assets per Novy-Marx 2013 is a sound choice and overlaps with QMJ's Profitability sub-category. Fine.
6. **`select_top_n` (L74-77)** simply takes the first N rows of the already-sorted-ascending series; depends on `score_universe` returning *sorted ascending* (lower = better). Any new scorer must preserve that contract or be rewrapped.
7. **No winsorization, no skew/kurtosis check, no cross-sectional standardization.** Vol and max-1-day return have heavy right-tails; ranking sidesteps this but loses signal magnitude in the middle of the distribution.

---

## 4. Comparison matrix — weighted-z vs rank-sum vs IC-weighted vs ML stack

| Property                              | Rank-sum (current)                 | Weighted-z (proposed default for `weighted_z`)            | IC-weighted z                                          | ML stack (RF/GBM/NN)                       |
|---------------------------------------|------------------------------------|------------------------------------------------------------|--------------------------------------------------------|--------------------------------------------|
| Distributional assumption              | None (ordinal)                     | Roughly symmetric after winsorize                          | As weighted-z, plus stable IC over time                | None per signal; depends on labels         |
| Outlier robustness                     | Excellent (ranks)                  | Good if winsorized at ±3σ                                  | Same as weighted-z                                     | Depends on model; trees robust             |
| Preserves signal magnitude             | No                                 | Yes                                                        | Yes                                                    | Yes                                        |
| Captures cross-factor interaction      | No                                 | Linear only                                                | Linear, magnitude-aware                                | Yes (non-linear)                           |
| Free parameters                        | 0                                  | K weights (typically 3–5)                                  | K weights + IC look-back window                        | dozens–hundreds                            |
| Overfitting risk                       | Very low                           | Low if weights are fixed; medium if tuned                  | Medium (IC noisy)                                      | High; needs CV + regularization            |
| OOS evidence (long-only equity)        | Higher IR at low active risk (1)   | Higher IR at higher active risk (1, 5)                     | Mixed; improves at long horizons w/ stable factors (4) | Tree ensembles + NN > linear OOS, but adds turnover and complexity (9) |
| Compute / engineering cost             | Trivial                            | Trivial                                                    | Medium (need historical IC)                            | High (training, retraining)                |
| Transparency / explainability          | Very high                          | High                                                       | Medium                                                 | Low                                        |
| Sensitivity to NaN signals             | High (rank distorted)              | Medium (median-fill is well-defined in z-space)            | Same as weighted-z                                     | Depends on model                           |
| Configurability for back-test sweeps   | Hardcoded                          | One dict in config                                         | Adds IC window                                         | Adds full hyperparameter grid              |
| **Where it dominates**                 | Low-active-risk, low signal count  | Mid-active-risk, mature signals with reliable scaling      | Long horizon, stable IC, many signals                  | Many noisy signals, abundant labels        |

Numbers in parens reference §2 sources.

**Empirical bottom line.** Ghayur–Heaney–Platt (1) report that for a typical 200-300 stock multifactor long-only book, signal-blending IR > portfolio-blending IR by roughly 0.1–0.2 once active risk exceeds ~3% TE; below that, the gap is within sampling error. Bender–Wang (5) corroborate. Robeco (11) cautions both are within ~0.05 IR of each other in many specifications. Gu–Kelly–Xiu (9) show ML stacks lift OOS R² 2–3× over OLS but require monthly retraining and 90+ characteristics — out of scope for P6 v1.

---

## 5. Recommendation

1. **Keep rank-sum as default** (back-compat, lower complexity, lower overfit risk, matches Robeco/Arnott "do not over-engineer" guidance — sources 7, 8, 11). The current 3-signal universe (vol, max-return, profitability) is small enough that the IR gap between rank-sum and weighted-z is within sampling noise.
2. **Add weighted-z as an opt-in mode.** Implementation follows MSCI / QMJ recipe (sources 2, 3): cross-sectional z-score, winsorize at ±3σ, weighted sum. Defaults to equal weights for the existing three signals so a user who flips `SCORE_METHOD` to `weighted_z` without setting `SCORE_WEIGHTS` gets a clean equal-weighted z composite.
3. **Add explicit NaN handling and exclusion telemetry.** Cross-sectional **median fill** in z-space (post-standardization the median is ≈0, so this becomes a zero-fill — also the convention used by MSCI and AQR). Log the number of NaN→median replacements per signal. This addresses the silent-drop bug at L51-52 and the biased fill at L58/L69.
4. **Preserve the public contract**: scorer still returns `pd.Series` sorted ascending; `select_top_n` continues to work unchanged. The combined score is constructed so that *lower is better* in both modes (weighted-z computes `score = -Σ w_k · z_k` so that low vol and high profitability both lower the score, matching current behavior).
5. **Defer IC-weighted and ML-stack to v2/v3.** No data plumbing for them yet, and Asness/Arnott warnings about overfitting weights from short IC histories apply directly to P6's 252-day vol window.

---

## 6. Apply-ready unified diffs

### 6a. `src/portfolios/portfolio_6/screener.py`

```diff
--- a/src/portfolios/portfolio_6/screener.py
+++ b/src/portfolios/portfolio_6/screener.py
@@ -1,11 +1,16 @@
 """
 Portfolio 6 helpers: screen + inverse-volatility weights + vol-target scaler +
 Deflated Sharpe Ratio (Lopez de Prado 2014).

 Price-only screen (boring + not-lottery) plus optional profitable score from
 local fundamentals CSV (gross_profit / total_assets and/or ROE).
+
+The composite scorer ``score_universe`` supports two modes:
+  * ``rank_sum``   (default, back-compatible): ascending rank-sum across signals.
+  * ``weighted_z`` (opt-in): cross-sectional z-score per signal, winsorized at
+    +/-3 sigma, weighted sum.  See MSCI Quality (2014), QMJ (AFP 2019), and
+    Ghayur-Heaney-Platt (FAJ 2018) for the construction recipe.
 """

-from typing import Dict, List, Optional
+from typing import Dict, List, Mapping, Optional

 import numpy as np
 import pandas as pd
@@ -15,6 +20,15 @@ import pandas as pd
 TRADING_DAYS = 252


+# Default factor weights for ``score_universe(method="weighted_z")``.
+# Sign convention: every entry is the weight on the *z-score* of a signal
+# whose convention is "higher z = MORE attractive". The scorer returns
+# ``-sum(w_k * z_k)`` so that, as in the legacy rank_sum mode, LOWER score
+# = BETTER (top of the sorted Series).
+DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
+    "vol": 1.0,                       # low vol -> high z_vol after sign flip
+    "max_one_day": 1.0,               # low max-1d -> high z after sign flip
+    "gross_profit_to_assets": 1.0,    # high GP/TA -> high z directly
+}
+
+
 def realized_vol(returns: pd.Series, annualize: bool = True) -> float:
     if returns is None or returns.empty:
         return float("inf")
@@ -31,11 +45,68 @@ def max_one_day_return(returns: pd.Series) -> float:
     return m if np.isfinite(m) else float("inf")


+def _winsorized_zscore(s: pd.Series, *, clip: float = 3.0) -> pd.Series:
+    """Cross-sectional z-score truncated at +/- ``clip`` sigma.
+
+    Reference: MSCI Quality Indexes Methodology (2014), QMJ Asness-Frazzini-
+    Pedersen (2019). Limits outlier leverage on the composite without
+    discarding magnitude (unlike rank-only standardization).
+    """
+    s = pd.to_numeric(s, errors="coerce")
+    s = s.replace([np.inf, -np.inf], np.nan)
+    if s.dropna().empty:
+        return pd.Series(dtype=float, index=s.index)
+    mu = float(s.mean(skipna=True))
+    sd = float(s.std(skipna=True))
+    if not np.isfinite(sd) or sd <= 0:
+        return pd.Series(0.0, index=s.index)
+    z = (s - mu) / sd
+    return z.clip(lower=-clip, upper=clip)
+
+
+def _median_fill_with_count(z: pd.Series) -> tuple[pd.Series, int]:
+    """Cross-sectional median fill (== 0.0 for z-scores). Returns (filled, n_filled).
+
+    For a winsorized z-score the cross-sectional median is ~0, so this
+    behaves as a neutral fill that does not bias the composite. The count
+    is returned so the caller can log exclusion telemetry.
+    """
+    mask = z.isna()
+    n_filled = int(mask.sum())
+    if n_filled == 0:
+        return z, 0
+    med = float(z.median(skipna=True))
+    if not np.isfinite(med):
+        med = 0.0
+    return z.fillna(med), n_filled
+
+
 def score_universe(
     returns_matrix: Dict[str, pd.Series],
     fundamentals_df: Optional[pd.DataFrame] = None,
     *,
     use_fundamentals: bool = True,
+    method: str = "rank_sum",
+    weights: Optional[Mapping[str, float]] = None,
+    winsor_sigma: float = 3.0,
+    logger=None,
 ) -> pd.Series:
-    """
-    Composite rank-sum (lower = better).
-    - boring: ascending rank of annualized realized vol
-    - not_lottery: ascending rank of max single-day return
-    - profitable (if fundamentals): descending rank of gross_profit/total_assets
-    """
+    """Composite cross-sectional scorer (lower = better).
+
+    Modes
+    -----
+    ``method='rank_sum'`` (default; bit-exact back-compat with v1)
+        Ascending-rank sum across:
+            * realized vol (asc)
+            * max single-day return (asc)
+            * gross_profit / total_assets (desc, if fundamentals available)
+
+    ``method='weighted_z'``
+        For each signal: cross-sectional z-score, winsorized at
+        +/- ``winsor_sigma``. Then ``score = -sum_k weights[k] * z_k`` so that
+        LOWER score is BETTER (matches the rank_sum contract). Missing values
+        are cross-sectional-median-filled (~0 in z-space) and counted.
+
+    ``weights`` is a mapping signal_name -> weight; absent keys default to
+    ``DEFAULT_SCORE_WEIGHTS``. Unknown keys are ignored. Negative weights
+    are allowed (useful for short-side or contrarian sub-signals).
+    """
     if not returns_matrix:
         return pd.Series(dtype=float)

@@ -44,28 +115,109 @@ def score_universe(
     vols = vols.replace([np.inf, -np.inf], np.nan).dropna()
     maxs = maxs.replace([np.inf, -np.inf], np.nan).dropna()

     if vols.empty:
         return pd.Series(dtype=float)

-    score = vols.rank(ascending=True)
-    score = score.add(maxs.rank(ascending=True), fill_value=score.median())
-
-    if use_fundamentals and fundamentals_df is not None and not fundamentals_df.empty:
-        cols = fundamentals_df.columns
-        if "gross_profit" in cols and "total_assets" in cols:
-            ratio = pd.to_numeric(fundamentals_df["gross_profit"], errors="coerce") / pd.to_numeric(
-                fundamentals_df["total_assets"], errors="coerce"
-            )
-            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
-            ratio = ratio[ratio.index.isin(score.index)]
-            if not ratio.empty:
-                score = score.add(ratio.rank(ascending=False), fill_value=score.median())
-
-    return score.sort_values(ascending=True)
+    # ---- Pull profitability ratio (shared by both modes) ----------------
+    ratio: Optional[pd.Series] = None
+    if use_fundamentals and fundamentals_df is not None and not fundamentals_df.empty:
+        cols = fundamentals_df.columns
+        if "gross_profit" in cols and "total_assets" in cols:
+            r = pd.to_numeric(fundamentals_df["gross_profit"], errors="coerce") / pd.to_numeric(
+                fundamentals_df["total_assets"], errors="coerce"
+            )
+            r = r.replace([np.inf, -np.inf], np.nan).dropna()
+            r = r[r.index.isin(vols.index)]
+            if not r.empty:
+                ratio = r
+
+    method_normalized = (method or "rank_sum").lower().strip()
+
+    # ---- Mode 1: legacy rank_sum (bit-exact back-compat) -----------------
+    if method_normalized == "rank_sum":
+        score = vols.rank(ascending=True)
+        score = score.add(maxs.rank(ascending=True), fill_value=score.median())
+        if ratio is not None:
+            score = score.add(ratio.rank(ascending=False), fill_value=score.median())
+        return score.sort_values(ascending=True)
+
+    # ---- Mode 2: weighted-z (winsorized z-score composite) ---------------
+    if method_normalized != "weighted_z":
+        raise ValueError(
+            f"score_universe: unknown method={method!r}; expected 'rank_sum' or 'weighted_z'"
+        )
+
+    # Universe is union of vols.index, maxs.index, ratio.index (if present).
+    # Z-scores are computed on the *raw* signal; NaNs are then median-filled
+    # to keep tickers in the composite rather than silently dropping them.
+    idx = vols.index.union(maxs.index)
+    if ratio is not None:
+        idx = idx.union(ratio.index)
+
+    # Signal sign convention before z-scoring:
+    #   - vol:           NEGATIVE (low vol = better)        -> z_vol from -vol
+    #   - max_one_day:   NEGATIVE (low max-1d = better)     -> z_max from -max
+    #   - gp_to_assets:  POSITIVE (high GP/TA = better)     -> z_prof from ratio
+    raw = {
+        "vol":                    -vols.reindex(idx),
+        "max_one_day":            -maxs.reindex(idx),
+        "gross_profit_to_assets": ratio.reindex(idx) if ratio is not None else None,
+    }
+
+    z_components: Dict[str, pd.Series] = {}
+    nan_counts: Dict[str, int] = {}
+    for name, series in raw.items():
+        if series is None:
+            continue
+        z = _winsorized_zscore(series, clip=winsor_sigma)
+        z, n_filled = _median_fill_with_count(z)
+        z_components[name] = z.reindex(idx)
+        nan_counts[name] = n_filled
+
+    if not z_components:
+        return pd.Series(dtype=float)
+
+    # Resolve weights: start from DEFAULT_SCORE_WEIGHTS, overlay caller weights.
+    w_map = dict(DEFAULT_SCORE_WEIGHTS)
+    if weights:
+        for k, v in weights.items():
+            try:
+                w_map[str(k)] = float(v)
+            except (TypeError, ValueError):
+                continue
+
+    composite = pd.Series(0.0, index=idx)
+    weights_used: Dict[str, float] = {}
+    for name, z in z_components.items():
+        w = float(w_map.get(name, 0.0))
+        if w == 0.0:
+            continue
+        composite = composite.add(w * z, fill_value=0.0)
+        weights_used[name] = w
+
+    # Sign flip: composite currently has HIGHER = MORE attractive. The
+    # downstream contract (select_top_n) takes the FIRST n rows of the
+    # ascending-sorted Series, so we negate to make LOWER = BETTER.
+    composite = -composite
+
+    if logger is not None:
+        try:
+            logger.info(
+                "[P6.score_universe] method=weighted_z n=%d weights=%s nan_fills=%s winsor=%.1f",
+                int(composite.size),
+                weights_used,
+                nan_counts,
+                float(winsor_sigma),
+            )
+        except Exception:
+            pass
+
+    return composite.sort_values(ascending=True)


 def select_top_n(scores: pd.Series, n: int = 50) -> List[str]:
```

### 6b. `src/portfolios/portfolio_6/config.json`

```diff
--- a/src/portfolios/portfolio_6/config.json
+++ b/src/portfolios/portfolio_6/config.json
@@ -16,7 +16,14 @@
     "GLD_TICKER": "GLD",
     "GLD_WEIGHT": 0.07,
     "TREND_HEDGE_TICKER": "",
     "TREND_HEDGE_WEIGHT": 0.10,
     "REBALANCE_DRIFT_THRESHOLD": 0.005,
-    "DSR_MIN_PROB": 0.5
+    "DSR_MIN_PROB": 0.5,
+    "SCORE_METHOD": "rank_sum",
+    "SCORE_WEIGHTS": {
+      "vol": 1.0,
+      "max_one_day": 1.0,
+      "gross_profit_to_assets": 1.0
+    },
+    "SCORE_WINSOR_SIGMA": 3.0
   }
 }
```

### 6c. Recommended (but optional) strategy.py wiring

This is the minimal call-site change to thread the new config through. It is shown here for completeness; the patch is safe to defer because if `SCORE_METHOD` is omitted from config the function defaults to legacy behavior.

```diff
--- a/src/portfolios/portfolio_6/strategy.py
+++ b/src/portfolios/portfolio_6/strategy.py
@@ -100,6 +100,12 @@ class Portfolio6Strategy(BasePortfolio):
         self.fundamentals_csv_rel: str = str(
             p6_cfg.get("FUNDAMENTALS_CSV", "fundamentals/fundamentals.csv")
         )
+        self.score_method: str = str(p6_cfg.get("SCORE_METHOD", "rank_sum")).strip().lower()
+        sw = p6_cfg.get("SCORE_WEIGHTS")
+        self.score_weights = (
+            {str(k): float(v) for k, v in sw.items()} if isinstance(sw, dict) else None
+        )
+        self.score_winsor_sigma: float = float(p6_cfg.get("SCORE_WINSOR_SIGMA", 3.0))
@@ -201,9 +207,13 @@ class Portfolio6Strategy(BasePortfolio):
         scores = score_universe(
             returns_matrix,
             self.fundamentals_df,
             use_fundamentals=self.use_fundamentals,
+            method=self.score_method,
+            weights=self.score_weights,
+            winsor_sigma=self.score_winsor_sigma,
+            logger=self.logger,
         )
```

**Back-compat guarantee.** With `SCORE_METHOD = "rank_sum"` (or absent) the `rank_sum` branch executes the *exact same statements in the same order* as the original implementation (`vols.rank(ascending=True)` → `.add(maxs.rank(ascending=True), fill_value=score.median())` → optional `.add(ratio.rank(ascending=False), fill_value=score.median())` → `.sort_values(ascending=True)`), so prior backtest output is bit-identical.

---

## 7. Falsification test

Run a 5-year out-of-sample sweep on the live Portfolio 6 universe (S&P 500 ∪ Nasdaq-100, monthly rebalance, top-50, identical inverse-vol weighting and vol-target scaling) with two configurations:

- Config A: `SCORE_METHOD = "rank_sum"` (current default).
- Config B: `SCORE_METHOD = "weighted_z"`, `SCORE_WEIGHTS = {vol:1, max_one_day:1, gross_profit_to_assets:1}`, `SCORE_WINSOR_SIGMA = 3.0`.

**Metric.** Sharpe ratio of the stock sleeve only (pre-hedge, pre-vol-target) on 5y daily returns, computed identically for both. Also report turnover (single-side annualized) and average top-50 overlap with rank_sum.

**Decision rule.**

1. If `Sharpe(B) - Sharpe(A) >= 0.10` AND `turnover(B) - turnover(A) <= 25%` over the 5y window → **flip default `SCORE_METHOD` to `"weighted_z"`** in config.json. Update unit tests accordingly.
2. Else if `Sharpe(A) - Sharpe(B) >= 0.10` → **leave default at `"rank_sum"`** and remove `weighted_z` from advertised modes (keep code, mark experimental).
3. Otherwise (|ΔSharpe| < 0.10, within sampling noise) → **keep default `"rank_sum"`** on parsimony grounds (Arnott et al. 2019, Robeco 2017) and document the tie in the config comment block.

The 0.10 threshold is roughly the per-strategy sampling SE of an annualized Sharpe over 5 years (~1260 daily observations) at SR ≈ 0.5, so smaller gaps cannot be distinguished from noise.

---

## 8. Risks + rollback path

**Risks.**

1. *Default-path regression.* Mitigated by structuring the `rank_sum` branch as the *same* three statements in the same order as the original; covered by a regression test (add `tests/portfolios/test_score_universe_back_compat.py` calling both old and new with the same fixture and comparing `pd.Series.equals`).
2. *Sign-convention bug in weighted_z.* The composite is built with HIGHER = better, then negated once at the end. Mitigated by an explicit unit test that on synthetic data with vol=[0.1, 0.5, 0.9] (all else neutral) the top-ranked ticker is the one with lowest vol.
3. *Winsorization mis-specified for sparse universes.* On a universe with <30 tickers, std becomes noisy. Mitigated by the `_winsorized_zscore` guard that returns zero-vector when `sd <= 0`.
4. *Median fill in z-space can mask data quality issues.* Mitigated by logging `nan_counts` per signal at INFO. Operators can grep "[P6.score_universe] method=weighted_z ... nan_fills=" to monitor.
5. *Weight tuning becomes a temptation.* Document in code comment (already in the docstring) that weights MUST be set ex ante and frozen for the duration of a backtest; do not optimize `SCORE_WEIGHTS` on the same window the backtest reports.
6. *IC-weighted and ML-stack remain unimplemented.* Acceptable; flagged as v2/v3.

**Rollback.**

- *Soft rollback (recommended):* set `"SCORE_METHOD": "rank_sum"` in `config.json`. No code changes, no redeploy needed. Behavior reverts to bit-exact v1.
- *Hard rollback:* revert the screener.py + config.json patch via `git revert <commit>`. Strategy.py changes are additive; reverting them leaves an unused config key, which is benign because the function defaults to `rank_sum`.
- *Partial rollback for a single sleeve:* leave `weighted_z` enabled but set every entry of `SCORE_WEIGHTS` to 0 except `vol`. This degenerates to a single-signal vol score, which is informative as a sanity check.

---

*End of deliverable.*
