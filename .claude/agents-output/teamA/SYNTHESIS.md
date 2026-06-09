# Team A Synthesis — Signal Generation

**Date:** 2026-05-20
**Inputs:** A1 (factor audit), A2 (P7 sentiment), A3 (P8 RBP), A4 (composite scoring), A5 (lottery exclusion).
**Status:** Read-only synthesis. Final diffs apply-ready below. No files modified by this agent.

---

## 1. Decisions

| # | Decision | Source(s) |
|---|---|---|
| D1 | Build **Portfolio_7** = P6 + sentiment tilt (A2). Subclass `Portfolio6Strategy`. | A2 |
| D2 | Build **Portfolio_8** = P6 + RBP rank-overlay (A3). Subclass `Portfolio6Strategy`. **Reject** P7-flag option. | A3 |
| D3 | Both P7 and P8 register at capital weight **0.0** in `portfolio_manager_config.json` until OOS gates pass. | A2, A3 |
| D4 | Refactor `score_universe` to dispatch on `method ∈ {rank_sum, weighted_z}`. Default `rank_sum` = bit-exact back-compat. | A4 |
| D5 | Add **3 factor flags** to P6 (momentum 12-2, op-profitability, asset-growth) — all default OFF. | A1 |
| D6 | Add **8 hard-exclusion rules** via new `exclusions.py` — all default OFF. Run pre-scoring. | A5 |
| D7 | **Single unified `score_universe` signature** absorbs A1+A4+A5 additions (this synthesis resolves the 3-way signature conflict). | new |
| D8 | **Blocker**: `fundamentals/fundamentals.csv` has 10/519 tickers populated and `roe` column empty. Factor flags from A1 are statistically meaningless until this is refreshed. | A1 §3 |

## 2. Architecture summary

```
P6 base universe (519 tickers)
   │
   ├─ apply_exclusions(EXCLUSIONS cfg)        ← A5 (default no-op)
   │
   ├─ score_universe(method, weights, factors) ← A4 + A1 (default rank_sum equal-weight)
   │   ├─ vol (asc)
   │   ├─ max-1d (asc)
   │   ├─ gross_profit/total_assets (desc, if cols)
   │   ├─ momentum 12-2 (desc, if USE_MOMENTUM_12_2)
   │   ├─ op_profitability (desc, if USE_OPERATING_PROFITABILITY + cols)
   │   └─ asset_growth (asc, if USE_ASSET_GROWTH + col)
   │
   ├─ select_top_n(SCREEN_TOP_N=50)
   ├─ inverse_vol_weights(MAX_WEIGHT_PER_STOCK)
   ├─ vol_target_scale(VOL_TARGET_ANNUAL, MAX_LEVERAGE)
   └─ hedge sleeves (GLD_WEIGHT + TREND_HEDGE_WEIGHT)

P7 = P6 + sentiment tilt overlay (post-screening, pre-hedge):
       w_i ← w_i · exp(λ · z_i)            ← A2
       z_i = cross-sectional std of 21d EWM mean sentiment
       (strict ex-ante: published_at < context.time)

P8 = P6 with extra RBP rank inside score_universe:           ← A3
       composite_score += W_RBP · rank_desc(rbp_forecast_21d)
       RBP refreshed monthly inside _rebalance via RBPPipeline.run()
```

## 3. Unified `score_universe` signature (resolves A1+A4+A5 conflict)

```python
def score_universe(
    returns_matrix: Dict[str, pd.Series],
    fundamentals_df: Optional[pd.DataFrame] = None,
    *,
    # A4 — method dispatch
    method: str = "rank_sum",                              # {"rank_sum", "weighted_z"}
    weights: Optional[Mapping[str, float]] = None,         # per-leg weights
    winsor_sigma: float = 3.0,
    # A5 — pre-scoring exclusion stage
    exclusions_cfg: Optional[Dict] = None,
    # A1 — extra factor flags
    use_fundamentals: bool = True,
    use_momentum: bool = False,
    use_op_profitability: bool = False,
    use_asset_growth: bool = False,
    momentum_lookback_days: int = 252,
    momentum_skip_days: int = 21,
    # observability
    logger=None,
) -> pd.Series:
```

Leg names used in `weights` dict (consistent across modes):
- `vol`, `max_one_day`, `gross_profit_to_assets`, `momentum`, `op_profit`, `asset_growth`, `rbp` (P8 only).

Default weights = `1.0` per active leg → bit-exact back-compat with the current rank-sum implementation when all new flags are OFF.

## 4. Apply order (single PR)

The diffs must be applied in this order to avoid hunk conflicts. Each step is independently reversible (single config flip or git revert).

1. **NEW** `src/portfolios/portfolio_6/exclusions.py` (A5 §5 — apply verbatim).
2. **PATCH** `src/portfolios/portfolio_6/screener.py` — single coherent rewrite of `score_universe` per §3. The unified rewrite below supersedes A1 §6.2, A4 §6a, A5 §6a.
3. **PATCH** `src/portfolios/portfolio_6/strategy.py` — thread the new config knobs into the `score_universe` call (A1 §6.3 + A4 §6c + A5 §6a strategy.py hunks combined).
4. **PATCH** `src/portfolios/portfolio_6/config.json` — merge A1, A4, A5 config blocks into a single `PORTFOLIO_6_CONFIG` extension (one diff below).
5. **NEW** `src/portfolios/portfolio_7/{__init__.py, strategy.py, config.json}` (A2 §5).
6. **NEW** `src/portfolios/portfolio_8/{__init__.py, strategy.py, config.json}` (A3 §7).
7. **PATCH** `src/portfolios/portfolio_manager_config.json` — register P7 and P8 at weight 0.0.
8. **PATCH** `src/main_backtest.py` — import + register P7 and P8 classes.

## 5. Unified `score_universe` rewrite

```diff
--- a/src/portfolios/portfolio_6/screener.py
+++ b/src/portfolios/portfolio_6/screener.py
@@ -1,11 +1,18 @@
 """
 Portfolio 6 helpers: screen + inverse-volatility weights + vol-target scaler +
 Deflated Sharpe Ratio (Lopez de Prado 2014).
 
-Price-only screen (boring + not-lottery) plus optional profitable score from
-local fundamentals CSV (gross_profit / total_assets and/or ROE).
+Price-only screen (boring + not-lottery) plus optional factor tilts from
+local fundamentals CSV. Score composite supports two modes:
+  - rank_sum   (default; bit-exact back-compat)
+  - weighted_z (winsorized z-score sum; MSCI/QMJ-style)
+
+New optional legs (all flag-gated, default OFF):
+  - momentum 12-2 (Jegadeesh-Titman 1993; Daniel-Moskowitz 2016)
+  - operating profitability (Ball-Gerakos-Linnainmaa-Nikolaev 2016 / FF RMW)
+  - asset growth (Cooper-Gulen-Schill 2008 / FF CMA)
+
+Hard-exclusion stage (Bali-Cakici-Whitelaw, Ang-Hodrick-Xing-Zhang, Kumar,
+Ritter, Hou-Xue-Zhang) is provided by exclusions.apply_exclusions() and
+wired through the exclusions_cfg kwarg.
 """
 
-from typing import Dict, List, Optional
+from typing import Dict, List, Mapping, Optional
 
 import numpy as np
 import pandas as pd
 
+try:
+    from src.portfolios.portfolio_6.exclusions import apply_exclusions
+except ImportError:
+    from portfolios.portfolio_6.exclusions import apply_exclusions
+
 
 TRADING_DAYS = 252
 
+DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
+    "vol": 1.0,
+    "max_one_day": 1.0,
+    "gross_profit_to_assets": 1.0,
+    "momentum": 1.0,
+    "op_profit": 1.0,
+    "asset_growth": 1.0,
+}
+
 
 def realized_vol(returns: pd.Series, annualize: bool = True) -> float:
     ...
@@ -28,11 +60,71 @@ def max_one_day_return(returns: pd.Series) -> float:
     return m if np.isfinite(m) else float("inf")
 
 
+def momentum_12_2(
+    returns: pd.Series,
+    *,
+    lookback_days: int = 252,
+    skip_days: int = 21,
+) -> float:
+    """Jegadeesh-Titman 12-2 cumulative return (t-lookback to t-skip)."""
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
+def _winsorized_zscore(s: pd.Series, *, clip: float = 3.0) -> pd.Series:
+    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
+    if s.dropna().empty:
+        return pd.Series(dtype=float, index=s.index)
+    sd = float(s.std(skipna=True))
+    if not np.isfinite(sd) or sd <= 0:
+        return pd.Series(0.0, index=s.index)
+    return ((s - float(s.mean(skipna=True))) / sd).clip(lower=-clip, upper=clip)
+
+
+def _median_fill(z: pd.Series) -> pd.Series:
+    if z.dropna().empty:
+        return z.fillna(0.0)
+    med = float(z.median(skipna=True))
+    return z.fillna(med if np.isfinite(med) else 0.0)
+
+
 def score_universe(
     returns_matrix: Dict[str, pd.Series],
     fundamentals_df: Optional[pd.DataFrame] = None,
     *,
+    method: str = "rank_sum",
+    weights: Optional[Mapping[str, float]] = None,
+    winsor_sigma: float = 3.0,
+    exclusions_cfg: Optional[Dict] = None,
     use_fundamentals: bool = True,
+    use_momentum: bool = False,
+    use_op_profitability: bool = False,
+    use_asset_growth: bool = False,
+    momentum_lookback_days: int = 252,
+    momentum_skip_days: int = 21,
+    logger=None,
 ) -> pd.Series:
-    """Composite rank-sum (lower = better)."""
+    """Composite scorer (lower = better).
+
+    method='rank_sum'  (default): ascending rank-sum across active legs.
+    method='weighted_z': cross-sectional z-score per leg, winsor at ±winsor_sigma,
+                        weighted sum, then negated so lower = better.
+
+    Legs (sign convention before any score-flip):
+      vol asc, max_one_day asc, gross_profit_to_assets desc,
+      momentum desc, op_profit desc, asset_growth asc.
+    """
     if not returns_matrix:
         return pd.Series(dtype=float)
 
+    # ---- A5: hard-exclusion stage (no-op when exclusions_cfg empty) -----
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
     maxs = maxs.replace([np.inf, -np.inf], np.nan).dropna()
     if vols.empty:
         return pd.Series(dtype=float)
 
-    score = vols.rank(ascending=True)
-    score = score.add(maxs.rank(ascending=True), fill_value=score.median())
+    # Compute optional legs (price-only & fundamentals)
+    moms: Optional[pd.Series] = None
+    if use_momentum:
+        moms = pd.Series({
+            t: momentum_12_2(r, lookback_days=momentum_lookback_days, skip_days=momentum_skip_days)
+            for t, r in returns_matrix.items()
+        })
+        moms = moms.replace([np.inf, -np.inf], np.nan).dropna()
+
+    gp_ratio: Optional[pd.Series] = None
+    op_ratio: Optional[pd.Series] = None
+    ag_series: Optional[pd.Series] = None
+    if use_fundamentals and fundamentals_df is not None and not fundamentals_df.empty:
+        cols = set(fundamentals_df.columns)
+        if {"gross_profit", "total_assets"}.issubset(cols):
+            gp_ratio = _safe_ratio(fundamentals_df["gross_profit"], fundamentals_df["total_assets"])
+            gp_ratio = gp_ratio[gp_ratio.index.isin(vols.index)]
+        if use_op_profitability and {"revenue", "cost_of_revenue", "sga", "interest_expense", "book_equity"}.issubset(cols):
+            num = (
+                pd.to_numeric(fundamentals_df["revenue"], errors="coerce")
+                - pd.to_numeric(fundamentals_df["cost_of_revenue"], errors="coerce")
+                - pd.to_numeric(fundamentals_df["sga"], errors="coerce")
+                - pd.to_numeric(fundamentals_df["interest_expense"], errors="coerce")
+            )
+            op_ratio = _safe_ratio(num, fundamentals_df["book_equity"])
+            op_ratio = op_ratio[op_ratio.index.isin(vols.index)]
+        if use_asset_growth and "asset_growth" in cols:
+            ag_series = pd.to_numeric(fundamentals_df["asset_growth"], errors="coerce")
+            ag_series = ag_series.replace([np.inf, -np.inf], np.nan).dropna()
+            ag_series = ag_series[ag_series.index.isin(vols.index)]
+
+    w_map = dict(DEFAULT_SCORE_WEIGHTS)
+    if weights:
+        for k, v in weights.items():
+            try:
+                w_map[str(k)] = float(v)
+            except (TypeError, ValueError):
+                continue
+
+    method_norm = (method or "rank_sum").lower().strip()
+
+    # ---- Mode 1: rank_sum (bit-exact back-compat when only vol+max+gp legs active) ----
+    if method_norm == "rank_sum":
+        score = vols.rank(ascending=True) * w_map["vol"]
+        score = score.add(
+            maxs.rank(ascending=True) * w_map["max_one_day"],
+            fill_value=score.median(),
+        )
+        if gp_ratio is not None and not gp_ratio.empty:
+            score = score.add(
+                gp_ratio.rank(ascending=False) * w_map["gross_profit_to_assets"],
+                fill_value=score.median(),
+            )
+        if moms is not None and not moms.empty:
+            moms_aligned = moms[moms.index.isin(score.index)]
+            score = score.add(
+                moms_aligned.rank(ascending=False) * w_map["momentum"],
+                fill_value=score.median(),
+            )
+        if op_ratio is not None and not op_ratio.empty:
+            score = score.add(
+                op_ratio.rank(ascending=False) * w_map["op_profit"],
+                fill_value=score.median(),
+            )
+        if ag_series is not None and not ag_series.empty:
+            score = score.add(
+                ag_series.rank(ascending=True) * w_map["asset_growth"],
+                fill_value=score.median(),
+            )
+        return score.sort_values(ascending=True)
+
+    # ---- Mode 2: weighted_z (winsorized z-sum, negated so lower = better) ----
+    if method_norm != "weighted_z":
+        raise ValueError(f"score_universe: unknown method={method!r}")
+
+    idx = vols.index.union(maxs.index)
+    for s in (gp_ratio, moms, op_ratio, ag_series):
+        if s is not None:
+            idx = idx.union(s.index)
+
+    raw = {
+        "vol":                    -vols.reindex(idx),
+        "max_one_day":            -maxs.reindex(idx),
+        "gross_profit_to_assets": gp_ratio.reindex(idx) if gp_ratio is not None else None,
+        "momentum":               moms.reindex(idx) if moms is not None else None,
+        "op_profit":              op_ratio.reindex(idx) if op_ratio is not None else None,
+        "asset_growth":           -ag_series.reindex(idx) if ag_series is not None else None,
+    }
+
+    composite = pd.Series(0.0, index=idx)
+    nan_counts: Dict[str, int] = {}
+    weights_used: Dict[str, float] = {}
+    for name, series in raw.items():
+        if series is None:
+            continue
+        w = float(w_map.get(name, 0.0))
+        if w == 0.0:
+            continue
+        z = _winsorized_zscore(series, clip=winsor_sigma)
+        nan_counts[name] = int(z.isna().sum())
+        z = _median_fill(z)
+        composite = composite.add(w * z.reindex(idx), fill_value=0.0)
+        weights_used[name] = w
 
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
-    return score.sort_values(ascending=True)
+    if logger is not None:
+        try:
+            logger.info(
+                "[P6.score_universe] method=weighted_z n=%d weights=%s nan_fills=%s winsor=%.1f",
+                int(composite.size), weights_used, nan_counts, float(winsor_sigma),
+            )
+        except Exception:
+            pass
+
+    # Sign flip so lower = better matches the rank_sum contract.
+    return (-composite).sort_values(ascending=True)
```

## 6. Unified `config.json` for P6

```diff
--- a/src/portfolios/portfolio_6/config.json
+++ b/src/portfolios/portfolio_6/config.json
@@ -16,7 +16,67 @@
     "GLD_TICKER": "GLD",
     "GLD_WEIGHT": 0.07,
     "TREND_HEDGE_TICKER": "",
     "TREND_HEDGE_WEIGHT": 0.10,
     "REBALANCE_DRIFT_THRESHOLD": 0.005,
-    "DSR_MIN_PROB": 0.5
+    "DSR_MIN_PROB": 0.5,
+
+    "SCORE_METHOD": "rank_sum",
+    "SCORE_WEIGHTS": {
+      "vol": 1.0,
+      "max_one_day": 1.0,
+      "gross_profit_to_assets": 1.0,
+      "momentum": 1.0,
+      "op_profit": 1.0,
+      "asset_growth": 1.0
+    },
+    "SCORE_WINSOR_SIGMA": 3.0,
+
+    "USE_MOMENTUM_12_2": false,
+    "USE_OPERATING_PROFITABILITY": false,
+    "USE_ASSET_GROWTH": false,
+    "MOMENTUM_LOOKBACK_DAYS": 252,
+    "MOMENTUM_SKIP_DAYS": 21,
+
+    "EXCLUSIONS": {
+      "MAX":            { "ENABLED": false, "WINDOW_DAYS": 21, "DROP_TOP_PCT": 0.20 },
+      "MAX_N":          { "ENABLED": false, "WINDOW_DAYS": 21, "N": 5, "DROP_TOP_PCT": 0.20 },
+      "IVOL":           { "ENABLED": false, "WINDOW_DAYS": 21, "MIN_OBS": 17, "DROP_TOP_PCT": 0.10 },
+      "PENNY":          { "ENABLED": false, "MIN_PRICE": 5.0 },
+      "MICROCAP":       { "ENABLED": false, "MIN_MARKET_CAP_USD": 300000000.0 },
+      "RECENT_IPO":     { "ENABLED": false, "MIN_LISTED_MONTHS": 12 },
+      "SHORT_INTEREST": { "ENABLED": false, "MAX_DELTA_PCT": 0.50 },
+      "PEAD":           { "ENABLED": false, "EXCLUDE_DAYS_PRE": 3, "EXCLUDE_DAYS_POST": 3 }
+    }
   }
 }
```

## 7. Unified `strategy.py` wire-up for P6

```diff
--- a/src/portfolios/portfolio_6/strategy.py
+++ b/src/portfolios/portfolio_6/strategy.py
@@ -96,6 +96,23 @@ class Portfolio6Strategy(BasePortfolio):
         self.use_fundamentals: bool = bool(p6_cfg.get("USE_FUNDAMENTALS", True))
         self.dsr_min_prob: float = float(p6_cfg.get("DSR_MIN_PROB", 0.5))
         self.fundamentals_csv_rel: str = str(
             p6_cfg.get("FUNDAMENTALS_CSV", "fundamentals/fundamentals.csv")
         )
+        # A4 — scoring mode
+        self.score_method: str = str(p6_cfg.get("SCORE_METHOD", "rank_sum")).strip().lower()
+        sw = p6_cfg.get("SCORE_WEIGHTS")
+        self.score_weights = (
+            {str(k): float(v) for k, v in sw.items()} if isinstance(sw, dict) else None
+        )
+        self.score_winsor_sigma: float = float(p6_cfg.get("SCORE_WINSOR_SIGMA", 3.0))
+        # A1 — extra factor flags
+        self.use_momentum_12_2: bool = bool(p6_cfg.get("USE_MOMENTUM_12_2", False))
+        self.use_op_profitability: bool = bool(p6_cfg.get("USE_OPERATING_PROFITABILITY", False))
+        self.use_asset_growth: bool = bool(p6_cfg.get("USE_ASSET_GROWTH", False))
+        self.momentum_lookback_days: int = int(p6_cfg.get("MOMENTUM_LOOKBACK_DAYS", 252))
+        self.momentum_skip_days: int = int(p6_cfg.get("MOMENTUM_SKIP_DAYS", 21))
+        # A5 — exclusions block
+        self.exclusions_cfg: dict = dict(p6_cfg.get("EXCLUSIONS", {}))
@@ -201,8 +218,18 @@ class Portfolio6Strategy(BasePortfolio):
         scores = score_universe(
             returns_matrix,
             self.fundamentals_df,
+            method=self.score_method,
+            weights=self.score_weights,
+            winsor_sigma=self.score_winsor_sigma,
+            exclusions_cfg=self.exclusions_cfg,
             use_fundamentals=self.use_fundamentals,
+            use_momentum=self.use_momentum_12_2,
+            use_op_profitability=self.use_op_profitability,
+            use_asset_growth=self.use_asset_growth,
+            momentum_lookback_days=self.momentum_lookback_days,
+            momentum_skip_days=self.momentum_skip_days,
+            logger=self.logger,
         )
```

## 8. P7 + P8 — apply as-is from A2 §5 and A3 §7

P7 and P8 are independent subclasses; they call `score_universe` with the same signature defined above. The synthesis does not modify the source provided by A2 or A3 — adopt verbatim.

P8 should add `"rbp"` weight to `SCORE_WEIGHTS` in its `PORTFOLIO_6_CONFIG` override AND in `screener.py::DEFAULT_SCORE_WEIGHTS` (extend list). Adjust A3 §7.2 `_compose_score` to call `score_universe(..., weights={...,"rbp":W_RBP})` rather than wrapping post-hoc — cleaner and avoids the median-fill bias in A3 §7.2. Defer this micro-improvement until P8 OOS testing.

## 9. Manager + backtest registration (single combined diff)

```diff
--- a/src/portfolios/portfolio_manager_config.json
+++ b/src/portfolios/portfolio_manager_config.json
@@ -1,8 +1,10 @@
 {
   "master_portfolio_id": "0",
   "currency": "USD",
   "portfolio_weights": {
     "1": 0.10,
-    "2": 0.90
+    "2": 0.90,
+    "7": 0.0,
+    "8": 0.0
   }
 }
```

```diff
--- a/src/main_backtest.py
+++ b/src/main_backtest.py
@@ -29,6 +29,8 @@
 from src.portfolios.portfolio_6.strategy import Portfolio6Strategy
+from src.portfolios.portfolio_7.strategy import Portfolio7Strategy
+from src.portfolios.portfolio_8.strategy import Portfolio8Strategy
@@ -62,6 +64,8 @@
     RBPStrategy,
     Portfolio6Strategy,
+    Portfolio7Strategy,
+    Portfolio8Strategy,
 ]
```

## 10. Blockers + follow-ups (cannot be silently skipped)

| # | Issue | Owner | Action |
|---|---|---|---|
| B1 | `fundamentals.csv` has 10/519 tickers populated; `roe` column entirely null. Existing profitability rank is statistical noise on ~98% of universe. | Operator | Re-run `scripts/APIs/fmp_fundamentals.py` on full P6 universe BEFORE enabling any factor flag from A1. Verify ROW count ≈ universe size and `gross_profit` non-null rate ≥ 95%. |
| B2 | A1 R2 (op-profit) and R3 (asset-growth) need new FMP columns: `revenue, cost_of_revenue, sga, interest_expense, book_equity, asset_growth`. | Operator | Add to `fmp_fundamentals.py` fetcher (separate PR). Until done, the legs no-op (`.issubset(cols)` guards). |
| B3 | A5 R4/R6/R7/R8 (IPO date, mcap, short int, earnings calendar) need new FMP columns. | Operator | Add via `/stable/profile`, `/stable/historical-short-interest`, `/stable/earnings-calendar`. Until done, rules are inert. |
| B4 | `news_sentiment` schema mismatch: column is `avg_sentiment` in `schemaDefinitions.py:51` but `repository.py:255` writes `sentiment_score`. P7 routes around via `COALESCE`. | DB owner | Reconcile column name in a separate PR. |
| B5 | `RBP/pipeline.py` does not return ticker labels with predictions. P8 has defensive fallback but the clean fix is the 2-line patch in A3 §7.5. | RBP author | Apply A3 §7.5 in a separate PR before P8 promotion to weight > 0. |
| B6 | RBP `train_test_split_date='2023-01-01'` is hardcoded in `RBP/config.py`. P8 backtests starting < 2023 leak future data. | P8 owner | P8 config `RBP_BLEND.SPLIT_DATE` is wired; backtest runner MUST override it to `< backtest_start_date` at run-time. |
| B7 | Survivorship bias in universe (current S&P500 ∪ Nasdaq-100 snapshot, no historical constituents). | Out of scope here | Use FMP `historical-sp500-constituents` before any multi-year backtest. Surfaced for Team D2 (PIT audit). |

## 11. Falsification gate (post-merge)

Both P7 and P8 stay at capital weight 0.0 until **all** of these pass on a walk-forward purged-k-fold OOS backtest 2021–2026:

- **P7**: `Sharpe(P7) − Sharpe(P6) ≥ 0.15`, `MaxDD(P7) ≤ 1.10 · MaxDD(P6)`, `DSR(P7) ≥ 0.5`, no leakage canary failure.
- **P8**: `ΔIR ≥ +0.10` averaged across folds, `DSR ≥ 0.95`, RBI top-3 stable in ≥9/12 monthly refreshes.

Promotion = one-line edit in `portfolio_manager_config.json`. Demotion = same.

## 12. Inter-team dependencies

| Item | Will affect |
|---|---|
| P7/P8 weighting still uses P6's `inverse_vol_weights` | Team B (B1 HRP option must subclass cleanly) |
| Vol-target sits AFTER tilt in P7 — double-vol-targeting risk if master allocator also vol-targets | Team B (B3 vol-target audit) |
| P7 sentiment SQL filter uses strict `<` on `published_at` | Team D (D3 NLP look-ahead audit must verify ingest pipeline preserves PIT) |
| Univ snapshot is current, not PIT | Team D2 (PIT data audit) |
| Trend hedge ticker still empty in P6 config | Team C1 (must pick ticker) |
| P7 + P8 both extend `_rebalance` with inline logic duplicating P6 body | Future refactor: expose `_compose_stock_sleeve` as a hook on P6 |

---

End of Team A synthesis. Apply order in §4. All defaults preserve current Portfolio_6 behavior bit-exact.
