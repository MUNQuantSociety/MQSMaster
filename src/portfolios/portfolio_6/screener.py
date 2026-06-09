"""
Portfolio 6 helpers: screen + inverse-volatility weights + vol-target scaler +
Deflated Sharpe Ratio (Lopez de Prado 2014).

Price-only screen (boring + not-lottery) plus optional factor tilts from
local fundamentals CSV. Score composite supports two modes:
  - rank_sum   (default; bit-exact back-compat)
  - weighted_z (winsorized z-score sum; MSCI/QMJ-style)

New optional legs (all flag-gated, default OFF):
  - momentum 12-2 (Jegadeesh-Titman 1993; Daniel-Moskowitz 2016)
  - operating profitability (Ball-Gerakos-Linnainmaa-Nikolaev 2016 / FF RMW)
  - asset growth (Cooper-Gulen-Schill 2008 / FF CMA)

Hard-exclusion stage (Bali-Cakici-Whitelaw, Ang-Hodrick-Xing-Zhang, Kumar,
Ritter, Hou-Xue-Zhang) is provided by exclusions.apply_exclusions() and
wired through the exclusions_cfg kwarg.
"""

from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

try:
    from src.portfolios.portfolio_6.exclusions import apply_exclusions
except ImportError:
    from portfolios.portfolio_6.exclusions import apply_exclusions


TRADING_DAYS = 252


DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
    "vol": 1.0,
    "max_one_day": 1.0,
    "gross_profit_to_assets": 1.0,
    "momentum": 1.0,
    "op_profit": 1.0,
    "asset_growth": 1.0,
}


def realized_vol(returns: pd.Series, annualize: bool = True) -> float:
    if returns is None or returns.empty:
        return float("inf")
    sd = float(returns.std())
    if not np.isfinite(sd):
        return float("inf")
    return sd * (TRADING_DAYS ** 0.5) if annualize else sd


def max_one_day_return(returns: pd.Series) -> float:
    if returns is None or returns.empty:
        return float("inf")
    m = float(returns.max())
    return m if np.isfinite(m) else float("inf")


def momentum_12_2(
    returns: pd.Series,
    *,
    lookback_days: int = 252,
    skip_days: int = 21,
) -> float:
    """Jegadeesh-Titman 12-2 cumulative return (t-lookback to t-skip)."""
    if returns is None or returns.empty:
        return float("-inf")
    r = returns.dropna()
    if len(r) < lookback_days + 1:
        return float("-inf")
    window = r.iloc[-lookback_days:-skip_days] if skip_days > 0 else r.iloc[-lookback_days:]
    if window.empty:
        return float("-inf")
    return float((1.0 + window).prod() - 1.0)


def _safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    n = pd.to_numeric(numer, errors="coerce")
    d = pd.to_numeric(denom, errors="coerce")
    out = n / d
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def _winsorized_zscore(s: pd.Series, *, clip: float = 3.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        return pd.Series(dtype=float, index=s.index)
    sd = float(s.std(skipna=True))
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(0.0, index=s.index)
    return ((s - float(s.mean(skipna=True))) / sd).clip(lower=-clip, upper=clip)


def _median_fill(z: pd.Series) -> pd.Series:
    if z.dropna().empty:
        return z.fillna(0.0)
    med = float(z.median(skipna=True))
    return z.fillna(med if np.isfinite(med) else 0.0)


def score_universe(
    returns_matrix: Dict[str, pd.Series],
    fundamentals_df: Optional[pd.DataFrame] = None,
    *,
    method: str = "rank_sum",
    weights: Optional[Mapping[str, float]] = None,
    winsor_sigma: float = 3.0,
    exclusions_cfg: Optional[Dict] = None,
    use_fundamentals: bool = True,
    use_momentum: bool = False,
    use_op_profitability: bool = False,
    use_asset_growth: bool = False,
    momentum_lookback_days: int = 252,
    momentum_skip_days: int = 21,
    logger=None,
) -> pd.Series:
    """Composite scorer (lower = better).

    method='rank_sum'  (default): ascending rank-sum across active legs.
    method='weighted_z': cross-sectional z-score per leg, winsor at +/- winsor_sigma,
                        weighted sum, then negated so lower = better.
    """
    if not returns_matrix:
        return pd.Series(dtype=float)

    # A5: hard-exclusion stage (no-op when exclusions_cfg empty)
    if exclusions_cfg:
        excl_df = pd.DataFrame({"returns_21d": pd.Series(returns_matrix)})
        excl_df = apply_exclusions(excl_df, exclusions_cfg)
        returns_matrix = {
            t: returns_matrix[t] for t in excl_df.index if t in returns_matrix
        }
        if not returns_matrix:
            return pd.Series(dtype=float)

    vols = pd.Series({t: realized_vol(r) for t, r in returns_matrix.items()})
    maxs = pd.Series({t: max_one_day_return(r) for t, r in returns_matrix.items()})
    vols = vols.replace([np.inf, -np.inf], np.nan).dropna()
    maxs = maxs.replace([np.inf, -np.inf], np.nan).dropna()
    if vols.empty:
        return pd.Series(dtype=float)

    moms: Optional[pd.Series] = None
    if use_momentum:
        moms = pd.Series({
            t: momentum_12_2(r, lookback_days=momentum_lookback_days, skip_days=momentum_skip_days)
            for t, r in returns_matrix.items()
        })
        moms = moms.replace([np.inf, -np.inf], np.nan).dropna()

    gp_ratio: Optional[pd.Series] = None
    op_ratio: Optional[pd.Series] = None
    ag_series: Optional[pd.Series] = None
    if use_fundamentals and fundamentals_df is not None and not fundamentals_df.empty:
        cols = set(fundamentals_df.columns)
        if {"gross_profit", "total_assets"}.issubset(cols):
            gp_ratio = _safe_ratio(fundamentals_df["gross_profit"], fundamentals_df["total_assets"])
            gp_ratio = gp_ratio[gp_ratio.index.isin(vols.index)]
        if use_op_profitability and {"revenue", "cost_of_revenue", "sga", "interest_expense", "book_equity"}.issubset(cols):
            num = (
                pd.to_numeric(fundamentals_df["revenue"], errors="coerce")
                - pd.to_numeric(fundamentals_df["cost_of_revenue"], errors="coerce")
                - pd.to_numeric(fundamentals_df["sga"], errors="coerce")
                - pd.to_numeric(fundamentals_df["interest_expense"], errors="coerce")
            )
            op_ratio = _safe_ratio(num, fundamentals_df["book_equity"])
            op_ratio = op_ratio[op_ratio.index.isin(vols.index)]
        if use_asset_growth and "asset_growth" in cols:
            ag_series = pd.to_numeric(fundamentals_df["asset_growth"], errors="coerce")
            ag_series = ag_series.replace([np.inf, -np.inf], np.nan).dropna()
            ag_series = ag_series[ag_series.index.isin(vols.index)]

    w_map = dict(DEFAULT_SCORE_WEIGHTS)
    if weights:
        for k, v in weights.items():
            try:
                w_map[str(k)] = float(v)
            except (TypeError, ValueError):
                continue

    method_norm = (method or "rank_sum").lower().strip()

    # Mode 1: rank_sum (bit-exact back-compat when only vol+max+gp legs active)
    if method_norm == "rank_sum":
        score = vols.rank(ascending=True) * w_map["vol"]
        score = score.add(
            maxs.rank(ascending=True) * w_map["max_one_day"],
            fill_value=score.median(),
        )
        if gp_ratio is not None and not gp_ratio.empty:
            score = score.add(
                gp_ratio.rank(ascending=False) * w_map["gross_profit_to_assets"],
                fill_value=score.median(),
            )
        if moms is not None and not moms.empty:
            moms_aligned = moms[moms.index.isin(score.index)]
            score = score.add(
                moms_aligned.rank(ascending=False) * w_map["momentum"],
                fill_value=score.median(),
            )
        if op_ratio is not None and not op_ratio.empty:
            score = score.add(
                op_ratio.rank(ascending=False) * w_map["op_profit"],
                fill_value=score.median(),
            )
        if ag_series is not None and not ag_series.empty:
            score = score.add(
                ag_series.rank(ascending=True) * w_map["asset_growth"],
                fill_value=score.median(),
            )
        return score.sort_values(ascending=True)

    if method_norm != "weighted_z":
        raise ValueError(f"score_universe: unknown method={method!r}")

    # Mode 2: weighted_z (winsorized z-sum, negated so lower = better)
    idx = vols.index.union(maxs.index)
    for s in (gp_ratio, moms, op_ratio, ag_series):
        if s is not None:
            idx = idx.union(s.index)

    raw = {
        "vol":                    -vols.reindex(idx),
        "max_one_day":            -maxs.reindex(idx),
        "gross_profit_to_assets": gp_ratio.reindex(idx) if gp_ratio is not None else None,
        "momentum":               moms.reindex(idx) if moms is not None else None,
        "op_profit":              op_ratio.reindex(idx) if op_ratio is not None else None,
        "asset_growth":           -ag_series.reindex(idx) if ag_series is not None else None,
    }

    composite = pd.Series(0.0, index=idx)
    nan_counts: Dict[str, int] = {}
    weights_used: Dict[str, float] = {}
    for name, series in raw.items():
        if series is None:
            continue
        w = float(w_map.get(name, 0.0))
        if w == 0.0:
            continue
        z = _winsorized_zscore(series, clip=winsor_sigma)
        nan_counts[name] = int(z.isna().sum())
        z = _median_fill(z)
        composite = composite.add(w * z.reindex(idx), fill_value=0.0)
        weights_used[name] = w

    if logger is not None:
        try:
            logger.info(
                "[P6.score_universe] method=weighted_z n=%d weights=%s nan_fills=%s winsor=%.1f",
                int(composite.size), weights_used, nan_counts, float(winsor_sigma),
            )
        except Exception:
            pass

    return (-composite).sort_values(ascending=True)


def select_top_n(scores: pd.Series, n: int = 50) -> List[str]:
    if scores is None or scores.empty:
        return []
    return list(scores.iloc[: max(n, 0)].index)


def inverse_vol_weights(
    returns_matrix: Dict[str, pd.Series],
    *,
    max_weight: float = 0.05,
    max_iterations: int = 20,
) -> Dict[str, float]:
    """
    weight_i = (1/vol_i) / sum(1/vol_j), iteratively capped at max_weight with
    slack redistributed to under-cap names. Returns dict ticker -> weight.
    """
    inv = {}
    for t, r in returns_matrix.items():
        v = realized_vol(r)
        if v > 0 and np.isfinite(v):
            inv[t] = 1.0 / v
    if not inv:
        return {}
    s = pd.Series(inv)
    w = s / s.sum()

    for _ in range(max_iterations):
        over = w > max_weight
        if not over.any():
            break
        slack = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        under_sum = float(w[under].sum())
        if under_sum <= 0:
            break
        w[under] = w[under] + slack * (w[under] / under_sum)

    return w.to_dict()


def vol_target_scale(
    portfolio_returns: pd.Series,
    *,
    target_annual_vol: float = 0.13,
    max_scale: float = 1.5,
) -> float:
    """
    Single-layer sleeve volatility scaler. Returns
        k = min(target_annual_vol / sigma_realized_ann, max_scale).

    Contract (B3 audit, 2026-05-20):
      - This is the ONLY vol-target layer in MQSMaster. The capital
        allocator (src/risk_manager/daily_allocator.py) does NOT vol-target;
        it allocates by fixed dollar weight from portfolio_manager_config.json.
      - Adding a second vol-target layer (e.g. inside DailyAllocator) would
        compound multiplicatively and silently collapse realized vol below
        target. See .claude/agents-output/teamB/B3_vol_target_audit.md.
      - References: Moreira-Muir 2017 (JoF 72:1611), Harvey et al. 2018
        (JPM 45:14), Asness-Frazzini-Pedersen 2012 (FAJ 68:47).
    """
    if portfolio_returns is None or portfolio_returns.empty:
        return 1.0
    sd = float(portfolio_returns.std())
    if not np.isfinite(sd) or sd <= 0:
        return 1.0
    realized_ann = sd * (TRADING_DAYS ** 0.5)
    if realized_ann <= 0:
        return 1.0
    return float(min(target_annual_vol / realized_ann, max_scale))


def deflated_sharpe_ratio(
    daily_returns: pd.Series,
    *,
    n_trials: int,
    benchmark_sr_annual: float = 0.0,
) -> float:
    """
    Lopez de Prado (2014). Returns probability that the true (annualized) Sharpe
    exceeds the benchmark, accounting for multiple-testing inflation from selecting
    the best of n_trials candidates.
    """
    try:
        from scipy.stats import norm
    except ImportError:
        return float("nan")

    if daily_returns is None:
        return 0.0
    r = daily_returns.dropna()
    if r.empty or len(r) < 30 or n_trials < 1:
        return 0.0
    n = len(r)
    mu = float(r.mean())
    sigma = float(r.std())
    if sigma <= 0 or not np.isfinite(sigma):
        return 0.0

    sr_daily = mu / sigma
    skew = float(r.skew()) if n > 3 else 0.0
    kurt_excess = float(r.kurtosis()) if n > 3 else 0.0

    var_sr = (1 - skew * sr_daily + (kurt_excess / 4.0) * sr_daily ** 2) / max(n - 1, 1)
    if var_sr <= 0:
        return 0.0
    sigma_sr = float(np.sqrt(var_sr))

    gamma = 0.5772156649015329
    z1 = norm.ppf(1 - 1.0 / max(n_trials, 2))
    z2 = norm.ppf(1 - 1.0 / (max(n_trials, 2) * np.e))
    sr_threshold_daily = sigma_sr * ((1 - gamma) * z1 + gamma * z2)

    benchmark_sr_daily = benchmark_sr_annual / (TRADING_DAYS ** 0.5)
    z = (sr_daily - sr_threshold_daily - benchmark_sr_daily) / sigma_sr
    return float(norm.cdf(z))
