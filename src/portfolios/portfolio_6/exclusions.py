"""
Portfolio 6 hard-exclusion stage. Run *before* ``score_universe``.

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
                 daily FF3 factors; fall back to residuals against the
                 cross-sectional equal-weight mean return (CSEW).
  - Penny      : Kumar (2009), J. Finance 64 -- drop names with last close < $5.
  - Microcap   : Hou, Xue, Zhang (2020), RFS 33 -- drop names below market-cap
                 floor (default $300M, ~Russell 2000 microcap line).
  - Recent IPO : Ritter (1991), J. Finance 46 -- drop names listed < N months.
  - PEAD/Short : stubs; require new FMP fetch columns.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def _series_or_none(value: Any) -> Optional[pd.Series]:
    if isinstance(value, pd.Series) and not value.empty:
        return value
    return None


def _max_1d_return(returns_21d: pd.Series, window: int = 21) -> float:
    s = _series_or_none(returns_21d)
    if s is None:
        return np.nan
    tail = s.tail(window).dropna()
    if tail.empty:
        return np.nan
    return float(tail.max())


def _max_n_avg_return(returns_21d: pd.Series, n: int = 5, window: int = 21) -> float:
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
    """Fallback IVOL: residual std after OLS on cross-sectional equal-weight mean."""
    if not returns_by_ticker:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(
        {t: pd.Series(r).tail(window) for t, r in returns_by_ticker.items()}
    ).dropna(how="all")
    if frame.empty or len(frame) < min_obs:
        return pd.Series(dtype=float)
    csew = frame.mean(axis=1)
    if float(csew.var()) <= 0:
        return pd.Series(dtype=float)

    out: Dict[str, float] = {}
    for ticker in frame.columns:
        r = frame[ticker].dropna()
        f = csew.reindex(r.index).dropna()
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
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty or pct <= 0:
        return []
    threshold = s.quantile(1.0 - pct)
    return s[s >= threshold].index.tolist()


def _rule_max(universe_df: pd.DataFrame, rule_cfg: dict) -> List[str]:
    if not rule_cfg.get("ENABLED", False) or "returns_21d" not in universe_df.columns:
        return []
    window = int(rule_cfg.get("WINDOW_DAYS", 21))
    pct = float(rule_cfg.get("DROP_TOP_PCT", 0.20))
    max_series = universe_df["returns_21d"].apply(lambda r: _max_1d_return(r, window=window))
    drops = _drop_top_pct(max_series, pct)
    logger.info("[exclusions.MAX] dropped %d (window=%d, top_pct=%.2f)", len(drops), window, pct)
    return drops


def _rule_max_n(universe_df: pd.DataFrame, rule_cfg: dict) -> List[str]:
    if not rule_cfg.get("ENABLED", False) or "returns_21d" not in universe_df.columns:
        return []
    window = int(rule_cfg.get("WINDOW_DAYS", 21))
    n = int(rule_cfg.get("N", 5))
    pct = float(rule_cfg.get("DROP_TOP_PCT", 0.20))
    maxn = universe_df["returns_21d"].apply(lambda r: _max_n_avg_return(r, n=n, window=window))
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
    """Apply hard exclusions and return surviving subset of ``universe_df``."""
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
        except Exception as e:
            logger.exception("[exclusions.%s] rule errored: %s", rule_name, e)
            continue
        drops.update(rule_drops)

    if not drops:
        return universe_df

    survivors = universe_df.index.difference(pd.Index(drops))
    logger.info(
        "[exclusions] total dropped=%d, survivors=%d (started=%d)",
        len(drops), len(survivors), len(universe_df),
    )
    return universe_df.loc[survivors]
