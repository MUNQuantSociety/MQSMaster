"""
Portfolio 6 -- Hierarchical Risk Parity weighting (Lopez de Prado 2016).

Adds an opt-in alternative to ``inverse_vol_weights``. Wired via
``PORTFOLIO_6_CONFIG.WEIGHTING_METHOD`` ("INV_VOL" | "HRP" | "ERC").

Algorithm reference: Marcos Lopez de Prado, "Building Diversified
Portfolios that Outperform Out-of-Sample," J. Portfolio Management
42(4):59-69, 2016 (SSRN 2708678).

Three steps:
  1. correlation -> distance  d(i,j) = sqrt(0.5 * (1 - rho_ij))
  2. quasi-diagonalization    leaf-order traversal of single-linkage dendrogram
  3. recursive bisection      alpha = 1 - V_L / (V_L + V_R)  with V_C = inv-var
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

logger = logging.getLogger(__name__)

DEFAULT_LINKAGE_METHOD = "single"


def _returns_to_dataframe(returns_matrix: Dict[str, pd.Series]) -> pd.DataFrame:
    if not returns_matrix:
        return pd.DataFrame()
    df = pd.DataFrame({t: pd.Series(r).astype(float) for t, r in returns_matrix.items()})
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all")
    return df


def _correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    c = corr.clip(lower=-1.0, upper=1.0).fillna(0.0)
    d = np.sqrt(np.maximum(0.5 * (1.0 - c.values), 0.0))
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)


def _get_quasi_diag(link: np.ndarray) -> list:
    Z = link.astype(int)
    num_items = int(Z[-1, 3])
    sort_ix = [int(Z[-1, 0]), int(Z[-1, 1])]
    while max(sort_ix) >= num_items:
        new = []
        for v in sort_ix:
            if v < num_items:
                new.append(int(v))
            else:
                row = Z[v - num_items]
                new.extend([int(row[0]), int(row[1])])
        sort_ix = new
    return sort_ix


def _ivp_weights(cov_sub: np.ndarray) -> np.ndarray:
    diag = np.diag(cov_sub).astype(float)
    diag = np.where(diag > 0.0, diag, np.finfo(float).tiny)
    inv = 1.0 / diag
    s = inv.sum()
    if not np.isfinite(s) or s <= 0.0:
        return np.full_like(diag, 1.0 / len(diag))
    return inv / s


def _cluster_variance(cov: pd.DataFrame, items: list) -> float:
    sub = cov.loc[items, items].to_numpy()
    w = _ivp_weights(sub).reshape(-1, 1)
    v = float((w.T @ sub @ w)[0, 0])
    return v if np.isfinite(v) and v > 0.0 else float(np.finfo(float).tiny)


def _recursive_bisection(cov: pd.DataFrame, ordered: list) -> pd.Series:
    w = pd.Series(1.0, index=ordered, dtype=float)
    clusters = [ordered]
    while clusters:
        clusters = [
            c[start:stop]
            for c in clusters
            for start, stop in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            vL = _cluster_variance(cov, left)
            vR = _cluster_variance(cov, right)
            alpha = 1.0 - vL / (vL + vR)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            w[left] *= alpha
            w[right] *= 1.0 - alpha
    return w


def _apply_iterative_cap(
    w: pd.Series,
    *,
    max_weight: float,
    max_iterations: int = 20,
) -> pd.Series:
    if max_weight <= 0.0:
        return w
    w = w.copy().astype(float)
    for _ in range(max_iterations):
        over = w > max_weight
        if not over.any():
            break
        slack = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        under_sum = float(w[under].sum())
        if under_sum <= 0.0:
            break
        w[under] = w[under] + slack * (w[under] / under_sum)
    return w


def hrp_weights(
    returns_matrix: Dict[str, pd.Series],
    *,
    max_weight: float = 0.05,
    cov: Optional[pd.DataFrame] = None,
    linkage_method: str = DEFAULT_LINKAGE_METHOD,
    max_iterations: int = 20,
) -> Dict[str, float]:
    """Hierarchical Risk Parity weights."""
    if not _SCIPY_OK:
        logger.error("[HRP] scipy not available; cannot compute HRP weights.")
        return {}

    rets_df = _returns_to_dataframe(returns_matrix)
    if rets_df.empty:
        return {}
    if rets_df.shape[1] == 1:
        return {str(rets_df.columns[0]): 1.0}

    if cov is None:
        sigma = rets_df.cov()
    else:
        common = [c for c in rets_df.columns if c in cov.index and c in cov.columns]
        if len(common) < 2:
            logger.warning(
                "[HRP] Provided cov covers <2 of the input assets; falling back to sample cov."
            )
            sigma = rets_df.cov()
        else:
            sigma = cov.loc[common, common].astype(float)
            rets_df = rets_df[common]

    sigma = sigma.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    diag = np.diag(sigma.to_numpy())
    keep = [t for t, v in zip(sigma.index, diag) if np.isfinite(v) and v > 0.0]
    if len(keep) < 2:
        if len(keep) == 1:
            return {keep[0]: 1.0}
        return {}
    sigma = sigma.loc[keep, keep]
    rets_df = rets_df[keep]

    corr = rets_df.corr().reindex(index=keep, columns=keep)
    dist = _correlation_distance(corr)

    try:
        condensed = squareform(dist.to_numpy(), checks=False)
        link = linkage(condensed, method=linkage_method)
    except Exception as e:
        logger.exception("[HRP] linkage failed (%s); returning IVP fallback.", e)
        ivp = _ivp_weights(sigma.to_numpy())
        w = pd.Series(ivp, index=keep)
        w = _apply_iterative_cap(w, max_weight=max_weight, max_iterations=max_iterations)
        return {str(k): float(v) for k, v in w.items()}

    order_ix = _get_quasi_diag(link)
    ordered_tickers = [keep[i] for i in order_ix]

    w = _recursive_bisection(sigma.loc[ordered_tickers, ordered_tickers], ordered_tickers)
    w = w.reindex(keep).fillna(0.0)
    total = float(w.sum())
    if total > 0.0:
        w = w / total

    w = _apply_iterative_cap(w, max_weight=max_weight, max_iterations=max_iterations)
    return {str(k): float(v) for k, v in w.items()}


def erc_weights(
    returns_matrix: Dict[str, pd.Series],
    *,
    max_weight: float = 0.05,
    cov: Optional[pd.DataFrame] = None,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> Dict[str, float]:
    """Equal-Risk-Contribution weights (Maillard-Roncalli-Teiletche 2010).

    TODO: full cyclic-coordinate-descent per Spinu 2013. Stub returns IVP
    (the diagonal-Sigma limit of ERC). DO NOT promote WEIGHTING_METHOD='ERC'
    to default until this is replaced.
    """
    logger.warning(
        "[ERC] erc_weights stub: returning IVP (diagonal-Sigma limit of ERC). "
        "Replace with full Spinu 2013 cyclic-coordinate-descent before "
        "promoting WEIGHTING_METHOD='ERC' to default."
    )
    rets_df = _returns_to_dataframe(returns_matrix)
    if rets_df.empty:
        return {}
    if cov is None:
        sigma = rets_df.cov().to_numpy()
    else:
        common = [c for c in rets_df.columns if c in cov.index and c in cov.columns]
        sigma = cov.loc[common, common].to_numpy() if len(common) >= 2 else rets_df.cov().to_numpy()
    ivp = _ivp_weights(sigma)
    w = pd.Series(ivp, index=rets_df.columns[: len(ivp)])
    w = _apply_iterative_cap(w, max_weight=max_weight, max_iterations=20)
    return {str(k): float(v) for k, v in w.items()}
