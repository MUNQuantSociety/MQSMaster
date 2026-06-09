"""
Covariance estimation utility for MQSMaster.

Public API: shrink_cov(returns_df, method="ledoit_wolf_cc", *, annualize=False,
            min_periods=60, min_overlap_frac=0.5, assume_centered=False)
    -> pd.DataFrame (n_assets x n_assets), symmetric, PSD.

Methods:
  "sample"               -- plain ddof=1 sample covariance (numpy).
  "ledoit_wolf_cc"       -- Ledoit-Wolf 2004 constant-correlation shrinkage
                           (hand-coded, numpy-only). DEFAULT.
  "ledoit_wolf_identity" -- sklearn.covariance.LedoitWolf (identity target).
  "oas"                  -- sklearn.covariance.OAS.

References:
  Ledoit, O. & Wolf, M. (2004) "Honey, I Shrunk the Sample Covariance Matrix",
    Journal of Portfolio Management 30(4): 110-119.
  Ledoit, O. & Wolf, M. (2003) "Improved estimation ... single-factor target",
    Journal of Empirical Finance 10(5): 603-621.
  Chen, Wiesel, Eldar, Hero (2010) "Shrinkage algorithms for MMSE covariance
    estimation", IEEE Trans. Signal Processing 58(10): 5016-5029.
  Lopez de Prado (2016) "Building Diversified Portfolios that Outperform OOS",
    Journal of Portfolio Management 42(4): 59-69.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


TRADING_DAYS = 252
_SUPPORTED_METHODS = ("sample", "ledoit_wolf_cc", "ledoit_wolf_identity", "oas")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def shrink_cov(
    returns_df: pd.DataFrame,
    method: str = "ledoit_wolf_cc",
    *,
    annualize: bool = False,
    min_periods: int = 60,
    min_overlap_frac: float = 0.5,
    assume_centered: bool = False,
) -> pd.DataFrame:
    """
    Estimate an asset return covariance matrix with optional shrinkage.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Wide-format daily (or any equi-spaced) returns: rows = observations,
        columns = tickers. NaN allowed; see ``min_periods`` / ``min_overlap_frac``.
    method : {"sample", "ledoit_wolf_cc", "ledoit_wolf_identity", "oas"}
        Estimator to use. Default ``"ledoit_wolf_cc"`` (Ledoit-Wolf 2004
        constant correlation target -- best for equity universes with N <= ~225).
    annualize : bool, default False
        If True, multiply the returned covariance by ``TRADING_DAYS=252``.
    min_periods : int, default 60
        Drop any column with fewer than this many non-NaN observations.
    min_overlap_frac : float, default 0.5
        Joint-overlap fraction across columns. Defends against the "two columns
        share no dates" pathology that breaks plain sample covariance.
    assume_centered : bool, default False
        If True, do NOT subtract the column mean before estimation.

    Returns
    -------
    pd.DataFrame
        ``(n_assets x n_assets)`` covariance matrix indexed by ticker.
        Symmetric, PSD, well-conditioned when a shrinkage method is selected.
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"shrink_cov: unknown method={method!r}. "
            f"Supported: {_SUPPORTED_METHODS}"
        )
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError(f"shrink_cov: returns_df must be a DataFrame, got {type(returns_df)}")
    if returns_df.empty:
        raise ValueError("shrink_cov: returns_df is empty.")

    R = returns_df.copy()
    R = R.replace([np.inf, -np.inf], np.nan)
    keep_cols = [c for c in R.columns if int(R[c].notna().sum()) >= min_periods]
    if len(keep_cols) < 2:
        raise ValueError(
            f"shrink_cov: only {len(keep_cols)} columns survive min_periods="
            f"{min_periods}; need >= 2."
        )
    R = R[keep_cols]

    obs_per_row = R.notna().sum(axis=1)
    thresh = max(2, int(np.ceil(min_overlap_frac * R.shape[1])))
    R = R.loc[obs_per_row >= thresh]
    if R.shape[0] < min_periods:
        raise ValueError(
            f"shrink_cov: only {R.shape[0]} rows survive min_overlap_frac="
            f"{min_overlap_frac}; need >= min_periods={min_periods}."
        )
    R = R.dropna(axis=0, how="any")
    if R.shape[0] < min_periods or R.shape[1] < 2:
        raise ValueError(
            "shrink_cov: insufficient overlap after dropna. Got "
            f"shape={R.shape}, need rows >= {min_periods} and cols >= 2."
        )

    tickers = list(R.columns)
    X = R.to_numpy(dtype=np.float64, copy=True)
    if not assume_centered:
        X = X - X.mean(axis=0, keepdims=True)

    if method == "sample":
        Sigma = _cov_sample(X)
    elif method == "ledoit_wolf_cc":
        Sigma, _, _ = _cov_ledoit_wolf_cc(X)
    elif method == "ledoit_wolf_identity":
        Sigma, _ = _cov_sklearn_lw(X)
    elif method == "oas":
        Sigma, _ = _cov_sklearn_oas(X)
    else:
        raise ValueError(method)

    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma = _psd_clip(Sigma)

    if annualize:
        Sigma = Sigma * TRADING_DAYS

    return pd.DataFrame(Sigma, index=tickers, columns=tickers)


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------


def _cov_sample(X: np.ndarray) -> np.ndarray:
    T = X.shape[0]
    if T < 2:
        raise ValueError("_cov_sample: need T >= 2 rows.")
    return (X.T @ X) / (T - 1)


def _cov_ledoit_wolf_cc(X: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Ledoit-Wolf 2004 shrinkage toward constant-correlation target.

    Returns (Sigma, average_correlation, delta_star).
    """
    T, N = X.shape
    if T < 2 or N < 2:
        raise ValueError(f"_cov_ledoit_wolf_cc: need T,N >= 2, got T={T},N={N}.")

    S = (X.T @ X) / float(T)

    var = np.diag(S).reshape(-1, 1)
    sqrt_var = np.sqrt(np.maximum(var, 0.0))
    unit_cor_var = sqrt_var @ sqrt_var.T

    eps = 1e-30
    with np.errstate(divide="ignore", invalid="ignore"):
        sample_cor = np.where(unit_cor_var > eps, S / unit_cor_var, 0.0)
    r_bar = (sample_cor.sum() - float(N)) / float(N * (N - 1))

    F = r_bar * unit_cor_var
    np.fill_diagonal(F, var.ravel())

    XX = X * X
    phi_mat = (XX.T @ XX) / float(T) - S * S
    pi_hat = float(phi_mat.sum())

    X3 = X ** 3
    theta_mat = (X3.T @ X) / float(T) - var * S
    np.fill_diagonal(theta_mat, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_sqrt_var = np.where(sqrt_var > eps, 1.0 / sqrt_var, 0.0)
    cross = inv_sqrt_var @ sqrt_var.T
    rho_hat = float(np.diag(phi_mat).sum() + r_bar * (cross * theta_mat).sum())

    diff = F - S
    gamma_hat = float((diff * diff).sum())

    if gamma_hat <= 0.0:
        delta_star = 0.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta_star = float(max(0.0, min(1.0, kappa / float(T))))

    Sigma = delta_star * F + (1.0 - delta_star) * S
    return Sigma, float(r_bar), delta_star


def _cov_sklearn_lw(X: np.ndarray) -> Tuple[np.ndarray, float]:
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf(assume_centered=True, store_precision=False).fit(X)
    return np.asarray(lw.covariance_, dtype=np.float64), float(lw.shrinkage_)


def _cov_sklearn_oas(X: np.ndarray) -> Tuple[np.ndarray, float]:
    from sklearn.covariance import OAS

    oas = OAS(assume_centered=True, store_precision=False).fit(X)
    return np.asarray(oas.covariance_, dtype=np.float64), float(oas.shrinkage_)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _psd_clip(M: np.ndarray, *, floor: float = 0.0) -> np.ndarray:
    w, V = np.linalg.eigh(M)
    if w.min() >= floor and np.isfinite(w).all():
        return M
    w_clipped = np.clip(w, floor, None)
    return (V * w_clipped) @ V.T


def condition_number(Sigma, *, floor: float = 1e-30) -> float:
    """Reporting helper: kappa(Sigma) = lambda_max / max(lambda_min, floor)."""
    arr = np.asarray(Sigma.values if isinstance(Sigma, pd.DataFrame) else Sigma, dtype=np.float64)
    arr = 0.5 * (arr + arr.T)
    w = np.linalg.eigvalsh(arr)
    return float(w.max() / max(w.min(), floor))
