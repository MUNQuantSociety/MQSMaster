"""
Combinatorially Symmetric Cross-Validation (CSCV) and Probability of Backtest
Overfitting (PBO) -- Bailey, Borwein, Lopez de Prado, Zhu 2014.

Reference: "The Probability of Backtest Overfitting", Journal of Portfolio
Management, 40(5), 94-107. SSRN 2326253.

Public API:
    cscv_pbo(returns_matrix_per_strategy, *, S=16, metric_func=None,
             threshold=0.0) -> dict
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


def _annualized_sharpe(returns: np.ndarray, *, periods_per_year: int = 252) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    sd = float(arr.std(ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return 0.0
    mu = float(arr.mean())
    return float(mu / sd * np.sqrt(periods_per_year))


def _coerce_matrix(returns_matrix_per_strategy):
    if isinstance(returns_matrix_per_strategy, pd.DataFrame):
        labels = [str(c) for c in returns_matrix_per_strategy.columns]
        arr = returns_matrix_per_strategy.to_numpy(dtype=float, copy=True)
    else:
        arr = np.asarray(returns_matrix_per_strategy, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        labels = [f"strategy_{i}" for i in range(arr.shape[1])]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, labels


def _rank_array(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import rankdata
        return rankdata(x, method="average")
    except ImportError:
        order = np.argsort(np.argsort(x))
        return order.astype(float) + 1.0


def _safe_logit(p: float, eps: float = 1e-12) -> float:
    p = float(min(max(p, eps), 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def cscv_pbo(
    returns_matrix_per_strategy,
    *,
    S: int = 16,
    metric_func: Callable[[np.ndarray], float] | None = None,
    threshold: float = 0.0,
) -> dict[str, object]:
    """Probability of Backtest Overfitting via CSCV (Bailey et al. 2014)."""
    if S % 2 != 0:
        raise ValueError(f"S must be even (got S={S}).")
    if S < 4:
        raise ValueError(f"S must be >= 4 to extract any PBO signal (got S={S}).")

    metric = metric_func or _annualized_sharpe

    m, labels = _coerce_matrix(returns_matrix_per_strategy)
    T_raw, N = m.shape
    if N < 4:
        raise ValueError(
            f"CSCV needs N >= 4 distinct strategies (got N={N}); "
            "expand by sweeping SCORE_WEIGHTS / WEIGHTING_METHOD permutations."
        )

    block_size = T_raw // S
    if block_size < 2:
        raise ValueError(
            f"T={T_raw} too small relative to S={S}; "
            f"need at least 2*S = {2 * S} periods."
        )
    T = block_size * S
    m = m[:T]

    block_ids = np.arange(S)
    block_rows: list[np.ndarray] = [
        np.arange(s * block_size, (s + 1) * block_size) for s in block_ids
    ]

    half = S // 2
    splits = list(combinations(block_ids.tolist(), half))
    n_splits = len(splits)

    logits: list[float] = []
    is_best_metric: list[float] = []
    oos_best_metric: list[float] = []
    oos_median_metric: list[float] = []

    for train_blocks in splits:
        train_set = set(train_blocks)
        test_blocks = tuple(b for b in block_ids.tolist() if b not in train_set)

        train_rows = np.concatenate([block_rows[b] for b in sorted(train_blocks)])
        test_rows = np.concatenate([block_rows[b] for b in sorted(test_blocks)])

        J = m[train_rows]
        J_hat = m[test_rows]

        R_is = np.array([metric(J[:, n]) for n in range(N)], dtype=float)
        R_oos = np.array([metric(J_hat[:, n]) for n in range(N)], dtype=float)

        rank_oos = _rank_array(R_oos)
        n_star = int(np.argmax(R_is))

        r_hat_star = float(rank_oos[n_star])
        omega_bar = r_hat_star / (float(N) + 1.0)
        logits.append(_safe_logit(omega_bar))

        is_best_metric.append(float(R_is[n_star]))
        oos_best_metric.append(float(R_oos[n_star]))
        oos_median_metric.append(float(np.median(R_oos)))

    logits_np = np.asarray(logits, dtype=float)
    pbo = float(np.mean(logits_np <= 0.0))

    is_arr = np.asarray(is_best_metric, dtype=float)
    oos_arr = np.asarray(oos_best_metric, dtype=float)
    if is_arr.size >= 2 and np.isfinite(is_arr).all() and np.isfinite(oos_arr).all():
        slope, intercept = np.polyfit(is_arr, oos_arr, 1)
        ss_tot = float(((oos_arr - oos_arr.mean()) ** 2).sum())
        ss_res = float(((oos_arr - (intercept + slope * is_arr)) ** 2).sum())
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        slope, intercept, r_squared = 0.0, 0.0, 0.0

    oos_arr_sorted = np.sort(oos_arr)
    median_arr_sorted = np.sort(np.asarray(oos_median_metric, dtype=float))
    cdf_x = np.arange(1, n_splits + 1, dtype=float) / float(n_splits)

    prob_oos_loss = float(np.mean(oos_arr < threshold))

    return {
        "pbo": pbo,
        "performance_degradation": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
        },
        "stochastic_dominance": {
            "cdf_probabilities": cdf_x.tolist(),
            "oos_best_sorted": oos_arr_sorted.tolist(),
            "oos_median_sorted": median_arr_sorted.tolist(),
        },
        "prob_oos_loss": prob_oos_loss,
        "n_combinations": int(n_splits),
        "n_strategies": int(N),
        "n_periods": int(T),
        "block_size": int(block_size),
        "logits": logits_np.tolist(),
        "strategy_labels": labels,
        "threshold": float(threshold),
        "S": int(S),
    }


def build_strategy_grid_matrix(
    series_per_strategy: Dict[str, pd.Series],
    *,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Build (T, N) returns matrix from {label: returns_series}."""
    if not series_per_strategy:
        return pd.DataFrame()
    aligned = pd.DataFrame(series_per_strategy)
    aligned = aligned.sort_index()
    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    aligned = aligned.fillna(fill_value)
    return aligned


__all__ = [
    "cscv_pbo",
    "build_strategy_grid_matrix",
]
