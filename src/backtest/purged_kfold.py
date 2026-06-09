"""
Purged K-Fold Cross-Validation with embargo -- Lopez de Prado, AFML 2018,
Chapter 7 (Snippets 7.1, 7.2, 7.3).

Public API:
    PurgedKFold(n_splits, t1, embargo_td)

Use case in MQSMaster:
    - RBP forecast labels (portfolio_5, portfolio_8) where ``t1[i]`` is the end
      of the forecast horizon for sample i.
    - Any ML model whose label spans multiple days.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd


class PurgedKFold:
    """K-fold cross-validation with purging and embargo (AFML §7)."""

    def __init__(
        self,
        n_splits: int = 5,
        t1: pd.Series = None,
        embargo_td: str = "pd.Timedelta | float | int",
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2 (got {n_splits}).")
        if t1 is None:
            raise ValueError(
                "t1 must be a pd.Series of label-end times, indexed by sample."
            )
        if not t1.index.is_monotonic_increasing:
            raise ValueError("t1.index must be monotonically increasing (sort first).")

        if isinstance(embargo_td, (int, float)):
            if 0.0 <= float(embargo_td) <= 1.0:
                self._embargo_pct: float = float(embargo_td)
                self._embargo_td: pd.Timedelta | None = None
            else:
                raise ValueError("Numeric embargo_td must be a fraction in [0, 1].")
        elif isinstance(embargo_td, pd.Timedelta):
            self._embargo_pct = None
            self._embargo_td = embargo_td
        else:
            raise ValueError("embargo_td must be a pd.Timedelta or fraction in [0, 1].")

        self.n_splits: int = int(n_splits)
        self.t1: pd.Series = t1.copy()

    def get_n_splits(self,
        _X: None = None,
        _y: None = None,
        _groups: None = None
    ) -> int:
        return self.n_splits

    def split(self,
        X,
        _y = None,
        _groups = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = self._infer_n_samples(X)
        if len(self.t1) != n:
            raise ValueError(
                f"t1 length ({len(self.t1)}) != X length ({n}); " +
                "t1 must be aligned 1-to-1 with X."
            )

        indices = np.arange(n)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1

        starts = np.cumsum(np.concatenate([[0], fold_sizes[:-1]]))
        stops = np.cumsum(fold_sizes)

        if self._embargo_pct is not None:
            h_pos = int(np.ceil(self._embargo_pct * n))
        else:
            h_pos = None

        feature_times = self.t1.index
        label_times = pd.Series(self.t1.values, index=feature_times)

        for fold in range(self.n_splits):
            test_start_idx = int(starts[fold])
            test_stop_idx = int(stops[fold])
            test_idx = indices[test_start_idx:test_stop_idx]

            test_feature_start = feature_times[test_start_idx]
            test_label_end = label_times.iloc[test_start_idx:test_stop_idx].max()

            overlap_mask = (
                (feature_times <= test_label_end)
                & (label_times.values >= test_feature_start)
            )
            purged_positions = set(np.where(overlap_mask)[0].tolist())

            embargo_positions: set = set()
            if self._embargo_td is not None:
                emb_cut = test_label_end + self._embargo_td
                emb_mask = (
                    (feature_times > test_label_end) & (feature_times <= emb_cut)
                )
                embargo_positions = set(np.where(emb_mask)[0].tolist())
            elif h_pos is not None and h_pos > 0:
                start_emb = test_stop_idx
                end_emb = min(test_stop_idx + h_pos, n)
                embargo_positions = set(range(start_emb, end_emb))

            drop = set(test_idx.tolist()) | purged_positions | embargo_positions
            train_idx = np.array(
                [i for i in indices if i not in drop],
                dtype=int,
            )
            if train_idx.size == 0:
                continue
            yield train_idx, test_idx

    @staticmethod
    def _infer_n_samples(X) -> int:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return len(X)
        arr = np.asarray(X)
        if arr.ndim == 0:
            raise ValueError("X must be array-like with at least 1 sample.")
        return arr.shape[0]


def t1_from_horizon(
    feature_times: Sequence[Any],
    horizon: pd.Timedelta,
) -> pd.Series:
    """Build a t1 Series for a fixed-horizon label (e.g. h=21d for RBP)."""
    idx = pd.DatetimeIndex(feature_times)
    return pd.Series(idx + horizon, index=idx)


__all__ = ["PurgedKFold", "t1_from_horizon"]
