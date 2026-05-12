"""Reliability metrics (fit, asymmetry, adjusted fit) and the GridCell record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class FitCalculator:
    """Computes the reliability metrics described in paper Eq. 11-14."""

    def fit(self, weights: pd.Series, outcomes: pd.Series) -> float:
        weights_aligned, outcomes_aligned = weights.align(outcomes)
        if weights_aligned.std() == 0 or outcomes_aligned.std() == 0:
            return 0.0
        correlation = np.corrcoef(weights_aligned, outcomes_aligned)[0, 1]
        return 0.0 if np.isnan(correlation) else float(correlation ** 2)

    def asymmetry(
        self,
        weights: pd.Series,
        outcomes: pd.Series,
        retained_mask: pd.Series,
    ) -> float:
        w_aligned, o_aligned = weights.align(outcomes)
        w_aligned, m_aligned = w_aligned.align(retained_mask)
        o_aligned, m_aligned = o_aligned.align(m_aligned)

        rho_plus = self._safe_correlation(w_aligned[m_aligned], o_aligned[m_aligned])
        rho_minus = self._safe_correlation(w_aligned[~m_aligned], o_aligned[~m_aligned])
        return 0.5 * (rho_plus - rho_minus) ** 2

    @staticmethod
    def adjusted_fit(fit: float, asymmetry: float, num_features: int) -> float:
        return num_features * (fit + asymmetry)

    @staticmethod
    def _safe_correlation(x: pd.Series, y: pd.Series) -> float:
        if len(x) < 2 or x.std() == 0 or y.std() == 0:
            return 0.0
        correlation = np.corrcoef(x, y)[0, 1]
        return 0.0 if np.isnan(correlation) else float(correlation)


@dataclass
class GridCell:
    """One cell in the RBP prediction grid: a (feature subset, censoring quantile) pair."""

    feature_names: Tuple[str, ...]
    censoring_quantile: float
    prediction: Optional[float] = None
    adjusted_fit: Optional[float] = None

    @property
    def num_features(self) -> int:
        return len(self.feature_names)
