"""Convert relevance scores into observation weights (paper Eq. 6-9)."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PredictionWeightCalculator:
    """Maps relevance scores to weights, optionally censoring low-relevance rows."""

    def calculate(
        self, relevance_scores: pd.Series, censoring_quantile: float = 0.0
    ) -> Tuple[pd.Series, pd.Series]:
        n = len(relevance_scores)
        index = relevance_scores.index

        if n < 2:
            logger.warning("Not enough observations to calculate weights.")
            return pd.Series(np.nan, index=index), pd.Series(False, index=index)

        if censoring_quantile == 0.0:
            return self._linear_weights(relevance_scores), pd.Series(True, index=index)

        threshold = relevance_scores.quantile(censoring_quantile)
        retained_mask = relevance_scores >= threshold
        n_retained = int(retained_mask.sum())

        if n_retained < 2:
            logger.warning(
                "Censoring quantile %s left < 2 rows; falling back to linear weights.",
                censoring_quantile,
            )
            return self._linear_weights(relevance_scores), pd.Series(True, index=index)

        retention_rate = n_retained / n
        retained_scores = relevance_scores[retained_mask]
        mean_retained = retained_scores.mean()

        var_full = (relevance_scores ** 2).sum() / (n - 1)
        var_retained = (retained_scores ** 2).sum() / (n_retained - 1)
        lambda_squared = 1.0 if var_retained == 0 else var_full / var_retained

        delta_relevance = relevance_scores.where(retained_mask, 0.0)
        weights = (
            1 / n
            + (lambda_squared / (n_retained - 1))
            * (delta_relevance - retention_rate * mean_retained)
        )
        return weights, retained_mask

    @staticmethod
    def _linear_weights(relevance_scores: pd.Series) -> pd.Series:
        n = len(relevance_scores)
        return (1 / n) + (1 / (n - 1)) * relevance_scores
