"""Mahalanobis distance utilities for the RBP relevance calculation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RIDGE_REGULARIZATION = 1e-6


class MahalanobisDistanceCalculator:
    """Computes squared Mahalanobis distances against a fixed training distribution.

    .. math::

        d^2(x, y) = (x - y)^\\top \\, \\Sigma^{-1} \\, (x - y)
    """

    def __init__(self, training_data: pd.DataFrame):
        self.feature_columns = training_data.columns.tolist()
        self.mean_vector = training_data.mean().values
        self.inverse_covariance = self._compute_inverse_covariance(training_data)

    @staticmethod
    def _compute_inverse_covariance(data: pd.DataFrame) -> np.ndarray:
        cov = data.cov().values
        try:
            return np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix is singular; adding ridge regularization.")
            stabilised = cov + np.eye(cov.shape[0]) * RIDGE_REGULARIZATION
            return np.linalg.inv(stabilised)

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        return float(diff @ self.inverse_covariance @ diff)
