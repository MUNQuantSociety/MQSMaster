"""The RBP grid predictor.

Given a single prediction task, builds a grid of cells over
{feature subsets} x {censoring quantiles}, scores each cell by adjusted
fit, and returns a reliability-weighted composite prediction.
"""

from __future__ import annotations

import itertools
import logging
from typing import List, Optional, Tuple

import pandas as pd

from RBP.core import (
    FitCalculator,
    GridCell,
    MahalanobisDistanceCalculator,
    PredictionWeightCalculator,
    RelevanceCalculator,
)

logger = logging.getLogger(__name__)

DEFAULT_CENSORING_QUANTILES: List[float] = [0.0, 0.2, 0.5, 0.8]


class RBPPredictor:
    """Grid-based Relevance-Based Predictor."""

    def __init__(
        self,
        feature_columns: List[str],
        censoring_quantiles: Optional[List[float]] = None,
        max_combination_size: Optional[int] = None,
    ):
        self.feature_columns = list(feature_columns)
        self.censoring_quantiles = list(censoring_quantiles or DEFAULT_CENSORING_QUANTILES)
        self.max_combination_size = max_combination_size or len(self.feature_columns)
        self.feature_combinations = self._generate_combinations()

        logger.info(
            "RBPPredictor: %d feature combinations (max_size=%d) x %d quantiles",
            len(self.feature_combinations),
            self.max_combination_size,
            len(self.censoring_quantiles),
        )

    def _generate_combinations(self) -> List[Tuple[str, ...]]:
        combos: List[Tuple[str, ...]] = []
        for size in range(1, self.max_combination_size + 1):
            combos.extend(itertools.combinations(self.feature_columns, size))
        return combos

    def predict(
        self,
        current_task: pd.Series,
        training_features: pd.DataFrame,
        training_outcomes: pd.Series,
    ) -> Tuple[float, pd.DataFrame]:
        """Return ``(composite_prediction, grid_results_df)`` for one task."""
        weight_calc = PredictionWeightCalculator()
        fit_calc = FitCalculator()

        rows = []
        for combo in self.feature_combinations:
            feature_list = list(combo)
            train_subset = training_features[feature_list]
            task_subset = current_task[feature_list].values

            distance_calc = MahalanobisDistanceCalculator(train_subset)
            relevance_calc = RelevanceCalculator(distance_calc)
            relevance_scores = relevance_calc.score_all(task_subset, train_subset)

            for quantile in self.censoring_quantiles:
                cell = GridCell(combo, quantile)
                weights, retained_mask = weight_calc.calculate(relevance_scores, quantile)

                aligned_outcomes, aligned_weights = training_outcomes.align(
                    weights, join="inner"
                )
                cell.prediction = float((aligned_weights * aligned_outcomes).sum())

                fit = fit_calc.fit(weights, training_outcomes)
                asymmetry = fit_calc.asymmetry(weights, training_outcomes, retained_mask)
                cell.adjusted_fit = fit_calc.adjusted_fit(fit, asymmetry, cell.num_features)

                rows.append(
                    {
                        "features": cell.feature_names,
                        "censoring_quantile": cell.censoring_quantile,
                        "num_features": cell.num_features,
                        "prediction": cell.prediction,
                        "adjusted_fit": cell.adjusted_fit,
                    }
                )

        grid_df = pd.DataFrame(rows)
        return self._combine(grid_df), grid_df

    @staticmethod
    def _combine(grid_results: pd.DataFrame) -> float:
        adjusted_fits = grid_results["adjusted_fit"].clip(lower=0)
        total = adjusted_fits.sum()
        if total == 0:
            logger.warning("All adjusted fits are zero; prediction unreliable.")
            return 0.0
        reliability = adjusted_fits / total
        return float((reliability * grid_results["prediction"]).sum())
