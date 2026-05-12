"""Relevance-Based Importance (RBI) — paper Eq. 18."""

from __future__ import annotations

from typing import List

import pandas as pd


class RBICalculator:
    """Per-task variable importance from a populated RBP grid."""

    def calculate(
        self, grid_results: pd.DataFrame, all_feature_names: List[str]
    ) -> pd.Series:
        scores = {}
        adjusted_fit = grid_results["adjusted_fit"]

        for feature in all_feature_names:
            includes = grid_results["features"].apply(lambda combo: feature in combo)
            with_feature = adjusted_fit[includes].mean()
            without_feature = adjusted_fit[~includes].mean()

            with_feature = 0.0 if pd.isna(with_feature) else with_feature
            without_feature = 0.0 if pd.isna(without_feature) else without_feature

            scores[feature] = with_feature - without_feature

        return pd.Series(scores)
