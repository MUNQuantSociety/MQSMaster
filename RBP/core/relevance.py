"""Relevance scoring: similarity + informativeness (paper Eq. 1-4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .distance import MahalanobisDistanceCalculator


class RelevanceCalculator:
    """Scores how relevant past observations are to a current prediction task.

    Relevance combines:

    * **Similarity** — closeness of the past observation to the task vector.
    * **Informativeness** — distance of the past observation, and of the task,
      from the training mean (more unusual = more informative).
    """

    def __init__(self, distance_calculator: MahalanobisDistanceCalculator):
        self.distance = distance_calculator
        self.mean_vector = distance_calculator.mean_vector

    def score(self, past_observation: np.ndarray, current_task: np.ndarray) -> float:
        similarity = self.distance.distance(past_observation, current_task)
        info_past = self.distance.distance(past_observation, self.mean_vector)
        info_current = self.distance.distance(current_task, self.mean_vector)
        return -0.5 * similarity + 0.5 * (info_past + info_current)

    def score_all(
        self, current_task: np.ndarray, past_observations: pd.DataFrame
    ) -> pd.Series:
        """Return a relevance score per row in ``past_observations``."""
        return past_observations.apply(
            lambda row: self.score(row.values, current_task), axis=1
        )
