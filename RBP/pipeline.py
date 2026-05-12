"""End-to-end RBP pipeline.

Orchestrates: DB load -> feature engineering -> train/test split ->
parallel grid prediction + RBI scoring.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
from joblib import Parallel, delayed

from RBP.config import RBPConfig
from RBP.database import MarketDataLoader
from RBP.features import FeatureEngineer
from RBP.models import RBICalculator, RBPPredictor

logger = logging.getLogger(__name__)


class RBPPipeline:
    """Composes the loader, feature engineer, predictor and RBI calculator."""

    def __init__(
        self,
        config: RBPConfig,
        loader: Optional[MarketDataLoader] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
    ):
        self.config = config
        self.loader = loader or MarketDataLoader()
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.predictor = RBPPredictor(
            feature_columns=config.feature_columns,
            censoring_quantiles=config.censoring_quantiles,
            max_combination_size=config.max_combination_size,
        )
        self.rbi_calculator = RBICalculator()

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute the full pipeline. Returns ``(predictions_df, rbi_scores_df)``."""
        market_data = self._load_market_data()
        if market_data.empty:
            raise RuntimeError("No market data returned from DB; aborting.")

        engineered = self.feature_engineer.engineer(market_data)
        x_train, y_train, x_test, y_test = self._split(engineered)

        if x_test.empty or x_train.empty:
            raise RuntimeError(
                f"Train/test split produced empty set "
                f"(train={len(x_train)}, test={len(x_test)})."
            )

        return self._predict_batch(x_train, y_train, x_test, y_test)

    def _load_market_data(self) -> pd.DataFrame:
        end = datetime.now()
        start = end - timedelta(days=self.config.lookback_days)
        return self.loader.load(self.config.tickers, start, end)

    def _split(
        self, engineered: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        split_ts = pd.to_datetime(self.config.train_test_split_date)
        train_mask = engineered["timestamp"] < split_ts

        train = engineered[train_mask]
        test = engineered[~train_mask]

        x_train = train[self.config.feature_columns]
        y_train = train[self.config.target_column]
        x_test = test[self.config.feature_columns]
        y_test = test[self.config.target_column]

        if self.config.max_test_tasks is not None and len(x_test) > self.config.max_test_tasks:
            sample = x_test.sample(
                n=self.config.max_test_tasks, random_state=0
            ).sort_index()
            x_test = sample
            y_test = y_test.loc[sample.index]
            logger.info(
                "Capped test set to %d random tasks (set max_test_tasks=None to disable).",
                self.config.max_test_tasks,
            )

        logger.info(
            "Train: %d rows, Test: %d rows (split at %s)",
            len(x_train),
            len(x_test),
            self.config.train_test_split_date,
        )
        return x_train, y_train, x_test, y_test

    def _predict_batch(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Running predictions for %d test tasks...", len(x_test))

        # Predictor + feature_columns are pickle-safe; self is not (DB connection).
        predictor = self.predictor
        feature_columns = self.config.feature_columns

        outputs = Parallel(n_jobs=self.config.n_jobs, backend="loky", verbose=10)(
            delayed(_predict_one)(idx, row, x_train, y_train, predictor, feature_columns)
            for idx, row in x_test.iterrows()
        )
        outputs = [o for o in outputs if o is not None]
        if not outputs:
            raise RuntimeError("All prediction tasks failed.")

        prediction_rows, rbi_series = zip(*outputs)

        predictions_df = pd.DataFrame(prediction_rows).set_index("task_index")
        predictions_df["y_actual"] = y_test
        rbi_df = pd.DataFrame(list(rbi_series))

        return predictions_df, rbi_df


def _predict_one(
    task_index,
    task_features: pd.Series,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    predictor: RBPPredictor,
    feature_columns,
):
    """Module-level worker so joblib/loky can pickle it without dragging the DB connection."""
    try:
        prediction, grid = predictor.predict(task_features, x_train, y_train)
        rbi = RBICalculator().calculate(grid, feature_columns)
        rbi.name = task_index
        return {"task_index": task_index, "y_pred_rbp": prediction}, rbi
    except Exception as exc:
        logger.error("Task %s failed: %s", task_index, exc)
        return None
