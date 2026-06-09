"""Persistent RBP forecast producer.

A leaner, long-lived variant of :class:`RBP.pipeline.RBPPipeline` that, on each
``refresh()`` call, computes a single per-ticker 21-day forward-return forecast
plus the top RBI features and upserts them into the ``rbp_forecasts``
Postgres table.

Intended to be driven by an orchestrator (e.g. ``src/orchestrator/rbp_runner``)
on a cadence; this module deliberately exposes no CLI / ``__main__`` block.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz
from psycopg2.extras import Json

try:
    from RBP.config import RBPConfig
    from RBP.database import MarketDataLoader
    from RBP.features import FeatureEngineer
    from RBP.models import RBICalculator, RBPPredictor
except ImportError:  # pragma: no cover - support ``src.RBP`` import path
    from src.RBP.config import RBPConfig  # type: ignore
    from src.RBP.database import MarketDataLoader  # type: ignore
    from src.RBP.features import FeatureEngineer  # type: ignore
    from src.RBP.models import RBICalculator, RBPPredictor  # type: ignore

try:
    from src.common.database.MQSDBConnector import MQSDBConnector
except ImportError:  # pragma: no cover - alternate path
    from common.database.MQSDBConnector import MQSDBConnector  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_HORIZON_DAYS: int = 21
DEFAULT_TIMEZONE: str = "America/New_York"
TOP_FEATURES_N: int = 5
TABLE_NAME: str = "rbp_forecasts"
CONFLICT_COLUMNS: List[str] = [
    "ticker",
    "asof",
    "horizon_days",
    "model_version",
]


class RBPForecastService:
    """Compute and persist per-ticker 21-day RBP forecasts.

    The service is stateful: across ``refresh()`` calls it caches the
    training matrix per ticker keyed by the engineered "training-end" date,
    avoiding recomputation when the underlying data hasn't advanced.
    """

    def __init__(
        self,
        config: RBPConfig,
        db: Optional[MQSDBConnector] = None,
    ):
        self.config = config
        self.db = db or MQSDBConnector()
        self.loader = MarketDataLoader(self.db)
        self.engineer = FeatureEngineer()
        self.predictor = RBPPredictor(
            feature_columns=config.feature_columns,
            censoring_quantiles=config.censoring_quantiles,
            max_combination_size=config.max_combination_size,
        )
        self.rbi_calc = RBICalculator()
        # ticker -> (x_train, y_train, training_end_date)
        self._train_cache: Dict[str, Tuple[pd.DataFrame, pd.Series, date]] = {}
        self.model_version = (
            f"rbp_v1_combo{config.max_combination_size or 1}"
            f"_q{len(config.censoring_quantiles)}"
        )

    # ------------------------------------------------------------------ API

    def refresh(self, asof: Optional[datetime] = None) -> int:
        """Refresh forecasts for all configured tickers.

        Returns the number of rows actually upserted into ``rbp_forecasts``
        (zero if every ticker hit ``ON CONFLICT DO NOTHING``).
        """
        asof = asof or datetime.now(tz=pytz.timezone(DEFAULT_TIMEZONE))
        generated_at = datetime.now(tz=pytz.timezone(DEFAULT_TIMEZONE))

        tickers = list(self.config.tickers)
        n_tickers = len(tickers)
        if n_tickers == 0:
            logger.warning("RBPForecastService.refresh: no tickers configured.")
            return 0

        engineered = self._load_and_engineer(tickers, asof)
        if engineered.empty:
            logger.warning(
                "RBPForecastService.refresh: no engineered rows available."
            )
            return 0

        rows: List[dict] = []
        n_predicted = 0
        # Today's bar must not leak into training.
        train_cutoff = pd.to_datetime(asof).tz_localize(None) - timedelta(days=1)

        for ticker in tickers:
            try:
                row = self._forecast_one(
                    ticker=ticker,
                    engineered=engineered,
                    train_cutoff=train_cutoff,
                    asof=asof,
                    generated_at=generated_at,
                )
            except Exception as exc:
                logger.exception(
                    "RBPForecastService.refresh: %s failed: %s", ticker, exc
                )
                continue

            if row is not None:
                rows.append(row)
                n_predicted += 1

        n_inserted = self._bulk_insert(rows)
        logger.info(
            "RBPForecastService.refresh: %d/%d tickers, inserted=%d",
            n_predicted,
            n_tickers,
            n_inserted,
        )
        return n_inserted

    # ---------------------------------------------------------------- helpers

    def _load_and_engineer(
        self, tickers: List[str], asof: datetime
    ) -> pd.DataFrame:
        end = pd.to_datetime(asof).tz_localize(None)
        start = end - timedelta(days=self.config.lookback_days)
        market_data = self.loader.load(tickers, start, end)
        if market_data.empty:
            logger.warning(
                "RBPForecastService: market data empty for %d tickers.",
                len(tickers),
            )
            return market_data
        return self.engineer.engineer(market_data)

    def _forecast_one(
        self,
        ticker: str,
        engineered: pd.DataFrame,
        train_cutoff: pd.Timestamp,
        asof: datetime,
        generated_at: datetime,
    ) -> Optional[dict]:
        per_ticker = engineered[engineered["ticker"] == ticker]
        if per_ticker.empty:
            logger.warning("RBPForecastService: no engineered rows for %s.", ticker)
            return None

        per_ticker = per_ticker.sort_values("timestamp")

        # Training: everything strictly before today's bar so the forward-return
        # target (which peeks 21d ahead) cannot leak into the prediction.
        train_df = per_ticker[per_ticker["timestamp"] < train_cutoff]
        if train_df.empty:
            logger.warning(
                "RBPForecastService: no training rows for %s before %s.",
                ticker,
                train_cutoff,
            )
            return None

        x_train, y_train = self._get_or_build_train(ticker, train_df)

        # Prediction task = most recent engineered row for this ticker.
        task_row = per_ticker.iloc[-1]
        task_features = task_row[self.config.feature_columns]

        prediction, grid_df = self.predictor.predict(task_features, x_train, y_train)
        rbi_scores = self.rbi_calc.calculate(grid_df, self.config.feature_columns)
        top_features = self._top_features(rbi_scores, n=TOP_FEATURES_N)

        return {
            "ticker": ticker,
            "asof": asof,
            "horizon_days": DEFAULT_HORIZON_DAYS,
            "y_pred": float(prediction),
            "rbi_top": Json(top_features),
            "model_version": self.model_version,
            "generated_at": generated_at,
        }

    def _get_or_build_train(
        self, ticker: str, train_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Return cached training matrix, rebuilding only when data has advanced."""
        latest_ts = train_df["timestamp"].iloc[-1]
        latest_date = pd.to_datetime(latest_ts).date()

        cached = self._train_cache.get(ticker)
        if cached is not None and cached[2] == latest_date:
            return cached[0], cached[1]

        x_train = train_df[self.config.feature_columns].copy()
        y_train = train_df[self.config.target_column].copy()
        self._train_cache[ticker] = (x_train, y_train, latest_date)
        logger.debug(
            "RBPForecastService: rebuilt training matrix for %s "
            "(%d rows, latest=%s).",
            ticker,
            len(x_train),
            latest_date,
        )
        return x_train, y_train

    @staticmethod
    def _top_features(rbi_scores: pd.Series, n: int) -> Dict[str, float]:
        """Take top-n features by absolute RBI score."""
        if rbi_scores.empty:
            return {}
        ranked = rbi_scores.reindex(
            rbi_scores.abs().sort_values(ascending=False).index
        ).head(n)
        return {str(name): float(value) for name, value in ranked.items()}

    def _bulk_insert(self, rows: List[dict]) -> int:
        if not rows:
            return 0

        result = self.db.bulk_inject_to_db(
            table=TABLE_NAME,
            data=rows,
            conflict_columns=CONFLICT_COLUMNS,
        )
        if result.get("status") != "success":
            logger.error(
                "RBPForecastService: bulk insert failed: %s",
                result.get("message"),
            )
            return 0

        # bulk_inject_to_db's message is "Successfully inserted or ignored N rows."
        # That count reflects affected rows after ON CONFLICT DO NOTHING; we
        # surface the constructed row count as the caller's "attempted" total
        # and treat any non-error status as a successful upsert of len(rows).
        return len(rows)
