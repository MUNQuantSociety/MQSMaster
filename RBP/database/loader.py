"""Market data loader for the RBP pipeline.

Reads OHLCV bars from the project's PostgreSQL ``market_data`` table via
:class:`MQSDBConnector`. Mirrors the pattern used by
``src/portfolios/portfolio_BASE/strategy.py``.
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.database.MQSDBConnector import MQSDBConnector

logger = logging.getLogger(__name__)

DateLike = Union[str, date, datetime]


class MarketDataLoader:
    """Loads OHLCV bars from the ``market_data`` table.

    The table stores intraday bars; this loader resamples to daily by default
    so the RBP feature windows (21d / 63d / 252d) are interpreted in trading
    days, matching the original notebook.
    """

    MARKET_DATA_QUERY = (
        "SELECT ticker, timestamp, open_price, high_price, low_price, "
        "close_price, volume "
        "FROM market_data "
        "WHERE ticker IN ({placeholders}) "
        "AND timestamp BETWEEN %s AND %s "
        "ORDER BY ticker, timestamp"
    )

    def __init__(self, db: Optional[MQSDBConnector] = None, resample_to_daily: bool = True):
        self.db = db or MQSDBConnector()
        self.resample_to_daily = resample_to_daily

    def load(
        self,
        tickers: Sequence[str],
        start: DateLike,
        end: DateLike,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for the given tickers between ``start`` and ``end``.

        Returns an empty DataFrame if the query yields no rows.
        """
        if not tickers:
            logger.warning("No tickers provided; returning empty DataFrame.")
            return pd.DataFrame()

        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)

        placeholders = ", ".join(["%s"] * len(tickers))
        sql = self.MARKET_DATA_QUERY.format(placeholders=placeholders)
        params = list(tickers) + [start_ts, end_ts]

        logger.info(
            "Loading market data for %d tickers from %s to %s",
            len(tickers),
            start_ts.date(),
            end_ts.date(),
        )

        result = self.db.execute_query(sql, params, fetch="all")

        if result["status"] != "success" or not result.get("data"):
            logger.warning("Market data query returned no rows: %s", result.get("message"))
            return pd.DataFrame()

        df = pd.DataFrame(result["data"])
        df = self._coerce_types(df)

        if self.resample_to_daily:
            df = self._to_daily(df)
            logger.info("Resampled to daily: %d rows across %d tickers", len(df), df["ticker"].nunique())

        return df

    @staticmethod
    def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)

        numeric_cols = ["open_price", "high_price", "low_price", "close_price", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=["timestamp", "ticker", "close_price"], inplace=True)
        df.sort_values(["ticker", "timestamp"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def _to_daily(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate intraday bars to daily OHLCV per ticker."""
        df = df.copy()
        df["date"] = df["timestamp"].dt.normalize()
        agg = df.groupby(["ticker", "date"]).agg(
            open_price=("open_price", "first"),
            high_price=("high_price", "max"),
            low_price=("low_price", "min"),
            close_price=("close_price", "last"),
            volume=("volume", "sum"),
        ).reset_index()
        agg.rename(columns={"date": "timestamp"}, inplace=True)
        agg.sort_values(["ticker", "timestamp"], inplace=True)
        agg.reset_index(drop=True, inplace=True)
        return agg
