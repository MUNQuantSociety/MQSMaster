import logging
from datetime import datetime
from typing import Optional

import pandas as pd

_LOGGER = logging.getLogger(__name__)


class AssetData:
    """
    Represents the market data for a single asset at a specific point in time.
    """

    def __init__(
        self,
        ticker: str,
        asset_specific_df: pd.DataFrame,
        current_time: Optional[datetime],
    ):
        self._ticker = ticker
        self._df = asset_specific_df
        self._time = current_time

        if self._df.empty:
            self.Exists = False
            self._set_defaults()
            return

        effective_time = current_time
        if effective_time is None:
            try:
                effective_time = self._df.index.max()
            except Exception:
                effective_time = None

        # Obtain latest row up to effective_time (or overall last row if None)
        if effective_time is None:
            latest_data = self._df
        else:
            try:
                latest_data = self._df.loc[self._df.index <= effective_time]
            except TypeError:
                latest_data = self._df

        if latest_data.empty:
            self.Exists = False
            self._set_defaults()
            return

        self.latest_row = latest_data.iloc[-1]

        # Safely extract numeric fields; if any core price is missing, mark as non-existent
        def _to_float(val):
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        self.Open = _to_float(self.latest_row.get("open_price"))
        self.High = _to_float(self.latest_row.get("high_price"))
        self.Low = _to_float(self.latest_row.get("low_price"))
        self.Close = _to_float(self.latest_row.get("close_price"))
        self.Volume = _to_float(self.latest_row.get("volume"))

        if self.Close is None:
            self.Exists = False
            self._set_defaults()
            return
        self.Timestamp = self.latest_row.name
        self.Exists = True

    def _set_defaults(self):
        """Helper to set properties to None when no data is available."""
        self.latest_row = None
        self.Open = None
        self.High = None
        self.Low = None
        self.Close = None
        self.Volume = None
        self.Timestamp = None

    def History(self, lookback_period: str) -> pd.DataFrame:
        if self._df.empty:
            return pd.DataFrame()

        def _index_summary(index: pd.Index) -> str:
            try:
                return (
                    "len={length}, dtype={dtype}, min={min_val}, max={max_val}".format(
                        length=len(index),
                        dtype=getattr(index, "dtype", None),
                        min_val=index.min(),
                        max_val=index.max(),
                    )
                )
            except Exception:
                try:
                    return "len={length}, dtype={dtype}".format(
                        length=len(index),
                        dtype=getattr(index, "dtype", None),
                    )
                except Exception:
                    return "unavailable"

        # Use effective time: if _time is None, use the latest timestamp in the data
        end_date = self._time
        if end_date is None:
            try:
                end_date = self._df.index.max()
            except Exception:
                _LOGGER.warning(
                    "AssetData.History failed to compute end_date; self._time=%s lookback_period=%s index=%s",
                    self._time,
                    lookback_period,
                    _index_summary(self._df.index),
                    exc_info=True,
                )
                return pd.DataFrame()

        try:
            start_date = end_date - pd.to_timedelta(lookback_period)
        except Exception:
            _LOGGER.warning(
                "AssetData.History failed to parse lookback_period; self._time=%s lookback_period=%s index=%s",
                self._time,
                lookback_period,
                _index_summary(self._df.index),
                exc_info=True,
            )
            return pd.DataFrame()

        hist_df = self._df.loc[
            (self._df.index >= start_date) & (self._df.index <= end_date)
        ]
        return hist_df.copy()

    def __repr__(self) -> str:
        if not self.Exists:
            return f"AssetData(ticker='{self._ticker}', Exists=False)"
        return f"AssetData(ticker='{self._ticker}', Time='{self.Timestamp}', Close={self.Close})"


class MarketData:
    """
    A high-level and performant interface for accessing all market data.
    It pre-processes the data by grouping it by ticker upon initialization.
    """

    def __init__(self, market_data_df: pd.DataFrame, current_time: datetime):
        self._time = current_time
        self._cache = {}

        if market_data_df is not None and not market_data_df.empty:
            df = market_data_df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df["ticker"] = df["ticker"].astype("string")
            df = df.dropna(subset=["timestamp", "ticker"]).infer_objects(copy=False)
            df = df.set_index("timestamp")
            self._grouped_data = dict(list(df.groupby("ticker")))
            self._unique_tickers = set(self._grouped_data.keys())
        else:
            self._grouped_data = {}
            self._unique_tickers = set()

    def __getitem__(self, ticker: str) -> AssetData:
        if ticker in self._cache:
            return self._cache[ticker]

        asset_specific_df = self._grouped_data.get(ticker, pd.DataFrame())
        asset = AssetData(ticker, asset_specific_df, self._time)
        self._cache[ticker] = asset
        return asset

    def __contains__(self, ticker: str) -> bool:
        return ticker in self._unique_tickers
