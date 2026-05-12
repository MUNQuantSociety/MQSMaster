"""Feature engineering for the RBP pipeline.

Produces the predictive features (X) and forward-return target (Y)
used by the relevance-based predictor. Combines simple price-based
features with the project's stateful technical indicators.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolios.indicators.average_true_range import AverageTrueRange
from src.portfolios.indicators.displaced_moving_average import DisplacedMovingAverage
from src.portfolios.indicators.rate_of_change import RateOfChange
from src.portfolios.indicators.relative_momentum_index import RelativeMomentumIndex
from src.portfolios.indicators.relative_strength_index import RelativeStrengthIndex
from src.portfolios.indicators.simple_moving_average import SimpleMovingAverage
from src.portfolios.indicators.vwap import VWAP

logger = logging.getLogger(__name__)


PRICE_FEATURES: List[str] = [
    "past_return_21d",
    "past_return_63d",
    "past_return_252d",
    "past_vol_21d",
    "past_vol_63d",
]

INDICATOR_FEATURES: List[str] = [
    "sma_21",
    "rsi_14",
    "rmi_14",
    "roc_21",
    "atr_14",
    "dma_21",
    "vwap_21",
]

DEFAULT_FEATURES: List[str] = PRICE_FEATURES + INDICATOR_FEATURES
TARGET_COLUMN: str = "target_return_21d"


class FeatureEngineer:
    """Converts raw OHLCV bars into RBP-ready features and targets."""

    def __init__(self, target_horizon: int = 21):
        self.target_horizon = target_horizon

    def engineer(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based and indicator features plus the forward-return target."""
        if market_data.empty:
            logger.warning("Empty market_data passed to FeatureEngineer.")
            return market_data

        data = market_data.copy()
        data.sort_values(["ticker", "timestamp"], inplace=True)

        data = self._add_price_features(data)
        data = self._add_target(data)
        data = self._add_indicator_features(data)

        rows_before = len(data)
        data.dropna(subset=DEFAULT_FEATURES + [TARGET_COLUMN], inplace=True)
        logger.info(
            "Feature engineering complete. %d/%d rows remain after NaN drop.",
            len(data),
            rows_before,
        )
        return data

    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        grouped = data.groupby("ticker")["close_price"]
        daily_returns = grouped.pct_change()

        data["past_return_21d"] = grouped.pct_change(21)
        data["past_return_63d"] = grouped.pct_change(63)
        data["past_return_252d"] = grouped.pct_change(252)

        data["past_vol_21d"] = daily_returns.groupby(data["ticker"]).transform(
            lambda x: x.rolling(21).std()
        )
        data["past_vol_63d"] = daily_returns.groupby(data["ticker"]).transform(
            lambda x: x.rolling(63).std()
        )
        return data

    def _add_target(self, data: pd.DataFrame) -> pd.DataFrame:
        future_close = data.groupby("ticker")["close_price"].shift(-self.target_horizon)
        data[TARGET_COLUMN] = future_close / data["close_price"] - 1
        return data

    def _add_indicator_features(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in INDICATOR_FEATURES:
            data[col] = np.nan

        results: Dict[int, Dict[str, float]] = {}

        for ticker, group in data.groupby("ticker", sort=False):
            indicators = self._build_indicators(ticker)

            for idx, row in group.iterrows():
                ts = row["timestamp"]
                close = float(row["close_price"])
                high = float(row.get("high_price", close)) if pd.notna(row.get("high_price")) else close
                low = float(row.get("low_price", close)) if pd.notna(row.get("low_price")) else close
                volume = float(row.get("volume", 1.0)) if pd.notna(row.get("volume")) else 1.0

                indicators["sma_21"].Update(ts, close)
                indicators["rsi_14"].Update(ts, close)
                indicators["rmi_14"].Update(ts, close)
                indicators["roc_21"].Update(ts, close)
                indicators["atr_14"].Update(ts, close, high_price=high, low_price=low)
                indicators["dma_21"].Update(ts, close)
                indicators["vwap_21"].Update(ts, close, volume=volume)

                row_vals = {
                    name: ind.Current
                    for name, ind in indicators.items()
                    if ind.IsReady and ind.Current is not None
                }
                if row_vals:
                    results[idx] = row_vals

        if results:
            updates = pd.DataFrame.from_dict(results, orient="index")
            data.update(updates)

        for col in INDICATOR_FEATURES:
            ready = data[col].notna().sum()
            logger.info("  %s: %d non-NaN values out of %d rows", col, ready, len(data))

        return data

    @staticmethod
    def _build_indicators(ticker: str) -> Dict[str, object]:
        return {
            "sma_21": SimpleMovingAverage(ticker, period=21),
            "rsi_14": RelativeStrengthIndex(ticker, period=14),
            "rmi_14": RelativeMomentumIndex(ticker, period=14, momentum_period=3),
            "roc_21": RateOfChange(ticker, period=21),
            "atr_14": AverageTrueRange(ticker, period=14),
            "dma_21": DisplacedMovingAverage(ticker, period=21, displacement=5),
            "vwap_21": VWAP(ticker, period=21),
        }
