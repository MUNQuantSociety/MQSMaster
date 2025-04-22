# portfolios/portfolio_3/strategy.py

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from portfolios.portfolio_BASE.strategy import BasePortfolio
from typing import List, Dict, Optional, Union

class MovingAverageCrossover(BasePortfolio):
    """
    Implements a Moving Average Crossover strategy with dynamic confidence.
    - Buys on Golden Cross (SMA > LMA), confidence based on diff magnitude.
    - Sells on Death Cross (SMA < LMA), confidence based on diff magnitude.
    (Uses shorter windows for potentially more frequent trading)
    """
    def __init__(self, db_connector, executor, debug=False):
        child_dir = os.path.dirname(__file__)
        config_path = os.path.join(child_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.json in {child_dir}")
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load/parse config.json: {e}") from e

        super().__init__(db_connector=db_connector,
                         executor=executor,
                         debug=debug,
                         config_dict=config_data)

        self.short_window_days = 10 # e.g., 5-day SMA
        self.long_window_days = 30  # e.g., 30-day LMA

        # Convert days to timedelta string format for pandas rolling
        self.short_window_str = f"{self.short_window_days}D"
        self.long_window_str = f"{self.long_window_days}D"
        # Adjust minimum lookback needed based on the longest window
        self.min_lookback_td = timedelta(days=self.long_window_days + 5) # e.g., 35 days

        # Confidence calculation parameters (can keep as before or adjust)
        self.confidence_scaling_factor = 20.0 # Scales relative diff; 20 -> 5% diff = 1.0 confidence
        self.min_confidence_threshold = 0.0 # Allow any calculated confidence > 0
        self.epsilon = 1e-9 # For floating point comparisons

        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")
        self.logger.info(
            f"{self.__class__.__name__} '{self.portfolio_id}' initialized "
            # *** Update log message to reflect new windows ***
            f"(Windows: {self.short_window_str}/{self.long_window_str}, "
            f"Min Conf: {self.min_confidence_threshold})"
        )

    def _preprocess_market_data(self, market_data: Union[pd.DataFrame, List[Dict]]) -> Optional[pd.DataFrame]:
        """Handles DataFrame/List input and basic validation."""
        # (No changes needed in this method)
        if isinstance(market_data, pd.DataFrame):
            df = market_data
        elif isinstance(market_data, list):
            if not market_data: return None
            try:
                df = pd.DataFrame(market_data)
                required_cols = ['timestamp', 'ticker', 'close_price']
                if not all(col in df.columns for col in required_cols):
                    self.logger.error("Live market data missing required columns.")
                    return None
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
                df = df.dropna(subset=required_cols)
                df.sort_values('timestamp', inplace=True)
            except Exception as e:
                self.logger.error(f"Error processing live market data list: {e}"); return None
        else:
            self.logger.error(f"Unexpected data type: {type(market_data)}"); return None

        return df if not df.empty else None

    def generate_signals_and_trade(self,
                                     market_data: Union[pd.DataFrame, List[Dict]],
                                     current_time: Optional[datetime] = None):
        """Calculates SMAs/LMAs and generates trade signals based on crossovers and confidence."""
        # (No changes needed in the logic of this method, it uses the updated window parameters from __init__)

        df = self._preprocess_market_data(market_data)
        if df is None:
            # self.logger.debug("Strategy received no valid market data.") # Optional: reduce noise
            return

        ref_time = current_time if current_time is not None else df['timestamp'].max()
        if pd.isna(ref_time):
            self.logger.warning("Could not determine reference time."); return

        trade_ts = current_time # Simulation time for accurate logging

        for ticker in self.tickers:
            ticker_data = df[df['ticker'] == ticker].copy()
            if ticker_data['timestamp'].nunique() < 2: continue

            time_range = ticker_data['timestamp'].max() - ticker_data['timestamp'].min()
            # Uses the updated self.min_lookback_td
            if time_range < self.min_lookback_td:
                continue

            try:
                ticker_data.set_index('timestamp', inplace=True, drop=False)
                ticker_data = ticker_data[~ticker_data.index.duplicated(keep='last')].sort_index()

                # Uses the updated self.short/long_window_str and days
                sma = ticker_data['close_price'].rolling(self.short_window_str, min_periods=self.short_window_days // 2).mean()
                lma = ticker_data['close_price'].rolling(self.long_window_str, min_periods=self.long_window_days // 2).mean()

                sma_valid = sma.dropna()
                lma_valid = lma.dropna()

                if len(sma_valid) < 2 or len(lma_valid) < 2: continue

                sma_last, sma_prev = sma_valid.iloc[-1], sma_valid.iloc[-2]
                lma_last, lma_prev = lma_valid.iloc[-1], lma_valid.iloc[-2]

                if not all(np.isfinite([sma_last, sma_prev, lma_last, lma_prev])): continue

            except Exception as e:
                self.logger.debug(f"{ticker}: Error calculating MAs @ {ref_time}: {e}") # Log as debug
                continue

            # Signal & Confidence Logic (remains the same)
            signal_type = None
            confidence = 0.0
            relative_diff = 0.0

            if sma_last > lma_last + self.epsilon and sma_prev <= lma_prev + self.epsilon:
                signal_type = 'BUY'
                if abs(lma_last) > self.epsilon:
                     relative_diff = abs(sma_last - lma_last) / abs(lma_last)
                     confidence = min(1.0, self.confidence_scaling_factor * relative_diff)

            elif sma_last < lma_last - self.epsilon and sma_prev >= lma_prev - self.epsilon:
                signal_type = 'SELL'
                if abs(lma_last) > self.epsilon:
                     relative_diff = abs(sma_last - lma_last) / abs(lma_last)
                     confidence = min(1.0, self.confidence_scaling_factor * relative_diff)
            
            if signal_type and confidence >= self.min_confidence_threshold:
                self.execute_trade(ticker, signal_type, confidence=confidence, timestamp=trade_ts)