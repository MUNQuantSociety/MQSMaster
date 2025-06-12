# portfolios/portfolio_2/strategy.py

import os
import json
import logging
import pandas as pd # Import pandas
from datetime import datetime, timedelta
from portfolios.portfolio_BASE.strategy import BasePortfolio
from typing import List, Dict, Optional, Union


class SimpleMeanReversion(BasePortfolio):
    def __init__(self, db_connector, executor, debug=False):
        # 1->> Load local config.json into the portfolio!
        child_dir = os.path.dirname(__file__)
        config_path = os.path.join(child_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.json in {child_dir}")

        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding config.json: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load config.json: {e}") from e

        # 2->> Pass config_data to the BasePortfolio constructor
        super().__init__(db_connector=db_connector,
                         executor=executor,
                         debug=debug,
                         config_dict=config_data)

        # Strategy specific logger and params
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")
        self.strategy_lookback_minutes = 30
        self.logger.info(f"SimpleMeanReversion portfolio '{self.portfolio_id}'.")


    # *** MODIFIED: Update signature and logic ***
    def generate_signals_and_trade(self,
                                     market_data: Union[pd.DataFrame, List[Dict]],
                                     current_time: Optional[datetime] = None):
        """
        Calculates a 30-minute rolling mean and compares to the latest price.
        Handles both DataFrame (backtest) and List[Dict] (live) input.
        """
        # --- Handle Input Type ---
        if isinstance(market_data, pd.DataFrame):
            df = market_data # Backtest path
        elif isinstance(market_data, list):
            # Live path: Convert list of dicts to DataFrame
            if not market_data: return
            try:
                df = pd.DataFrame(market_data)
                # Ensure required columns exist and have correct types
                if not all(col in df.columns for col in ['timestamp', 'ticker', 'close_price']):
                     self.logger.error("Live market data missing required columns.")
                     return
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
                df = df.dropna(subset=['timestamp', 'ticker', 'close_price'])
                df.sort_values('timestamp', inplace=True) # Sort for consistent processing
            except Exception as e:
                 self.logger.error(f"Error processing live market data list: {e}")
                 return
        else:
            self.logger.error(f"generate_signals_and_trade received unexpected data type: {type(market_data)}")
            return

        if df.empty:
            self.logger.debug("Strategy received empty or unusable market data.")
            return

        # Determine the reference time
        # Use current_time if provided (from backtester), MUST be the correct time for backtest trades!
        # If live mode (current_time is None), use the latest time from the data batch.
        ref_time = current_time if current_time is not None else datetime.now()
        if pd.isna(ref_time):
             self.logger.warning("Could not determine reference time.")
             return

        window_start = ref_time - timedelta(minutes=self.strategy_lookback_minutes)
        # Filter the dataframe (whether from backtest slice or converted live data)
        df_window = df[df['timestamp'] >= window_start].copy()

        if df_window.empty:
            self.logger.debug(f"No data within the {self.strategy_lookback_minutes}-minute window ending {ref_time}")
            return

        # --- Process each ticker ---
        for ticker in self.tickers:
            ticker_df = df_window[df_window['ticker'] == ticker]
            if len(ticker_df) < 2:
                self.logger.debug(f"Ticker {ticker}: Not enough data points ({len(ticker_df)}) in window.")
                continue

            try:
                mean_price = ticker_df['close_price'].mean()
            except Exception as e:
                self.logger.error(f"Ticker {ticker}: Error calculating mean: {e}")
                continue

            # Get the price closest to the reference time within the window
            # For backtest, ref_time == current_time, so this should find the exact row
            # For live, this finds the latest price in the fetched batch
            latest_price_row = ticker_df[ticker_df['timestamp'] == ref_time]
            if latest_price_row.empty:
                # If exact match not found (can happen in live if data isn't perfectly aligned)
                # Fallback: use the absolute latest price for that ticker in the window
                latest_price_row = ticker_df.loc[[ticker_df['timestamp'].idxmax()]]
                if latest_price_row.empty:
                     self.logger.warning(f"Ticker {ticker}: Could not find any price in window ending {ref_time}.")
                     continue # Skip if still no price found

            latest_price = latest_price_row['close_price'].iloc[0]
            latest_ts_used = latest_price_row['timestamp'].iloc[0] # Log which timestamp's price we actually used

            if pd.isna(latest_price) or pd.isna(mean_price):
                 self.logger.warning(f"Ticker {ticker}: NaN price encountered (Latest: {latest_price}, Mean: {mean_price}).")
                 continue

            # --- Generate Buy/Sell Signal & Execute ---
            # Use the current_time passed from the backtester for accurate trade log timestamping
            # If current_time is None (live mode), execute_trade will handle the fallback timestamp internally.
            trade_ts = current_time # This IS the simulation time during backtest

            if latest_price < mean_price:
                self.logger.debug(f"Signal: BUY {ticker} (Price {latest_price:.4f} @ {latest_ts_used} < Mean {mean_price:.4f}) based on time {ref_time}")
                self.execute_trade(ticker, 'BUY', confidence=1.0, timestamp=trade_ts) # Pass backtest time
            elif latest_price > mean_price:
                self.logger.debug(f"Signal: SELL {ticker} (Price {latest_price:.4f} @ {latest_ts_used} > Mean {mean_price:.4f}) based on time {ref_time}")
                self.execute_trade(ticker, 'SELL', confidence=1.0, timestamp=trade_ts) # Pass backtest time
            else:
                 self.logger.debug(f"Signal: HOLD {ticker} (Price {latest_price:.4f} approx = Mean {mean_price:.4f}) based on time {ref_time}")