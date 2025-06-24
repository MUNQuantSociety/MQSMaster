import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from portfolios.portfolio_BASE.strategy import BasePortfolio
from typing import List, Dict, Optional, Union
import pytz


class SimpleMeanReversion(BasePortfolio):
    def __init__(self, db_connector, executor, debug=False):
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

        super().__init__(db_connector=db_connector,
                         executor=executor,
                         debug=debug,
                         config_dict=config_data)

        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")
        self.strategy_lookback_minutes = 30
        
        # --- START OF NEW CODE: Setup dedicated debug logger ---
        # This logger will write detailed debug info to a file for analysis.
        self.debug_logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}_DEBUG")
        self.debug_logger.setLevel(logging.DEBUG)
        
        # Prevent debug messages from propagating to the main console logger
        self.debug_logger.propagate = False

        # If handlers are already present, clear them to avoid duplicates on re-init
        if self.debug_logger.hasHandlers():
            self.debug_logger.handlers.clear()

        # Create a file handler to write logs to a file
        debug_log_file = os.path.join(child_dir, 'debug_portfolio_2.log')
        file_handler = logging.FileHandler(debug_log_file, mode='w') # 'w' overwrites the file on each run
        file_handler.setLevel(logging.DEBUG)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.debug_logger.addHandler(file_handler)
        self.logger.info(f"SimpleMeanReversion portfolio '{self.portfolio_id}' initialized. Detailed debug output will be saved to {debug_log_file}")
        # --- END OF NEW CODE ---


    def generate_signals_and_trade(self,
                                   dataframes_dict: Dict[str, pd.DataFrame],
                                   current_time: Optional[datetime] = None):

        market_data = dataframes_dict.get('MARKET_DATA', None)
        cash_available = dataframes_dict.get('CASH_EQUITY', None)
        positions = dataframes_dict.get('POSITIONS', None)
        port_notional = dataframes_dict.get('PORT_NOTIONAL', None)

        if market_data is None or market_data.empty:
            self.logger.debug("Strategy received no valid market data.")
            return

        df = market_data

        for ticker in self.tickers:
            try:
                self.debug_logger.debug(f"\n--- Processing Ticker: {ticker} ---")
                
                ticker_data = df[df['ticker'] == ticker].copy()
                
                if ticker_data.empty:
                    self.debug_logger.debug(f"[{ticker}] No market data found for this ticker in the provided DataFrame. Skipping.")
                    continue

                ticker_data.set_index('timestamp', inplace=True, drop=False)
                ticker_data = ticker_data[~ticker_data.index.duplicated(keep='last')].sort_index()

                tz = ticker_data.index.tz
                if tz is None:
                    tz = pytz.UTC

                trade_ts = current_time if current_time is not None else datetime.now(tz)
                if trade_ts.tzinfo is None:
                    trade_ts = tz.localize(trade_ts)
                
                window_start = trade_ts - timedelta(minutes=self.strategy_lookback_minutes)
                
                self.debug_logger.debug(f"[{ticker}] Current simulation time (trade_ts): {trade_ts}")
                self.debug_logger.debug(f"[{ticker}] Lookback window start time:        {window_start}")
                
                df_window = ticker_data[(ticker_data['timestamp'] >= window_start) & 
                                        (ticker_data['timestamp'] <= trade_ts)].copy()

                self.debug_logger.debug(f"[{ticker}] Found {len(df_window)} data points in the lookback window.")
                
                if df_window.empty:
                    self.logger.debug(f"[{ticker}] No data in lookback window.")
                    continue

                mean_price = df_window['close_price'].mean()
                latest_row = ticker_data.iloc[-1]
                latest_price = latest_row['close_price']

                self.debug_logger.debug(f"[{ticker}] Latest Price = {latest_price:.2f} | Mean Price in Window = {mean_price:.2f}")

                if pd.isna(latest_price) or pd.isna(mean_price):
                    self.logger.warning(f"[{ticker}] Invalid price or mean encountered.")
                    continue

                signal = 'BUY' if latest_price < mean_price else 'SELL'
                if latest_price == mean_price:
                    self.logger.debug(f"[{ticker}] No trade signal (price ≈ mean).")
                    continue
                
                self.debug_logger.debug(f"[{ticker}] Signal Generated: {signal}. Preparing to execute trade.")

                ticker_positions = positions[positions['ticker'] == ticker]['quantity'] if not positions.empty else pd.Series(dtype=float)
                current_quantity = ticker_positions.iloc[0] if not ticker_positions.empty else 0.0

                self.executor.execute_trade(
                    self.portfolio_id, ticker, signal_type=signal, confidence=1.0, arrival_price=latest_price,
                    cash=cash_available.iloc[0]['notional'] if not cash_available.empty else 0.0,
                    positions=current_quantity,
                    port_notional=port_notional.iloc[0]['notional'] if not port_notional.empty else 0.0,
                    ticker_weight=self.portfolio_weights.get(ticker, 1.0 / len(self.tickers)),
                    timestamp=trade_ts
                )
                self.logger.debug(f"Executed {signal} at {latest_price:.2f} (mean={mean_price:.2f})")

            except Exception as e:
                self.logger.exception(f"[{ticker}] Error in signal generation or execution: {e}")