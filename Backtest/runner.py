#Backtest\runner.pyMore actions

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union

from .multiTickerexecutor import MultiTickerExecutor
from .reporting import generate_backtest_report # Consolidated reporting function
from .utils import fetch_historical_data # Keep data fetching separate
# BasePortfolio import is handled via string hint below

class BacktestRunner:
    """
    Orchestrates the execution of a multi-ticker backtest.
    (Includes iloc slicing optimization)
    """
    def __init__(self,
                 # Use string hint for BasePortfolio to avoid circular import
                 portfolio: 'BasePortfolio',
                 start_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
                 end_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
                 initial_capital: float = 100000.0):
        """
        Initializes the BacktestRunner.

        Args:
            portfolio: The portfolio instance to backtest (must inherit from BasePortfolio).
            start_date: The start date for the backtest period. Defaults to 2 years before end_date.
            end_date: The end date for the backtest period. Defaults to yesterday.
            initial_capital_per_ticker: The starting capital allocated to each ticker's sub-portfolio.
        """
        self.portfolio = portfolio
        self.logger = portfolio.logger # Use the portfolio's logger
        self.total_start_capital = initial_capital
        self.start_date = self._ensure_datetime(start_date)
        self.end_date = self._ensure_datetime(end_date, default_is_yesterday=True)

        # --- NEW: Set the lookback window using the portfolio's configuration ---
        lookback_days = getattr(self.portfolio, 'lookback_days', 365) # Default to 365 if not set
        self.strategy_lookback_window = pd.Timedelta(days=lookback_days)
        self.logger.info(f"Using strategy lookback window of {lookback_days} days, as defined by the portfolio.")
        # ---

        # Default start_date if None
        if self.start_date is None:
            # Default to 2 years before end_date if end_date is valid
            if self.end_date:
                 self.start_date = self.end_date - timedelta(days=365 * 2)
            else:

                 self.logger.error("Cannot determine start_date as end_date is invalid.")




        self.perf_records: List[Dict] = []
        self.main_data_df: pd.DataFrame = pd.DataFrame()
        self.multi_executor: Optional[MultiTickerExecutor] = None


        if hasattr(self.portfolio, 'tickers') and self.portfolio.tickers:
             self.initial_capital_per_ticker= self.total_start_capital / len(self.portfolio.tickers)
        else:
             self.logger.warning("Portfolio tickers are missing or empty. Total start capital set to 0.")
             self.total_start_capital = 0.0

    def _ensure_datetime(self, dt_val, default_is_yesterday=False) -> Optional[datetime]:
        """Converts input to a timezone-naive datetime object at midnight."""
        # (Implementation remains the same as previous version)
        if dt_val is None:
            if default_is_yesterday:
                # Use current date based on system time, then subtract a day
                try:
                    today = datetime.now().date()
                    yesterday = today - timedelta(days=1)
                    # Return timezone-naive midnight
                    return datetime(yesterday.year, yesterday.month, yesterday.day)
                except Exception as e:
                     self.logger.error(f"Error getting yesterday's date: {e}")
                     return None
            return None
        if isinstance(dt_val, datetime):
            # Return only date part, set time to midnight, make timezone-naive
            return datetime(dt_val.year, dt_val.month, dt_val.day)
        # Handle date objects directly
        if hasattr(dt_val, 'year') and not isinstance(dt_val, datetime):
            return datetime(dt_val.year, dt_val.month, dt_val.day)
        try:
            # Attempt conversion using pandas, which is flexible
            pd_dt = pd.to_datetime(dt_val, errors='coerce')
            if pd.isna(pd_dt):
                self.logger.warning(f"Could not parse '{dt_val}' as a datetime.")
                return None
            # Return standard library datetime object at midnight, timezone-naive
            return datetime(pd_dt.year, pd_dt.month, pd_dt.day)
        except Exception as e:
            self.logger.error(f"Failed to convert '{dt_val}' to datetime: {e}")
            return None


    def _prepare_data(self) -> bool:
        """Fetches, cleans, sorts, and prepares historical market data."""
        if not self.start_date or not self.end_date:
                self.logger.error("Invalid start or end date for data preparation.")
                return False
        self.logger.info(f"Preparing data for tickers: {self.portfolio.tickers}")
        self.logger.info(f"Requested date range (naive): {self.start_date.strftime('%Y-%m-%d')} -> {self.end_date.strftime('%Y-%m-%d')}")

        # --- Fetch all data from the database ---
        df = fetch_historical_data(self.portfolio, self.start_date, self.end_date)
        
        if df.empty:
            self.logger.error("No historical data found for the specified criteria.")
            return False

        # --- Data Preprocessing ---
        try:
            self.logger.info("Converting timestamp column to unified America/New_York timezone already done.")
            
            failed_rows = df['timestamp'].isnull().sum()
            if failed_rows > 0:
                self.logger.warning(f"{failed_rows} rows still have unparseable timestamps and will be dropped.")

            numeric_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
            for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

            initial_rows = len(df)
            essential_cols = ['timestamp', 'close_price', 'ticker']
            df = df.dropna(subset=essential_cols)
            if len(df) < initial_rows:
                self.logger.warning(f"Dropped {initial_rows - len(df)} rows due to NaNs in essential columns.")

            if df.empty:
                self.logger.error("No data remains after cleaning.")
                return False

            # --- Timezone-aware Filtering ---
            start_filter = self.start_date
            end_filter = self.end_date + timedelta(days=1)
            data_tz = df['timestamp'].dt.tz
            self.logger.info(f"Applying filter to NY timezeone-normalized data (tz={data_tz}).")
            start_filter = pd.Timestamp(start_filter).tz_localize(data_tz)
            end_filter = pd.Timestamp(end_filter).tz_localize(data_tz)
            self.logger.info(f"Filtering data between {start_filter} and {end_filter}")
            df = df[(df['timestamp'] >= start_filter) & (df['timestamp'] < end_filter)]

            if df.empty:
                self.logger.error("No data remains after filtering for the requested range.")
                return False

            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
            self.main_data_df = df
            self.logger.info(f"Data prepared: {len(self.main_data_df)} rows loaded.")
            return True
        except Exception as e:
            self.logger.exception(f"Error during data preparation: {e}", exc_info=True)
            return False


    def _setup_executor(self) -> None:
        """Sets up the MultiTickerExecutor."""
        self.logger.info(f"Setting up MultiTickerExecutor with {self.initial_capital_per_ticker:.2f} initial capital per ticker.")
        self.multi_executor = MultiTickerExecutor(
            tickers=self.portfolio.tickers,
            initial_capital_per_ticker=self.initial_capital_per_ticker
        )
        # Temporarily replace the portfolio's executor during the backtest
        # Store the original to restore it later
        # Ensure the attribute exists before trying to access it
        self.portfolio._original_executor = getattr(self.portfolio, 'executor', None)
        self.portfolio.executor = self.multi_executor


    def _run_event_loop(self) -> None:
        """
        Runs the main backtest simulation loop.
        (This version is optimized to use a rolling window for performance)
        """
        if self.main_data_df.empty or self.multi_executor is None:
            self.logger.error("Cannot run event loop: Data or executor not ready.")
            return

        # Ensure unique_times is a sorted DatetimeIndex for efficient slicing
        unique_times = pd.to_datetime(np.sort(self.main_data_df['timestamp'].unique()))
        total_timestamps = len(unique_times)
        if total_timestamps == 0:
            self.logger.warning("No unique timestamps found in data range. Event loop will not run.")
            return

        self.logger.info(f"Starting event loop over {total_timestamps} unique timestamps...")

        poll_td = pd.Timedelta(seconds=self.portfolio.poll_interval)
        last_poll_time: Optional[pd.Timestamp] = None
        self.perf_records = []
        
        # We need the timestamp column as a Series for fast searching
        timestamps_series = self.main_data_df['timestamp']

        # --- Loop starts ---
        for i, current_timestamp in enumerate(unique_times):
            
            # --- Polling interval check ---
            if last_poll_time is not None and (current_timestamp - last_poll_time) < poll_td:
                continue
            last_poll_time = current_timestamp

            # --- Get data for the current time step efficiently ---
            current_data_chunk = self.main_data_df[timestamps_series == current_timestamp]
            if current_data_chunk.empty:
                continue

            # --- Update Prices ---
            price_updates = dict(zip(current_data_chunk['ticker'], current_data_chunk['close_price']))
            for ticker, price in price_updates.items():
                if ticker in self.multi_executor.executors and pd.notna(price):
                    self.multi_executor.update_price(ticker, float(price))

            # --- MAJOR OPTIMIZATION: Create a rolling window slice ---
            window_start_time = current_timestamp - self.strategy_lookback_window
            
            # Use pandas' searchsorted for a highly efficient index lookup
            start_index = timestamps_series.searchsorted(window_start_time, side='left')
            end_index = current_data_chunk.index.max()

            # Create the final, smaller historical slice by index location
            historical_slice_df = self.main_data_df.iloc[start_index : end_index + 1].copy()
            # --- END OF MAJOR OPTIMIZATION ---

            # --- Call Strategy ---
            if not historical_slice_df.empty:
                try:
                    sim_time = current_timestamp.to_pydatetime()
                    data_dict = {
                        'MARKET_DATA': historical_slice_df,
                        'CASH_EQUITY': self.multi_executor.get_cash_equity_df(),
                        'POSITIONS': self.multi_executor.get_positions_df(),
                        'PORT_NOTIONAL': self.multi_executor.get_port_notional_df()
                    }
                    self.portfolio.generate_signals_and_trade(data_dict, current_time=sim_time)
                except Exception as e:
                    self.logger.exception(f"Error in strategy at {current_timestamp}: {e}", exc_info=True)

            # --- Record Portfolio State ---
            # For accuracy, we get the total portfolio value directly from the executor,
            # which always tracks the full history of cash and positions correctly.
            record = {'timestamp': current_timestamp}
            total_portfolio_value = 0.0
            for t in self.portfolio.tickers:
                if t in self.multi_executor.executors:
                    try:
                        ticker_value = self.multi_executor.executors[t].get_portfolio_value()
                        record[t] = ticker_value
                        total_portfolio_value += ticker_value
                    except Exception as e:
                        self.logger.error(f"Error getting portfolio value for ticker {t}: {e}")
                        record[t] = 'Error'
                else:
                    record[t] = 0.0 # Ticker not managed by executor
            record['portfolio_value'] = total_portfolio_value
            self.perf_records.append(record)
            # (Progress logging can be added here if desired)

        self.logger.info("Event loop finished.")


    def _calculate_results(self) -> Optional[pd.DataFrame]:
        """Calculates performance metrics from recorded data."""
        if not self.perf_records:
            self.logger.warning("No performance records generated during the backtest.")
            return None
        try:
            perf_df = pd.DataFrame(self.perf_records)
            # Ensure timestamp is datetime
            perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'])
            perf_df.sort_values('timestamp', inplace=True)
            perf_df.reset_index(drop=True, inplace=True)

            # Convert potential 'Error' strings in value columns to NaN before calculation
            value_cols = list(self.portfolio.tickers) + ['portfolio_value']
            for col in value_cols:
                 if col in perf_df.columns:
                      perf_df[col] = pd.to_numeric(perf_df[col], errors='coerce')

            # Add overall PnL percentage calculation
            if self.total_start_capital > 0:
                perf_df['pnl_pct'] = (perf_df['portfolio_value'] - self.total_start_capital) / self.total_start_capital
            else:
                perf_df['pnl_pct'] = 0.0
                self.logger.warning("Total start capital is zero, PnL percentage calculation skipped.")

            self.logger.info("Performance DataFrame calculated.")
            return perf_df
        except Exception as e:
            self.logger.exception(f"Error calculating results DataFrame: {e}", exc_info=True)
            return None


    def _restore_executor(self) -> None:
        """Restores the portfolio's original executor."""
        if hasattr(self.portfolio, '_original_executor'):
            self.portfolio.executor = self.portfolio._original_executor
            # Clean up the temporary attribute
            try:
                 del self.portfolio._original_executor
            except AttributeError:
                 pass # Already deleted or never set properly
            self.logger.info("Restored original portfolio executor.")
        else:
             # Check if the current executor is the multi-executor we set up
             if isinstance(getattr(self.portfolio, 'executor', None), MultiTickerExecutor):
                 self.logger.warning("Could not restore original executor: '_original_executor' attribute missing.")
             # If executor is already None or something else, maybe restoration wasn't needed or failed earlier.
             elif getattr(self.portfolio, 'executor', None) is None:
                 self.logger.info("Portfolio executor was None or already restored.")


    def run(self) -> None:
        """Executes the entire backtest process."""
        self.logger.info("===== Starting Backtest Run =====")
        if not hasattr(self.portfolio, 'portfolio_id') or not hasattr(self.portfolio, 'tickers'):
             self.logger.error("Portfolio object is missing required attributes (portfolio_id, tickers). Aborting.")
             return

        self.logger.info(f"Portfolio ID: {self.portfolio.portfolio_id}")
        self.logger.info(f"Tickers: {self.portfolio.tickers}")

        perf_df = None # Initialize perf_df to None

        try:
            if not self._prepare_data():
                self.logger.error("Backtest aborted due to data preparation failure.")
                return # Exit if data prep fails

            self._setup_executor()
            self._run_event_loop()
            perf_df = self._calculate_results() # Calculate performance dataframe

            if perf_df is not None and not perf_df.empty:
                # --- Generate and Save Report ---
                 generate_backtest_report(
                    portfolio=self.portfolio, # Passes portfolio for tickers, ID, executor access
                    perf_df=perf_df,
                    initial_capital_per_ticker=self.initial_capital_per_ticker
                 )
            else:
                 self.logger.warning("Skipping report generation due to empty or invalid results.")

        except Exception as e:
            self.logger.exception(f"An critical error occurred during the backtest run: {e}", exc_info=True) # Log traceback
        finally:
            # --- Always attempt to restore executor ---
            self._restore_executor()