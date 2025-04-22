# data_infra/tradingOps/backtest/runner.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union

from .multiTickerExecutor import MultiTickerExecutor
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
                 initial_capital_per_ticker: float = 100000.0):
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
        self.initial_capital_per_ticker = initial_capital_per_ticker
        self.start_date = self._ensure_datetime(start_date)
        self.end_date = self._ensure_datetime(end_date, default_is_yesterday=True)

        # Default start_date if None
        if self.start_date is None:
            # Default to 2 years before end_date if end_date is valid
            if self.end_date:
                 self.start_date = self.end_date - timedelta(days=365 * 2)
            else:
                 # Handle case where end_date could also not be determined
                 self.logger.error("Cannot determine start_date as end_date is invalid.")
                 # Or raise an error, depending on desired behavior
                 # raise ValueError("Cannot determine backtest dates.")


        self.perf_records: List[Dict] = []
        self.main_data_df: pd.DataFrame = pd.DataFrame()
        self.multi_executor: Optional[MultiTickerExecutor] = None

        # Ensure tickers is iterable before calculating total_start_capital
        if hasattr(self.portfolio, 'tickers') and self.portfolio.tickers:
             self.total_start_capital = len(self.portfolio.tickers) * self.initial_capital_per_ticker
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
        self.logger.info(f"Data range: {self.start_date.strftime('%Y-%m-%d')} -> {self.end_date.strftime('%Y-%m-%d')}")

        df = fetch_historical_data(self.portfolio, self.start_date, self.end_date)
        if df.empty:
            self.logger.error("No historical data found for the specified criteria.")
            return False

        # --- Data Preprocessing ---
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # Convert numeric columns, coercing errors to NaN
            numeric_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
            for col in numeric_cols:
                 if col in df.columns:
                      df[col] = pd.to_numeric(df[col], errors='coerce')

            initial_rows = len(df)
            # Drop rows where essential data is missing AFTER conversion attempts
            df = df.dropna(subset=['timestamp', 'close_price', 'ticker'])
            if len(df) < initial_rows:
                self.logger.warning(f"Dropped {initial_rows - len(df)} rows due to NaNs in essential columns.")

            # Filter strictly within the requested date range (inclusive)
            df = df[(df['timestamp'] >= self.start_date) & (df['timestamp'] <= self.end_date)]

            if df.empty:
                self.logger.error("No data remains after filtering for the requested range or cleaning.")
                return False

            # Sort data globally by timestamp - CRITICAL for sequential processing
            df.sort_values('timestamp', inplace=True)
            # *** Ensure default RangeIndex for iloc performance ***
            df.reset_index(drop=True, inplace=True)

            self.main_data_df = df
            self.logger.info(f"Data prepared: {len(self.main_data_df)} rows loaded.")
            self.logger.debug(f"Data sample:\n{self.main_data_df.head()}")
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
        """Runs the main backtest simulation loop using iloc optimization."""
        if self.main_data_df.empty or self.multi_executor is None:
            self.logger.error("Cannot run event loop: Data or executor not ready.")
            return

        # --- OPTIMIZATION: Pre-calculate max index for each timestamp ---
        self.logger.info("Pre-calculating timestamp index map...")
        timestamp_to_max_index: Optional[pd.Series] = None # Initialize
        try:
            # Check if timestamp column exists and is datetime-like
            if 'timestamp' in self.main_data_df.columns and pd.api.types.is_datetime64_any_dtype(self.main_data_df['timestamp']):
                 # Group by timestamp once and find the max index in the original DataFrame for each group
                 grouped_by_time = self.main_data_df.groupby('timestamp', sort=True)
                 # Create a Series: timestamp -> max_original_index
                 timestamp_to_max_index = grouped_by_time.apply(lambda x: x.index.max())
                 # Ensure the map itself is sorted by timestamp (index of the Series)
                 if not timestamp_to_max_index.index.is_monotonic_increasing:
                     timestamp_to_max_index = timestamp_to_max_index.sort_index()
                 self.logger.info("Timestamp index map calculated successfully.")
            else:
                 self.logger.error("Timestamp column missing or not datetime type. Cannot pre-calculate index map.")

        except Exception as e:
             self.logger.exception(f"Failed to pre-calculate index map: {e}. Will find index within loop (slower).", exc_info=True)
             timestamp_to_max_index = None # Ensure fallback if error occurs
        # --- End Pre-calculation ---

        # Determine unique timestamps, preferably from the pre-calculated map
        try:
             if timestamp_to_max_index is not None and not timestamp_to_max_index.empty:
                  unique_times = timestamp_to_max_index.index
             elif 'timestamp' in self.main_data_df.columns:
                  unique_times = np.sort(self.main_data_df['timestamp'].unique())
             else:
                   self.logger.error("Cannot determine unique timestamps.")
                   return # Cannot proceed

             total_timestamps = len(unique_times)
             if total_timestamps == 0:
                  self.logger.warning("No unique timestamps found in data range. Event loop will not run.")
                  return
             self.logger.info(f"Starting event loop over {total_timestamps} unique timestamps...")
        except Exception as e:
             self.logger.exception(f"Error determining unique timestamps: {e}", exc_info=True)
             return


        poll_td = pd.Timedelta(seconds=self.portfolio.poll_interval)
        last_poll_time: Optional[datetime] = None
        self.perf_records = [] # Reset records
        logged_milestones = set()
        milestones = { # Recalculate milestones based on actual unique_times count
            int(total_timestamps * 0.25): "25%", int(total_timestamps * 0.50): "50%",
            int(total_timestamps * 0.75): "75%", total_timestamps: "100%"
        }
        last_known_max_index = -1 # Track previous index for efficient chunk slicing

        # --- Loop starts ---
        for i, current_timestamp in enumerate(unique_times): # Iterate directly over sorted unique timestamps
            step_number = i + 1

            # --- Poll Interval Check ---
            # Ensure current_timestamp is comparable (it should be datetime from map index or unique())
            if last_poll_time is not None and isinstance(current_timestamp, datetime):
                 time_diff = current_timestamp - last_poll_time
                 if time_diff < poll_td:
                      continue # Skip this timestamp if interval not met
            # Update last poll time only if it's a valid datetime
            if isinstance(current_timestamp, datetime):
                 last_poll_time = current_timestamp

            # --- OPTIMIZATION: Get current_max_index efficiently ---
            current_max_index: int = -1 # Use type hint
            try:
                if timestamp_to_max_index is not None:
                    # Direct lookup in the pre-calculated map (Series)
                    current_max_index = timestamp_to_max_index.loc[current_timestamp]
                else:
                    # Fallback: Find index within the loop (slower)
                    # This requires getting the chunk first, which is less efficient
                    chunk_df_for_index = self.main_data_df[self.main_data_df['timestamp'] == current_timestamp]
                    if not chunk_df_for_index.empty:
                         current_max_index = chunk_df_for_index.index.max()

                if not isinstance(current_max_index, (int, np.integer)) or current_max_index < 0:
                     raise ValueError(f"Invalid index found: {current_max_index}")

            except KeyError:
                 self.logger.warning(f"Timestamp {current_timestamp} not found in pre-calculated index map (KeyError).")
                 current_max_index = -1 # Ensure it's marked as invalid
            except Exception as e:
                 self.logger.error(f"Error determining max index for timestamp {current_timestamp}: {e}")
                 current_max_index = -1 # Ensure it's marked as invalid

            if current_max_index == -1:
                 self.logger.warning(f"Skipping step for timestamp {current_timestamp} due to index lookup failure.")
                 continue
            # ---

            # --- Update Prices (Needs data for the current timestamp only) ---
            # Efficiently get the chunk for price updates using iloc
            start_index = last_known_max_index + 1
            # Slice from start_index up to and including current_max_index
            chunk_df = self.main_data_df.iloc[start_index : current_max_index + 1]

            for _, row in chunk_df.iterrows(): # Iterate over the small chunk
                ticker = row.get('ticker')
                price_val = row.get('close_price')
                # Check if ticker is tracked and price is valid before updating
                if ticker and ticker in self.multi_executor.executors and pd.notna(price_val):
                    try:
                        self.multi_executor.update_price(ticker, float(price_val))
                    except (ValueError, TypeError) as e:
                         self.logger.error(f"Error updating price for {ticker} with value {price_val}: {e}")

            # --- OPTIMIZATION: Create historical slice using iloc ---
            # Slice up to and including current_max_index
            # Use copy() to avoid potential SettingWithCopyWarning in strategy if it modifies the slice
            historical_slice_df = self.main_data_df.iloc[:current_max_index + 1].copy()
            # --- END OPTIMIZATION ---

            # --- Call Strategy ---
            if not historical_slice_df.empty:
                try:
                     # Ensure current_timestamp is passed as datetime
                     sim_time = pd.to_datetime(current_timestamp).to_pydatetime() if not isinstance(current_timestamp, datetime) else current_timestamp
                     self.portfolio.generate_signals_and_trade(
                         market_data=historical_slice_df,
                         current_time=sim_time
                     )
                except Exception as e:
                     self.logger.exception(f"Error occurred within strategy generate_signals_and_trade at {current_timestamp}: {e}", exc_info=True)
                     # Optionally: decide whether to continue or stop the backtest on strategy error


            # --- Record Portfolio State ---
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
                         record[t] = 'Error' # Indicate error in record
                else:
                    record[t] = 0.0 # Ticker not managed by executor
            record['portfolio_value'] = total_portfolio_value
            self.perf_records.append(record)

            # --- Progress Logging Check ---
            for milestone_step, percentage in milestones.items():
                if step_number >= milestone_step and percentage not in logged_milestones:
                    self.logger.info(f"Backtest progress: {percentage} completed ({step_number}/{total_timestamps} steps).")
                    logged_milestones.add(percentage)
            # ---

            # Update index tracker for next iteration's chunk slicing
            last_known_max_index = current_max_index

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
            self.logger.info("===== Backtest Run Finished =====")