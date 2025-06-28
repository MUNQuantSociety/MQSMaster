# Backtest/runner.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union

# --- NEW IMPORT ---
from tqdm.auto import tqdm

from .reporting import generate_backtest_report
from .utils import fetch_historical_data
from .executor import BacktestExecutor
from portfolios.portfolio_BASE.strategy import BasePortfolio


class BacktestRunner:
    """
    Orchestrates the execution of a multi-ticker backtest using a unified
    portfolio model to ensure accuracy.
    """
    def __init__(self,
                 portfolio: 'BasePortfolio',
                 start_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
                 end_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
                 initial_capital: float = 100000.0):
        """
        Initializes the BacktestRunner.
        """
        self.portfolio = portfolio
        self.logger = portfolio.logger
        self.total_start_capital = initial_capital
        self.start_date = self._ensure_datetime(start_date)
        self.end_date = self._ensure_datetime(end_date, default_is_yesterday=True)

        lookback_days = getattr(self.portfolio, 'lookback_days', 365)
        self.strategy_lookback_window = pd.Timedelta(days=lookback_days)
        self.logger.info(f"Using strategy lookback window of {lookback_days} days.")

        if self.start_date is None:
            if self.end_date:
                 self.start_date = self.end_date - timedelta(days=365 * 2)
            else:
                 self.logger.error("Cannot determine start_date as end_date is invalid.")

        self.perf_records: List[Dict] = []
        self.main_data_df: pd.DataFrame = pd.DataFrame()
        self.executor: Optional[BacktestExecutor] = None


    def _ensure_datetime(self, dt_val, default_is_yesterday=False) -> Optional[datetime]:
        """Converts input to a timezone-naive datetime object at midnight."""
        if dt_val is None:
            if default_is_yesterday:
                try:
                    today = datetime.now().date()
                    yesterday = today - timedelta(days=1)
                    return datetime(yesterday.year, yesterday.month, yesterday.day)
                except Exception as e:
                     self.logger.error(f"Error getting yesterday's date: {e}")
                     return None
            return None
        if isinstance(dt_val, datetime):
            return datetime(dt_val.year, dt_val.month, dt_val.day)
        if hasattr(dt_val, 'year') and not isinstance(dt_val, datetime):
            return datetime(dt_val.year, dt_val.month, dt_val.day)
        try:
            pd_dt = pd.to_datetime(dt_val, errors='coerce')
            if pd.isna(pd_dt):
                self.logger.warning(f"Could not parse '{dt_val}' as a datetime.")
                return None
            return datetime(pd_dt.year, pd_dt.month, pd_dt.day)
        except Exception as e:
            self.logger.error(f"Failed to convert '{dt_val}' to datetime: {e}")
            return None


    def _prepare_data(self) -> bool:
        """Fetches, cleans, sorts, and prepares historical market data."""
        if not self.start_date or not self.end_date:
                self.logger.error("Invalid start or end date for data preparation.")
                return False

        df = fetch_historical_data(self.portfolio, self.start_date, self.end_date)
        if df.empty:
            self.logger.error("No historical data found for the specified criteria.")
            return False

        try:
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
            self.main_data_df = df
            self.logger.info(f"Data prepared: {len(self.main_data_df)} rows loaded.")
            return True
        except Exception as e:
            self.logger.exception(f"Error during data preparation: {e}", exc_info=True)
            return False


    def _setup_executor(self) -> None:
        """Sets up the new unified BacktestExecutor."""
        self.executor = BacktestExecutor(
            initial_capital=self.total_start_capital,
            tickers=self.portfolio.tickers
        )
        self.portfolio._original_executor = getattr(self.portfolio, 'executor', None)
        self.portfolio.executor = self.executor


    def _run_event_loop(self) -> None:
        """
        Runs the main backtest simulation loop with a progress bar.
        """
        if self.main_data_df.empty or self.executor is None:
            self.logger.error("Cannot run event loop: Data or executor not ready.")
            return

        self.logger.info("Starting backtest event loop...")

        poll_td = pd.Timedelta(seconds=self.portfolio.poll_interval)
        timestamps_series = self.main_data_df['timestamp']
        self.perf_records = []
        last_poll_time: Optional[pd.Timestamp] = None

        data_groups = self.main_data_df.groupby('timestamp', sort=True)

        # --- PROGRESS BAR IMPLEMENTATION ---
        # Wrap the main loop's iterable with tqdm to show progress.
        progress_bar = tqdm(data_groups, total=len(data_groups), desc="Running Backtest", unit=" steps", leave=True)

        for current_timestamp, current_data_chunk in progress_bar:
            if last_poll_time and (current_timestamp - last_poll_time) < poll_td:
                continue
            last_poll_time = current_timestamp

            price_updates = dict(zip(current_data_chunk['ticker'], current_data_chunk['close_price']))
            for ticker, price in price_updates.items():
                if pd.notna(price):
                    self.executor.update_price(ticker, float(price))

            window_start_time = current_timestamp - self.strategy_lookback_window
            start_index = timestamps_series.searchsorted(window_start_time, side='left')
            end_index = current_data_chunk.index.max()
            
            historical_slice_df = self.main_data_df.iloc[start_index : end_index + 1]

            if not historical_slice_df.empty:
                try:
                    sim_time = current_timestamp.to_pydatetime()
                    data_dict = self.executor.get_data_feeds()
                    data_dict['MARKET_DATA'] = historical_slice_df
                    self.portfolio.generate_signals_and_trade(data_dict, current_time=sim_time)
                except Exception as e:
                    self.logger.exception(f"Error in strategy at {current_timestamp}: {e}", exc_info=True)

            record = {'timestamp': current_timestamp}
            for ticker in self.portfolio.tickers:
                record[ticker] = self.executor.get_position_value(ticker)
            record['portfolio_value'] = self.executor.get_port_notional()
            self.perf_records.append(record)

        self.logger.info("Event loop finished.")


    def _calculate_results(self) -> Optional[pd.DataFrame]:
        """Calculates performance metrics from recorded data."""
        if not self.perf_records:
            self.logger.warning("No performance records generated during the backtest.")
            return None
        try:
            perf_df = pd.DataFrame(self.perf_records)
            perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'])
            perf_df.sort_values('timestamp', inplace=True)
            perf_df.reset_index(drop=True, inplace=True)

            value_cols = list(self.portfolio.tickers) + ['portfolio_value']
            for col in value_cols:
                 if col in perf_df.columns:
                      perf_df[col] = pd.to_numeric(perf_df[col], errors='coerce')

            if self.total_start_capital > 0:
                perf_df['pnl_pct'] = (perf_df['portfolio_value'] - self.total_start_capital) / self.total_start_capital
            else:
                perf_df['pnl_pct'] = 0.0

            self.logger.info("Performance DataFrame calculated.")
            return perf_df
        except Exception as e:
            self.logger.exception(f"Error calculating results DataFrame: {e}", exc_info=True)
            return None


    def _restore_executor(self) -> None:
        """Restores the portfolio's original executor."""
        if hasattr(self.portfolio, '_original_executor'):
            self.portfolio.executor = self.portfolio._original_executor
            try:
                 del self.portfolio._original_executor
            except AttributeError:
                 pass
            self.logger.info("Restored original portfolio executor.")
        else:
             if isinstance(getattr(self.portfolio, 'executor', None), BacktestExecutor):
                 self.logger.warning("Could not restore original executor: '_original_executor' attribute missing.")
             elif getattr(self.portfolio, 'executor', None) is None:
                 self.logger.info("Portfolio executor was None or already restored.")


    def run(self) -> None:
        """Executes the entire backtest process."""
        self.logger.info("===== Starting Backtest Run =====")
        if not hasattr(self.portfolio, 'portfolio_id') or not hasattr(self.portfolio, 'tickers'):
             self.logger.error("Portfolio object is missing required attributes. Aborting.")
             return

        self.logger.info(f"Portfolio ID: {self.portfolio.portfolio_id}")
        self.logger.info(f"Tickers: {self.portfolio.tickers}")

        perf_df = None

        try:
            if not self._prepare_data():
                self.logger.error("Backtest aborted due to data preparation failure.")
                return

            self._setup_executor()
            self._run_event_loop()
            perf_df = self._calculate_results()

            if perf_df is not None and not perf_df.empty:
                 generate_backtest_report(
                    portfolio=self.portfolio,
                    perf_df=perf_df,
                    initial_capital=self.total_start_capital,
                    full_historical_data=self.main_data_df
                 )
            else:
                 self.logger.warning("Skipping report generation due to empty or invalid results.")

        except Exception as e:
            self.logger.exception(f"An critical error occurred during the backtest run: {e}", exc_info=True)
        finally:
            self._restore_executor()
