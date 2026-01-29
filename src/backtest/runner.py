from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from zoneinfo import ZoneInfo  # <-- ADDED for timezone fix

import pandas as pd
from tqdm.auto import tqdm

from src.portfolios.portfolio_BASE.strategy import BasePortfolio

from .executor import BacktestExecutor
from .reporting import generate_backtest_report
from .utils import fetch_historical_data

# Define the exchange timezone
NY_TZ = ZoneInfo("America/New_York")


class BacktestRunner:
    """
    Orchestrates the execution of a multi-ticker backtest using a unified
    portfolio model to ensure accuracy.
    """

    def __init__(
        self,
        portfolio: "BasePortfolio",
        start_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
        end_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
        initial_capital: float = 100000.0,
        slippage: float = 0.0,
    ):
        """
        Initializes the BacktestRunner.
        """
        self.portfolio = portfolio
        self.logger = portfolio.logger
        self.total_start_capital = initial_capital

        # FIX 3: Use new timezone-aware method
        self.start_date = self._ensure_datetime(start_date)
        self.end_date = self._ensure_datetime(end_date, default_is_yesterday=True)

        # --- FIX 1: Save the *actual* backtest start date ---
        self.backtest_loop_start_date = self.start_date
        # --- END FIX 1 ---

        self.slippage = slippage

        lookback_days = getattr(self.portfolio, "lookback_days", 365)
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

    def _ensure_datetime(
        self, dt_val, default_is_yesterday=False
    ) -> Optional[datetime]:
        """
        FIX 3: Converts input to a timezone-AWARE datetime object at midnight
        in 'America/New_York'.
        """
        if dt_val is None:
            if default_is_yesterday:
                try:
                    ny_now = datetime.now(NY_TZ)
                    yesterday = (ny_now - timedelta(days=1)).date()
                    return datetime(
                        yesterday.year, yesterday.month, yesterday.day, tzinfo=NY_TZ
                    )
                except Exception as e:
                    self.logger.error(f"Error getting yesterday's date: {e}")
                    return None
            return None

        try:
            pd_dt = pd.to_datetime(dt_val, errors="coerce")
            if pd.isna(pd_dt):
                self.logger.warning(f"Could not parse '{dt_val}' as a datetime.")
                return None

            # Create naive datetime at midnight, then localize to NY
            naive_dt = datetime(pd_dt.year, pd_dt.month, pd_dt.day)
            # Use replace() to correctly handle DST changes
            return naive_dt.replace(tzinfo=NY_TZ)
        except Exception as e:
            self.logger.error(f"Failed to convert '{dt_val}' to datetime: {e}")
            return None

    def _prepare_data(self) -> bool:
        """
        Fetches, cleans, sorts, and prepares historical market data.
        (This function is now correct)
        """

        if not self.start_date or not self.end_date:
            self.logger.error("Invalid start or end date for data preparation.")
            return False

        # This is your correct "cold start" fix for the *data query*
        lookback_days = getattr(self.portfolio, "lookback_days", None)
        if lookback_days:
            adjusted_start = self.start_date - pd.Timedelta(days=lookback_days)
            # This 'self.start_date' is now only used for the *query*
            self.start_date = adjusted_start
            self.logger.info(
                f"Adjusted data query Start Date to {self.start_date} to include lookback_days={lookback_days}"
            )

        df = fetch_historical_data(self.portfolio, self.start_date, self.end_date)
        if df.empty:
            self.logger.error("No historical data found for the specified criteria.")
            return False

        try:
            df.sort_values("timestamp", inplace=True)
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
            tickers=self.portfolio.tickers,
            slippage=self.slippage,
        )
        self.portfolio._original_executor = getattr(self.portfolio, "executor", None)
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

        # This series is built from the *full* dataframe, so lookups are correct
        timestamps_series = self.main_data_df["timestamp"]
        self.perf_records = []
        last_poll_time: Optional[pd.Timestamp] = None

        # --- FIX 2: Filter the timestamps we iterate over ---

        # 1. Group the *full* dataframe once for efficient lookups
        data_groups = self.main_data_df.groupby("timestamp", sort=True)

        # 2. Get all unique timestamps, which are already sorted
        all_timestamps = self.main_data_df["timestamp"].unique()

        # 3. Filter them to start *only* from the intended backtest start date
        loop_timestamps = all_timestamps[
            all_timestamps >= self.backtest_loop_start_date
        ]
        if len(loop_timestamps) == 0:
            self.logger.error(
                f"No data found on or after the intended start date: {self.backtest_loop_start_date}"
            )
            return
        # --- END FIX 2 ---

        # Wrap the *filtered* timestamps with tqdm
        progress_bar = tqdm(
            loop_timestamps,
            total=len(loop_timestamps),
            desc="Running Backtest",
            unit=" steps",
            leave=True,
        )

        for current_timestamp in progress_bar:  # <-- Iterate over filtered timestamps
            if last_poll_time and (current_timestamp - last_poll_time) < poll_td:
                continue
            last_poll_time = current_timestamp

            # Get the data chunk for this timestamp from the *full* group
            try:
                current_data_chunk = data_groups.get_group(current_timestamp)
            except KeyError:
                continue  # Should not happen, but safe to check

            price_updates = dict(
                zip(current_data_chunk["ticker"], current_data_chunk["close_price"])
            )
            for ticker, price in price_updates.items():
                if pd.notna(price):
                    self.executor.update_price(ticker, float(price))

            # This logic now works perfectly:
            # current_timestamp is (e.g.) `2025-01-02 04:30:00`
            # window_start_time is `2024-10-04 04:30:00` (i.e., 90 days ago)
            window_start_time = current_timestamp - self.strategy_lookback_window

            # start_index will search the *full* timestamps_series and find the correct index in 2024
            start_index = timestamps_series.searchsorted(window_start_time, side="left")

            # end_index is the absolute index of the current bar
            end_index = current_data_chunk.index.max()

            if start_index < 0:
                start_index = 0

            # This slice is now correct: [data_from_90_days_ago ... data_from_today]
            historical_slice_df = self.main_data_df.iloc[start_index : end_index + 1]

            if not historical_slice_df.empty:
                try:
                    sim_time = current_timestamp.to_pydatetime()
                    data_dict = self.executor.get_data_feeds()
                    data_dict["MARKET_DATA"] = historical_slice_df
                    self.portfolio.generate_signals_and_trade(
                        data_dict, current_time=sim_time
                    )
                except Exception as e:
                    self.logger.exception(
                        f"Error in strategy at {current_timestamp}: {e}", exc_info=True
                    )

            record = {"timestamp": current_timestamp}
            for ticker in self.portfolio.tickers:
                record[ticker] = self.executor.get_position_value(ticker)
            record["portfolio_value"] = self.executor.get_port_notional()
            self.perf_records.append(record)

        self.logger.info("Event loop finished.")
        self.executor.dump_trade_log()

    def _calculate_results(self) -> Optional[pd.DataFrame]:
        """Calculates performance metrics from recorded data."""
        if not self.perf_records:
            self.logger.warning("No performance records generated during the backtest.")
            return None
        try:
            perf_df = pd.DataFrame(self.perf_records)
            perf_df["timestamp"] = pd.to_datetime(perf_df["timestamp"])
            perf_df.sort_values("timestamp", inplace=True)
            perf_df.reset_index(drop=True, inplace=True)

            value_cols = list(self.portfolio.tickers) + ["portfolio_value"]
            for col in value_cols:
                if col in perf_df.columns:
                    perf_df[col] = pd.to_numeric(perf_df[col], errors="coerce")

            if self.total_start_capital > 0:
                perf_df["pnl_pct"] = (
                    perf_df["portfolio_value"] - self.total_start_capital
                ) / self.total_start_capital
            else:
                perf_df["pnl_pct"] = 0.0

            self.logger.info("Performance DataFrame calculated.")
            return perf_df
        except Exception as e:
            self.logger.exception(
                f"Error calculating results DataFrame: {e}", exc_info=True
            )
            return None

    def _restore_executor(self) -> None:
        """Restores the portfolio's original executor."""
        if hasattr(self.portfolio, "_original_executor"):
            self.portfolio.executor = self.portfolio._original_executor
            try:
                del self.portfolio._original_executor
            except AttributeError:
                pass
            self.logger.info("Restored original portfolio executor.")
        else:
            if isinstance(getattr(self.portfolio, "executor", None), BacktestExecutor):
                self.logger.warning(
                    "Could not restore original executor: '_original_executor' attribute missing."
                )
            elif getattr(self.portfolio, "executor", None) is None:
                self.logger.info("Portfolio executor was None or already restored.")

    def run(self) -> None:
        """Executes the entire backtest process."""
        self.logger.info("===== Starting Backtest Run =====")
        if not hasattr(self.portfolio, "portfolio_id") or not hasattr(
            self.portfolio, "tickers"
        ):
            self.logger.error(
                "Portfolio object is missing required attributes. Aborting."
            )
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
                    full_historical_data=self.main_data_df,
                )
            else:
                self.logger.warning(
                    "Skipping report generation due to empty or invalid results."
                )

        except Exception as e:
            self.logger.exception(
                f"An critical error occurred during the backtest run: {e}",
                exc_info=True,
            )
        finally:
            self._restore_executor()
