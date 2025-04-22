# data_infra/tradingOps/backtest/multi_ticker_executor.py

import logging
# Import necessary types
from typing import Optional
from datetime import datetime
# Correct relative import assuming singleTickerExecutor is in the same directory
from .singleTickerExecutor import SingleTickerExecutor


class MultiTickerExecutor:
    """
    A backtest executor that treats each ticker as an independent sub-portfolio,
    each with its own starting capital (e.g. $100k).
    """
    def __init__(self, tickers: list[str], initial_capital_per_ticker: float = 100000.0):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tickers = tickers
        self.initial_capital_per_ticker = initial_capital_per_ticker

        # Create one SingleTickerExecutor per ticker
        self.executors: dict[str, SingleTickerExecutor] = {}
        for t in tickers:
            self.executors[t] = SingleTickerExecutor(initial_capital=self.initial_capital_per_ticker)
        self.logger.info(f"MultiTickerExecutor initialized for {len(tickers)} tickers.")

    # *** MODIFIED: Add optional timestamp argument ***
    def __call__(self,
                 portfolio_id: str,
                 ticker: str,
                 signal_type: str,
                 confidence: float,
                 timestamp: Optional[datetime] = None): # Added timestamp
        """
        The portfolio calls self.executor(...) for trades.
        We route that call to the sub-executor for the specific ticker, passing the timestamp.
        """
        if ticker not in self.executors:
            self.logger.warning(f"Trade signal for untracked ticker {ticker}. Ignoring.")
            return

        # *** MODIFIED: Pass timestamp down to SingleTickerExecutor ***
        self.executors[ticker].execute_trade(
            portfolio_id, ticker, signal_type, confidence, timestamp=timestamp
        )

    def update_price(self, ticker: str, price: float):
        """
        Let the sub-executor for that ticker know the new price.
        """
        if ticker in self.executors:
            # Ensure price is a float
            try:
                 self.executors[ticker].latest_price = float(price)
            except (ValueError, TypeError):
                 self.logger.error(f"Invalid price format '{price}' for ticker {ticker}. Cannot update.")
        # else:
             # Optional: Log if price update is for an untracked ticker
             # self.logger.debug(f"Received price update for untracked ticker {ticker}.")


    def get_portfolio_value(self) -> float:
        """ Aggregated portfolio value across all tickers. """
        total = 0.0
        for ticker, executor in self.executors.items():
            try:
                 total += executor.get_portfolio_value()
            except Exception as e:
                 self.logger.error(f"Error getting portfolio value for ticker {ticker}: {e}")
        return total

    def get_ticker_values(self) -> dict[str, float]:
        """ Returns a dict of { ticker: value } for each sub-portfolio. """
        d = {}
        for ticker, executor in self.executors.items():
             try:
                 d[ticker] = executor.get_portfolio_value()
             except Exception as e:
                 self.logger.error(f"Error getting value for ticker {ticker}: {e}")
                 d[ticker] = 0.0 # Default value on error?
        return d

    def get_trade_logs(self) -> dict[str, list]:
        """ Returns a dictionary mapping each ticker to its list of trade log entries. """
        logs = {}
        for ticker, executor in self.executors.items():
             # Ensure trade_log exists and is a list
             log_data = getattr(executor, 'trade_log', None)
             if isinstance(log_data, list):
                  logs[ticker] = log_data
             else:
                  self.logger.warning(f"Trade log not found or invalid for ticker {ticker}.")
                  logs[ticker] = []
        return logs