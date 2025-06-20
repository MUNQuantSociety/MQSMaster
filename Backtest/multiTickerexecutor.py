import pandas as pd
import logging
from typing import Optional
from datetime import datetime
from .singleTickerexecutor import SingleTickerExecutor  # Adjust import as needed
#from .ba import SingleTickerExecutor


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
        self.executors: dict[str, SingleTickerExecutor] = {
            t: SingleTickerExecutor(initial_capital=self.initial_capital_per_ticker)
            for t in tickers
        }
        self.logger.info(f"Initialized MultiTickerExecutor for tickers: {tickers}")

    def execute_trade(self,
                      portfolio_id: str,
                      ticker: str,
                      signal_type: str,
                      confidence: float,
                      arrival_price: float,
                      cash: float,
                      positions: float,
                      port_notional: float,
                      ticker_weight: float,
                      timestamp: Optional[datetime] = None):
        """
        Route the multi‐param call from your strategy to the single‐ticker executors,
        passing all arguments along.
        """
        if ticker not in self.executors:
            self.logger.warning(f"Received trade for untracked ticker '{ticker}', ignoring.")
            return

        self.logger.debug(
            f"Routing {signal_type.upper()} for {ticker} — "
            f"conf={confidence:.2f}, price={arrival_price:.2f}, cash={cash:.2f}, "
            f"pos={positions:.2f}, port_notional={port_notional:.2f}, weight={ticker_weight:.2f}"
        )

        # Forward all params exactly as strategy expects
        self.executors[ticker].execute_trade(
            portfolio_id=portfolio_id,
            ticker=ticker,
            signal_type=signal_type,
            confidence=confidence,
            arrival_price=arrival_price,
            cash=cash,
            positions=positions,
            port_notional=port_notional,
            ticker_weight=ticker_weight,
            timestamp=timestamp
        )

    def update_price(self, ticker: str, price: float):
        if ticker not in self.executors:
            self.logger.debug(f"Price update for untracked ticker '{ticker}', ignoring.")
            return
        try:
            self.executors[ticker].latest_price = float(price)
            self.logger.debug(f"{ticker} price updated to {price:.2f}")
        except Exception as e:
            self.logger.error(f"Failed to update price for {ticker}: {e}")

    # ... rest of class unchanged ...
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
    

    def get_cash_equity_df(self) -> pd.DataFrame:
        """
        Returns per–ticker cash & equity (position × price) as a DataFrame with columns:
        ['ticker','cash','equity','notional']
        """
        rows = []
        for t, ex in self.executors.items():
            cash   = ex.cash
            equity = ex.quantity * ex.latest_price
            rows.append({
                'ticker':   t,
                'cash':     cash,
                'equity':   equity,
                'notional': cash + equity
            })
        return pd.DataFrame(rows)

    def get_positions_df(self) -> pd.DataFrame:
        """Returns each ticker’s current quantity as a DataFrame ['ticker','quantity']."""
        return pd.DataFrame([
            {'ticker': t, 'quantity': ex.quantity}
            for t, ex in self.executors.items()
        ])

    def get_port_notional_df(self) -> pd.DataFrame:
        """
        Returns a single‐row DataFrame ['notional'] = sum of all cash+equity.
        """
        total = sum(ex.cash + ex.quantity * ex.latest_price 
                    for ex in self.executors.values())
        return pd.DataFrame([{'notional': total}])