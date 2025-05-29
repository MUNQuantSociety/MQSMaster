# portfolios/portfolio_BASE/strategy.py

import os
import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any

class BasePortfolio(ABC):
    def __init__(self, db_connector, executor, debug=False, config_dict=None):
        """
        Base class for all portfolio strategies.
        :param db_connector: MQSDBConnector instance
        :param executor: a callable for trade execution
        :param debug: if True, runs only once
        :param config_dict: dictionary with config values, e.g.:
              {
                "PORTFOLIO_ID": "02",
                "TICKERS": ["AAPL","TSLA","NVDA"],
                "INTERVAL": 1,
                "LOOKBACK_DAYS": 30
              }
        """
        self.db = db_connector
        self.executor = executor
        self.running = True
        self.debug = debug

        # Either use the passed-in config or default to some placeholders
        if config_dict is not None:
            self.portfolio_id = config_dict.get("PORTFOLIO_ID", "0")
            self.tickers = config_dict.get("TICKERS", [])
            self.poll_interval = config_dict.get("INTERVAL", 1)  # seconds
            self.lookback_days = config_dict.get("LOOKBACK_DAYS", 1)
        else:
            # Fallback if no config provided
            self.portfolio_id = "0"
            self.tickers = []
            self.poll_interval = 1
            self.lookback_days = 1

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Initialized portfolio {self.portfolio_id} with {len(self.tickers)} tickers.")

        # Track last processed timestamp
        self.last_seen = {}

    def run(self):
        """Main polling loop."""
        self.logger.info("Portfolio execution started.")
        while self.running:
            try:
                market_data = self.get_latest_market_data()
                if market_data:
                    self.logger.info(f"Retrieved {len(market_data)} market data rows.")
                    self.generate_signals_and_trade(market_data)
                else:
                    self.logger.info("No new market data.")

                if self.debug:
                    self.logger.info("Debug mode: exiting after one iteration.")
                    break

                time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                self.logger.warning("Keyboard interrupt received. Exiting.")
                self.running = False
            except Exception as e:
                self.logger.exception(f"Exception during portfolio loop: {e}")

        self.logger.info("Portfolio execution stopped.")

    def backtest(self,
                 start_date: Optional[Union[str, datetime]] = None,
                 end_date: Optional[Union[str, datetime]] = None,
                 initial_capital_per_ticker: float = 100000.0): # Make capital configurable
        from data_infra.tradingOps.backtest.runner import BacktestRunner
        """
        Performs a backtest using the BacktestRunner.

        Args:
            start_date: Start date for the backtest.
            end_date: End date for the backtest.
            initial_capital_per_ticker: The starting capital for each ticker's sub-portfolio.
        """
        self.logger.info(f"Initiating backtest for portfolio '{self.portfolio_id}'...")

        # Ensure the portfolio has an executor assigned, even if None initially.
        # The runner will replace it temporarily.
        if not hasattr(self, 'executor'):
             self.executor = None # Or assign a default dummy executor if needed outside backtest

        try:
            # Instantiate the runner
            runner = BacktestRunner(
                portfolio=self,
                start_date=start_date,
                end_date=end_date,
                initial_capital_per_ticker=initial_capital_per_ticker
            )
            # Execute the backtest
            runner.run()

        except Exception as e:
             self.logger.exception(f"Backtest failed for portfolio '{self.portfolio_id}': {e}", exc_info=True)

        self.logger.info(f"Backtest process completed for portfolio '{self.portfolio_id}'. Check logs and report files.")

    def get_latest_market_data(self):
        """
        Fetch recent market data for portfolio tickers within lookback window.
        Optionally filters out previously seen timestamps.
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.lookback_days)

        placeholders = ', '.join(['%s'] * len(self.tickers))
        sql = f"""
            SELECT *
            FROM market_data
            WHERE timestamp BETWEEN %s AND %s
              AND ticker IN ({placeholders})
            ORDER BY ticker, timestamp DESC
        """
        params = [start_time, end_time] + self.tickers
        result = self.db.execute_query(sql, params, fetch=True)

        if result['status'] != 'success':
            self.logger.error(f"DB read failed: {result['message']}")
            return []

        # Optional: filter new timestamps (deduplication logic)
        new_data = []
        for row in result['data']:
            ticker = row['ticker']
            ts = row['timestamp']
            if ticker not in self.last_seen or ts > self.last_seen[ticker]:
                self.last_seen[ticker] = ts
                new_data.append(row)

        return new_data

    def execute_trade(self,
                      ticker: str,
                      signal_type: str,
                      confidence: float,
                      timestamp: Optional[datetime] = None):
        """
        Route trade signal through the executor function.
        Optionally accepts a timestamp for accurate logging during backtests.
        Args:
            ticker: Target ticker symbol.
            signal_type: 'BUY' or 'SELL'.
            confidence: Strategy confidence (0.0 to 1.0).
            timestamp: The simulation time (used for backtest logging). Defaults to None.
        """
        if self.executor is None:
            self.logger.error(f"Cannot execute trade for {ticker}: Executor is not set.")
            return

        signal_type = signal_type.upper()
        if signal_type not in ('BUY', 'SELL'):
            self.logger.error(f"Invalid signal type '{signal_type}' for {ticker}.")
            return

        # Clamp Confidence
        confidence = max(0.0, min(1.0, confidence))

        try:
            # *** MODIFICATION: Pass timestamp keyword argument to executor call ***
            # The executor callable (e.g., MultiTickerExecutor.__call__) MUST also accept **kwargs or `timestamp=None`
            self.executor(self.portfolio_id, ticker, signal_type, confidence, timestamp=timestamp)
            # *** END MODIFICATION ***
        except TypeError as e:
             # Catch TypeError specifically if the underlying executor doesn't accept timestamp yet
             if 'timestamp' in str(e):
                 self.logger.error(f"Executor {type(self.executor)} does not accept 'timestamp' keyword argument. Update executor definition.")
             else:
                  self.logger.exception(f"TypeError during executor call for {ticker} {signal_type}: {e}", exc_info=True)
        except Exception as e:
            self.logger.exception(f"Executor failed for {ticker} {signal_type}: {e}", exc_info=True)
    # --- END: Corrected execute_trade Method ---

    @abstractmethod
    def generate_signals_and_trade(self, market_data: list[dict]):
        """
        Subclasses implement this method for strategy-specific signal generation and trade logic.
        Must call `self.execute_trade(ticker, signal_type, confidence)`.
        """
        pass
