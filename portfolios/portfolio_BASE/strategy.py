import os
import time
import logging
import inspect
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from portfolios.common import read_config_param

class BasePortfolio(ABC):
    def __init__(self, db_connector, executor, debug=False):
        """
        Base class for all portfolio strategies.
        Handles:
        - Config loading
        - Real-time polling of market_data
        - Ticker filtering
        - Trade execution routing

        :param db_connector: MQSDBConnector instance
        :param executor: Callable in format (portfolio_id, ticker, signal_type, confidence)
        :param debug: If True, runs only once for testing
        """
        self.db = db_connector
        self.executor = executor
        self.running = True
        self.debug = debug

        # Load parameters from local config
        self.tickers = read_config_param("TICKERS")
        self.portfolio_id = read_config_param("PORTFOLIO_ID")
        self.poll_interval = read_config_param("INTERVAL")  # seconds
        self.lookback_days = read_config_param("LOOKBACK_DAYS")

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Initialized portfolio {self.portfolio_id} with {len(self.tickers)} tickers.")

        # Track last processed timestamp per ticker (optional optimization)
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

    def get_latest_market_data(self):
        """
        Fetch recent market data for portfolio tickers within lookback window.
        Optionally filters out previously seen timestamps.
        """
        end_time = datetime.utcnow()
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
        result = self.db.read_db(sql=sql, values=params)

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

    def execute_trade(self, ticker: str, signal_type: str, confidence: float):
        """
        Route trade signal through the executor function.
        :param ticker: Target ticker
        :param signal_type: 'BUY', 'SELL', etc.
        :param confidence: Float between 0 and 1
        """
        try:
            self.executor(self.portfolio_id, ticker, signal_type.upper(), confidence)
            self.logger.info(f"Trade executed: {signal_type.upper()} {ticker} (confidence={confidence:.2f})")
        except Exception as e:
            self.logger.error(f"Execution failed for {ticker}: {e}")

    @abstractmethod
    def generate_signals_and_trade(self, market_data: list[dict]):
        """
        Subclasses implement this method for strategy-specific signal generation and trade logic.
        """
        pass
