# portfolios/portfolio_BASE/strategy.py

import os
import time
import math
import logging
from abc import ABC, abstractmethod
import pandas as pd
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
            self.exchange = config_dict.get("EXCH", "NASDAQ")
            self.portfolio_weights = config_dict.get("WEIGHTS", None)  # Optional weights for tickers
            self.data_feeds = config_dict.get("DATA_FEEDS", ["MARKET_DATA", "POSITIONS", "CASH_EQUITY", "PORT_NOTIONAL"])
        else:
            # Fallback if no config provided
            self.portfolio_id = "0"
            self.tickers = []
            self.poll_interval = 1
            self.lookback_days = 1
            self.portfolio_weights = None  # Equal weights by default

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Initialized portfolio {self.portfolio_id} with {len(self.tickers)} tickers.")

        # TODO: if self.portfolio_weights is None: self.portfolio_weights = get_portf_weights_from_db

        self.last_seen = {}

    def run(self):
        """Main polling loop."""
        self.logger.info("Portfolio execution started.")
        while self.running:
            try:
                start_time = time.time()
                data = self.get_data(self.data_feeds)
                if data:
                    self.generate_signals_and_trade(data)
                else:
                    self.logger.info("No new market data.")

                if self.debug:
                    self.logger.info("Debug mode: exiting after one iteration.")
                    break
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.poll_interval - elapsed_time)
                logging.info(f"Trade execution finished in {elapsed_time:.2f}s.")
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                self.logger.warning("Keyboard interrupt received. Exiting.")
                self.running = False
            except Exception as e:
                self.logger.exception(f"Exception during portfolio loop: {e}")

        self.logger.info("Portfolio execution stopped.")

    def get_data(self, data_feeds: List[str]):
        """
        Fetches data from the specified data feeds.
        :param data_feeds: List of data feed names to fetch.
        :return: Dictionary with data feed names as keys and their data as values.
        """
        data = {}
        for feed in data_feeds:
            if feed == "MARKET_DATA":
                data[feed] = self.get_market_data()
            elif feed == "POSITIONS":
                data[feed] = self._get_current_positions(self.portfolio_id)
            elif feed == "CASH_EQUITY":
                data[feed] = self._get_cash_balance(self.portfolio_id)
            elif feed == "PORT_NOTIONAL":
                data[feed] = self._get_portfolio_notional(self.portfolio_id)
        return data

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

    def get_market_data(self):
        """
        Fetch recent market data for portfolio tickers within lookback window.
        Optionally filters out previously seen timestamps.
        """
        end_time = datetime.date.now()
        start_time = end_time - timedelta(days=self.lookback_days)

        placeholders = ', '.join(['%s'] * len(self.tickers))
        sql = f"""
            SELECT *
            FROM market_data
            WHERE ticker IN ({placeholders})
              AND date BETWEEN %s AND %s
        """
        params = [start_time, end_time] + self.tickers
        result = self.db.execute_query(sql, params, fetch=True)

        if result['status'] != 'success':
            self.logger.error(f"DB read failed: {result['message']}")
            return pd.DataFrame()
        if not result['data']:
            market_data = pd.DataFrame(result['data'])
            market_data['timestamp'] = pd.to_datetime(market_data['timestamp'], errors='coerce')
            market_data['close_price'] = market_data.to_numeric(market_data['close_price'], errors='coerce')
            market_data = market_data.dropna(subset=['timestamp', 'ticker', 'close_price'])
            market_data.sort_values('timestamp', inplace=True)
        return market_data
    

    def _get_cash_balance(self, portfolio_id):
        """Retrieve the latest cash balance (notional) for the portfolio."""
        sql_cash = """
            SELECT *
            FROM cash_equity_book
            WHERE portfolio_id = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """
        cash_result = self.db.execute_query(sql_cash, values=(portfolio_id), fetch=True)
        if cash_result['status'] != 'success' or not cash_result['data']:
            logging.error(f"Could not retrieve cash_equity_book for portfolio {portfolio_id}")
            return pd.DataFrame()

        return pd.DataFrame(cash_result)
    

    def _get_portfolio_notional(self, portfolio_id):
        """Retrieve the latest cash balance (notional) for the portfolio."""
        sql_cash = """
            SELECT *
            FROM pnl_book
            WHERE portfolio_id = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """
        portfolio_result = self.db.execute_query(sql_cash, values=(portfolio_id), fetch=True)
        if portfolio_result['status'] != 'success' or not portfolio_result['data']:
            logging.error(f"Could not retrieve cash_equity_book for portfolio {portfolio_id}")
            return pd.DataFrame()
        return pd.DataFrame(portfolio_result) #portfolio_result['data'][0]['notional'])
    
    def _get_current_positions(self, portfolio_id):
        """Retrieve the latest cash balance (notional) for the portfolio."""
        sql_positions = """
            SELECT DISTINCT ON (ticker)
                *
            FROM
                positions_book
            WHERE
                portfolio_id = %s
            ORDER BY
                ticker, timestamp DESC;
        """
        result = self.db.execute_query(sql_positions,values=(portfolio_id), fetch=True)

        if result['status'] != 'success':
            self.logger.error(f"positions read failed: {result['message']}")
            return pd.DataFrame()
        if not result['data']:
            positions_data = pd.DataFrame(result['data'])
        return positions_data

    @abstractmethod
    def generate_signals_and_trade(self, data: Dict[str, pd.DataFrame]):
        """
        Subclasses implement this method for strategy-specific signal generation and trade logic.
        Must call `self.execute_trade(ticker, signal_type, confidence)`.
        """
        pass
