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
            raise ValueError("config_dict is unreadable for portfolio configuration.")

        # Logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Initialized portfolio {self.portfolio_id} with {len(self.tickers)} tickers.")

        # TODO: if self.portfolio_weights is None: self.portfolio_weights = get_portf_weights_from_db

        self.last_seen = {}

    @abstractmethod
    def generate_signals_and_trade(self, data: Dict[str, pd.DataFrame], current_time: Optional[datetime] = None):
        """
        Subclasses implement this method for strategy-specific signal generation and trade logic.
        Must call `self.executor.execute_trade(...)`.
        
        Args:
            data: A dictionary containing dataframes for different feeds like 'MARKET_DATA'.
            current_time: The timestamp for the current event, used primarily for backtesting.
                          In live trading, this can be None.
        """
        pass

    def get_data(self, data_feeds: List[str]) -> Dict[str, pd.DataFrame]:
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
        


    def get_market_data(self) -> pd.DataFrame:
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
            WHERE ticker IN ({placeholders})
              AND date BETWEEN %s AND %s
        """
        # Note: The order of parameters must match the query
        params = self.tickers + [start_time, end_time]
        result = self.db.execute_query(sql, params, fetch=True)

        if result['status'] != 'success':
            self.logger.error(f"DB read failed: {result['message']}")
            return pd.DataFrame()
        
        if not result['data']:
            return pd.DataFrame()

        market_data = pd.DataFrame(result['data'])
        market_data['timestamp'] = pd.to_datetime(market_data['timestamp'], errors='coerce')
        market_data['close_price'] = pd.to_numeric(market_data['close_price'], errors='coerce')
        market_data = market_data.dropna(subset=['timestamp', 'ticker', 'close_price'])
        market_data.sort_values('timestamp', inplace=True)
        return market_data
    
    def _get_cash_balance(self, portfolio_id: str) -> pd.DataFrame:
        """Retrieve the latest cash balance (notional) for the portfolio."""
        sql_cash = """
            SELECT *
            FROM cash_equity_book
            WHERE portfolio_id = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """
        cash_result = self.db.execute_query(sql_cash, values=(portfolio_id,), fetch=True)
        if cash_result['status'] != 'success' or not cash_result['data']:
            logging.error(f"Could not retrieve cash_equity_book for portfolio {portfolio_id}")
            return pd.DataFrame()
        return pd.DataFrame(cash_result['data'])
    
    def _get_portfolio_notional(self, portfolio_id: str) -> pd.DataFrame:
        """
        Retrieve the latest portfolio notional value.
        If no entry exists in pnl_book (first live run), fall back to cash_equity_book.
        """
        sql_pnl = """
            SELECT *
            FROM pnl_book
            WHERE portfolio_id = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """
        pnl_result = self.db.execute_query(sql_pnl, values=(portfolio_id,), fetch=True)

        # If we got a successful row, return it directly
        if pnl_result.get('status') == 'success' and pnl_result.get('data'):
            return pd.DataFrame(pnl_result['data'])

        # Otherwise—first run or error—fall back to cash_equity
        logging.info(f"No pnl_book entry for portfolio {portfolio_id}; initializing notional from cash_equity_book.")
        cash_df = self._get_cash_balance(portfolio_id)
        if not cash_df.empty and 'notional' in cash_df.columns:
            # Return only the notional column (with its timestamp, if present)
            return cash_df[['timestamp', 'notional']].copy()
        
        # Last fallback: zero notional
        logging.warning(f"cash_equity_book also empty for portfolio {portfolio_id}; returning zero notional.")
        return pd.DataFrame([{
            'timestamp': datetime.now(),
            'notional': 0.0
        }])
    

    def _get_current_positions(self, portfolio_id: str) -> pd.DataFrame:
        """
        Retrieve the latest position for each ticker in the portfolio.
        If no rows exist in positions_book for this portfolio, initialize
        quantity=0 for each ticker (letting defaults fill other columns) and
        return those.
        """
        # 1) Try to read existing positions
        sql_positions = """
            SELECT DISTINCT ON (ticker)
                position_id,
                portfolio_id,
                ticker,
                quantity,
                updated_at
            FROM positions_book
            WHERE portfolio_id = %s
            ORDER BY ticker, updated_at DESC;
        """
        result = self.db.execute_query(sql_positions, values=(portfolio_id,), fetch=True)
        if result['status'] != 'success':
            self.logger.error(f"Positions read failed: {result['message']}")
            return pd.DataFrame()

        # 2) If truly empty *for this portfolio*, seed zero-quantity rows
        if not result.get('data'):
            self.logger.info(f"No positions for portfolio {portfolio_id}; inserting zero-quantity baseline.")
            insert_sql = """
                INSERT INTO positions_book
                    (portfolio_id, ticker, quantity)
                VALUES
                    (%s, %s, %s)
                RETURNING position_id, portfolio_id, ticker, quantity, updated_at;
            """
            seeded_rows = []
            for t in self.tickers:
                try:
                    res = self.db.execute_query(insert_sql, values=(portfolio_id, t, 0), fetch=True)
                    if res['status'] == 'success' and res.get('data'):
                        # fetch=True on INSERT with RETURNING yields data list of one dict
                        seeded_rows.append(res['data'][0])
                    else:
                        self.logger.error(f"Failed to seed position for {t}: {res.get('message')}")
                except Exception as e:
                    self.logger.exception(f"Exception while seeding zero position for {t}: {e}")
            if not seeded_rows:
                self.logger.error("Seeding zero-quantity positions failed for all tickers.")
                return pd.DataFrame()
            return pd.DataFrame(seeded_rows)

        # 3) Otherwise, build DataFrame of what we fetched
        df = pd.DataFrame(result['data'])

        # 4) In the rare case some tickers are still missing, append them with zero
        missing = set(self.tickers) - set(df['ticker'])
        if missing:
            self.logger.info(f"Missing positions for {missing}; inserting zero-quantity baseline for those.")
            insert_sql = """
                INSERT INTO positions_book
                    (portfolio_id, ticker, quantity)
                VALUES
                    (%s, %s, %s)
                RETURNING position_id, portfolio_id, ticker, quantity, updated_at;
            """
            for t in missing:
                try:
                    res = self.db.execute_query(insert_sql, values=(portfolio_id, t, 0), fetch=True)
                    if res['status'] == 'success' and res.get('data'):
                        df = pd.concat([df, pd.DataFrame(res['data'])], ignore_index=True)
                    else:
                        self.logger.error(f"Failed to seed missing ticker {t}: {res.get('message')}")
                except Exception as e:
                    self.logger.exception(f"Exception while seeding missing ticker {t}: {e}")

        # 5) Finally, return only the columns our strategies expect
        return df[['position_id', 'portfolio_id', 'ticker', 'quantity', 'updated_at']]