# portfolios/portfolio_BASE/strategy.py

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

class BasePortfolio(ABC):
    """
    Base class for all portfolio strategies. It now fetches core portfolio state
    (cash and positions) atomically to prevent data consistency issues.
    """

    def __init__(self, db_connector, executor, debug=False, config_dict=None):
        self.db = db_connector
        self.executor = executor
        self.running = True
        self.debug = debug

        if config_dict is None:
            raise ValueError("config_dict is required for portfolio configuration.")

        self.portfolio_id = config_dict.get("PORTFOLIO_ID", "0")
        self.tickers = config_dict.get("TICKERS", [])
        self.poll_interval = config_dict.get("INTERVAL", 60)
        self.lookback_days = config_dict.get("LOOKBACK_DAYS", 30)
        self.portfolio_weights = config_dict.get("WEIGHTS")
        self.data_feeds = config_dict.get("DATA_FEEDS", ["MARKET_DATA", "POSITIONS", "CASH_EQUITY", "PORT_NOTIONAL"])
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")
        self.logger.info(f"Initialized portfolio {self.portfolio_id} with {len(self.tickers)} tickers.")
    
    # --- SQL Query Constants for Maintainability ---
    
    # Unified query to fetch cash and positions in a single, atomic operation.
    # This prevents race conditions where cash and positions could be out of sync.
    ATOMIC_STATE_QUERY = """
    WITH latest_cash AS (
        SELECT *
        FROM cash_equity_book
        WHERE portfolio_id = %s
        ORDER BY timestamp DESC, id DESC
        LIMIT 1
    ),
    latest_positions AS (
        SELECT DISTINCT ON (ticker)
            position_id, portfolio_id, ticker, quantity, updated_at
        FROM positions_book
        WHERE portfolio_id = %s
        ORDER BY ticker, updated_at DESC
    )
    SELECT
        (SELECT row_to_json(lc) FROM latest_cash lc) AS cash_data,
        (SELECT json_agg(lp) FROM latest_positions lp) AS positions_data;
    """

    MARKET_DATA_QUERY = """
        SELECT *
        FROM market_data
        WHERE ticker IN ({placeholders})
          AND timestamp BETWEEN %s AND %s
    """

    LATEST_PNL_QUERY = """
        SELECT *
        FROM pnl_book
        WHERE portfolio_id = %s
        ORDER BY timestamp DESC
        LIMIT 1
    """
    
    SEED_POSITION_QUERY = """
        INSERT INTO positions_book (portfolio_id, ticker, quantity)
        VALUES (%s, %s, 0)
        RETURNING *;
    """
    @abstractmethod
    def generate_signals_and_trade(self, data: Dict[str, pd.DataFrame], current_time: Optional[datetime] = None):
        """
        Subclasses implement this method for strategy-specific signal generation and trade logic.
        """
        pass

    def get_data(self, data_feeds: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetches a consistent snapshot of portfolio data. Core state (cash, positions)
        is fetched atomically to prevent race conditions.
        """
        data = {feed: pd.DataFrame() for feed in data_feeds}

        # --- 1. Atomic Fetch for Core Portfolio State ---
        try:
            params = (self.portfolio_id, self.portfolio_id)
            state_result = self.db.execute_query(self.ATOMIC_STATE_QUERY, params, fetch='one')

            if state_result['status'] == 'success' and state_result.get('data'):
                result_data = state_result['data'][0]
                if "CASH_EQUITY" in data_feeds and result_data.get('cash_data'):
                    data["CASH_EQUITY"] = pd.DataFrame([result_data['cash_data']])
                if "POSITIONS" in data_feeds and result_data.get('positions_data'):
                    data["POSITIONS"] = pd.DataFrame(result_data['positions_data'])
        except Exception as e:
            self.logger.exception(f"Failed to fetch atomic state for portfolio {self.portfolio_id}: {e}")

        # --- 2. Seed Missing Positions (if needed) ---
        if "POSITIONS" in data_feeds:
            existing_tickers = set(data["POSITIONS"]['ticker']) if not data["POSITIONS"].empty else set()
            missing_tickers = set(self.tickers) - existing_tickers
            if missing_tickers:
                self._seed_missing_positions(data["POSITIONS"], missing_tickers)

        # --- 3. Fetch Non-Critical Data Separately ---
        if "MARKET_DATA" in data_feeds:
            data["MARKET_DATA"] = self._get_market_data()
        
        if "PORT_NOTIONAL" in data_feeds:
            data["PORT_NOTIONAL"] = self._get_portfolio_notional(fallback_cash_df=data.get("CASH_EQUITY"))

        return data

    def _seed_missing_positions(self, positions_df: pd.DataFrame, missing_tickers: set):
        """Helper to insert zero-quantity rows for tickers without a position record."""
        self.logger.info(f"Seeding zero-quantity positions for missing tickers: {missing_tickers}")
        seeded_rows = []
        for ticker in missing_tickers:
            try:
                res = self.db.execute_query(self.SEED_POSITION_QUERY, (self.portfolio_id, ticker), fetch=True)
                if res.get('data'):
                    seeded_rows.extend(res['data'])
            except Exception as e:
                self.logger.exception(f"Exception while seeding position for {ticker}: {e}")
        
        if seeded_rows:
            seeded_df = pd.DataFrame(seeded_rows)
            return pd.concat([positions_df, seeded_df], ignore_index=True)
        return positions_df

    def _get_market_data(self) -> pd.DataFrame:
        """Fetches recent market data for all portfolio tickers."""
        if not self.tickers:
            return pd.DataFrame()

        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.lookback_days)
        
        placeholders = ', '.join(['%s'] * len(self.tickers))
        sql = self.MARKET_DATA_QUERY.format(placeholders=placeholders)
        params = self.tickers + [start_time.date(), end_time.date()]
        
        result = self.db.execute_query(sql, params, fetch='all')

        if result['status'] != 'success' or not result.get('data'):
            return pd.DataFrame()

        df = pd.DataFrame(result['data'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['close_price'] = pd.to_numeric(df['close_price'])
        df.dropna(subset=['timestamp', 'ticker', 'close_price'], inplace=True)
        df.sort_values('timestamp', inplace=True)
        return df
    
    def _get_portfolio_notional(self, fallback_cash_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Retrieves the latest portfolio notional. Falls back to cash if no PnL record exists.
        """
        pnl_result = self.db.execute_query(self.LATEST_PNL_QUERY, (self.portfolio_id,), fetch='one')

        if pnl_result.get('status') == 'success' and pnl_result.get('data'):
            return pd.DataFrame(pnl_result['data'])

        self.logger.info(f"No pnl_book entry for portfolio {self.portfolio_id}; using cash balance as notional.")
        if fallback_cash_df is not None and not fallback_cash_df.empty:
            return fallback_cash_df[['timestamp', 'notional']].copy()
        
        self.logger.warning(f"Fallback cash is also empty for portfolio {self.portfolio_id}; returning zero notional.")
        return pd.DataFrame([{'timestamp': datetime.now(), 'notional': 0.0}])