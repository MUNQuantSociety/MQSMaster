# src/portfolios/portfolio_BASE/strategy.py

import importlib
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from src.portfolios.indicators.base import Indicator
from src.portfolios.strategy_api import StrategyContext


def _camel_to_snake(name: str) -> str:
    """Converts a CamelCase string to snake_case for dynamic module loading."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class BasePortfolio(ABC):
    """
    Base class for all portfolio strategies, featuring a dynamic, stateful
    indicator manager and the StrategyContext API.
    """

    def __init__(self, db_connector, executor, debug=False, config_dict=None, backtest_start_date: Optional[datetime] = None):
        """
        Initializes the base portfolio, loading configuration.
        """
        self.db = db_connector
        self.executor = executor
        self.running = True
        self.debug = debug
        self.backtest_start_date = backtest_start_date

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

        self.portfolio_config_dict = {
            "id": self.portfolio_id,
            "tickers": self.tickers,
            "weights": self.portfolio_weights,
            "poll_interval": self.poll_interval,
            "lookback_days": self.lookback_days
        }
        
        # --- Indicator Management ---
        self._indicators: List[Indicator] = []

    # --- DYNAMIC INDICATOR FACTORY ---
    def AddIndicator(self, indicator_class_name: str, ticker: str, **kwargs) -> Indicator:
        """
        Dynamically loads, instantiates, warms up, and registers an indicator.
        This is the scalable factory for all indicators.

        Args:
            indicator_class_name (str): The CamelCase name of the indicator class (e.g., "SimpleMovingAverage").
            ticker (str): The ticker the indicator should run on.
            **kwargs: Keyword arguments for the indicator (e.g., period=50).

        Returns:
            An instance of the requested indicator, fully warmed-up and ready to use.
        """
        if ticker not in self.tickers:
            raise ValueError(f"Ticker '{ticker}' is not part of this portfolio's universe.")

        try:
            module_name = _camel_to_snake(indicator_class_name)
            module = importlib.import_module(f"src.portfolios.indicators.{module_name}")
            indicator_class = getattr(module, indicator_class_name)
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Could not dynamically load indicator '{indicator_class_name}'. "
                              f"Ensure the file 'src/portfolios/indicators/{_camel_to_snake(indicator_class_name)}.py' "
                              f"and class '{indicator_class_name}' exist. Details: {e}")
            raise

        # Instantiate the indicator
        indicator = indicator_class(ticker=ticker, **kwargs)

        # Warm-up logic
        # Fetch more data than the period to ensure enough data points, accounting for non-trading days.
        warmup_days = int(kwargs.get('period', 20) * 1.7) 
        end_time = self.backtest_start_date or datetime.now()
        start_time = end_time - timedelta(days=warmup_days)
        
        sql = self.MARKET_DATA_QUERY.format(placeholders='%s')
        params = [ticker, start_time.date(), end_time.date()]
        result = self.db.execute_query(sql, params, fetch='all')

        price_col = kwargs.get('price_col', 'close_price')
        if result['status'] == 'success' and result.get('data'):
            df = pd.DataFrame(result['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce').dropna()
            df.sort_values('timestamp', inplace=True)
            
            for row in df.itertuples():
                indicator.Update(row.timestamp, getattr(row, price_col))
        
        self._indicators.append(indicator)
        return indicator
    
    def RegisterIndicatorSet(self, indicator_definitions: Dict[str, tuple]):
        """
        Initializes a set of indicators for every ticker and attaches them as
        ticker-keyed dictionaries to the strategy instance. This is the
        recommended way to reduce __init__ boilerplate.

        Args:
            indicator_definitions (dict): A dictionary where keys are the desired
                attribute names (e.g., 'fast_sma') and values are a tuple of
                (IndicatorClassName, {**kwargs}).

                Example:
                {
                    "fast_sma": ("SimpleMovingAverage", {"period": 10}),
                    "slow_sma": ("SimpleMovingAverage", {"period": 30})
                }
        """
        for attr_name, (class_name, kwargs) in indicator_definitions.items():
            # Create a dictionary to hold this indicator for all tickers
            indicator_dict = {
                ticker: self.AddIndicator(class_name, ticker=ticker, **kwargs)
                for ticker in self.tickers
            }
            
            # Attach the completed dictionary as an attribute (e.g., self.fast_sma)
            setattr(self, attr_name, indicator_dict)
            self.logger.info(f"Registered indicator set '{attr_name}' for all tickers.")


    def generate_signals_and_trade(self, data: Dict[str, pd.DataFrame], current_time: Optional[datetime] = None):
        """
        (Framework-Internal Method)
        Updates indicators, constructs the context, and calls the user's OnData method.
        """
        market_data_df = data.get('MARKET_DATA')
        if market_data_df is not None and not market_data_df.empty:
            # Get the single latest row for each ticker in the current data slice
            latest_data_points = market_data_df.sort_values('timestamp').groupby('ticker').last()
            for indicator in self._indicators:
                if indicator.ticker in latest_data_points.index:
                    latest_row = latest_data_points.loc[indicator.ticker]
                    price_col = getattr(indicator, 'price_col', 'close_price')
                    if price_col in latest_row and pd.notna(latest_row[price_col]):
                        timestamp = latest_row['timestamp']
                        value = latest_row[price_col]
                        indicator.Update(timestamp, value)

        context = StrategyContext(
            market_data_df=market_data_df,
            cash_df=data.get('CASH_EQUITY'),
            positions_df=data.get('POSITIONS'),
            port_notional_df=data.get('PORT_NOTIONAL'),
            current_time=current_time,
            executor=self.executor,
            portfolio_config=self.portfolio_config_dict
        )

        self.OnData(context)

    @abstractmethod
    def OnData(self, context: StrategyContext):
        """
        (User-Facing Method)
        This is the primary method that all user-defined strategies must implement.
        It is called by the framework on each time step (or polling interval)
        and provides a powerful context object with all necessary market and
        portfolio information and tools.

        Args:
            context (StrategyContext): The stateful API object for this point in time.
        """
        pass

    # --- Data Fetching Logic (Largely Unchanged) ---
    # The methods below are still required for the base class to function,
    # as it's responsible for fetching the data that will eventually be
    # passed into the StrategyContext.

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

    def get_data(self, data_feeds: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetches a consistent snapshot of portfolio data. Core state (cash, positions)
        is fetched atomically to prevent race conditions.
        """
        data = {feed: pd.DataFrame() for feed in data_feeds}

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

        if "POSITIONS" in data_feeds:
            existing_tickers = set(data["POSITIONS"]['ticker']) if not data["POSITIONS"].empty else set()
            missing_tickers = set(self.tickers) - existing_tickers
            if missing_tickers:
                # The _seed_missing_positions method returns a new DataFrame
                data["POSITIONS"] = self._seed_missing_positions(data["POSITIONS"], missing_tickers)

        if "MARKET_DATA" in data_feeds:
            data["MARKET_DATA"] = self._get_market_data()
        
        if "PORT_NOTIONAL" in data_feeds:
            data["PORT_NOTIONAL"] = self._get_portfolio_notional(fallback_cash_df=data.get("CASH_EQUITY"))

        return data

    def _seed_missing_positions(self, positions_df: pd.DataFrame, missing_tickers: set) -> pd.DataFrame:
        """Helper to insert zero-quantity rows for tickers without a position record."""
        self.logger.info(f"Seeding zero-quantity positions for missing tickers: {missing_tickers}")
        seeded_rows = []
        for ticker in missing_tickers:
            try:
                res = self.db.execute_query(self.SEED_POSITION_QUERY, (self.portfolio_id, ticker), fetch='all')
                if res.get('data'):
                    seeded_rows.extend(res['data'])
            except Exception as e:
                self.logger.exception(f"Exception while seeding position for {ticker}: {e}")
        
        if seeded_rows:
            seeded_df = pd.DataFrame(seeded_rows)
            # Ensure columns match before concatenating to avoid issues
            if not positions_df.empty:
                 seeded_df = seeded_df[positions_df.columns.intersection(seeded_df.columns)]
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
        # Add other price columns to numeric conversion for robustness
        for col in ['open_price', 'high_price', 'low_price', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
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