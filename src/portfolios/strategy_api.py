# src/portfolios/strategy_api.py

from datetime import datetime
import pandas as pd
# This import registers the .toolkit accessor globally
from src.portfolios import toolkit 

class AssetData:
    """
    Represents the market data for a single asset at a specific point in time.
    This version is optimized to work with pre-filtered DataFrames.
    """
    def __init__(self, ticker: str, asset_specific_df: pd.DataFrame, current_time: datetime):
        self._ticker = ticker
        self._df = asset_specific_df
        self._time = current_time

        if not self._df.empty:
            # Efficiently get the latest row up to the current time
            latest_data = self._df.loc[self._df.index <= current_time]
            if not latest_data.empty:
                self.latest_row = latest_data.iloc[-1]
                self.Exists = True
                self.Open = float(self.latest_row['open_price'])
                self.High = float(self.latest_row['high_price'])
                self.Low = float(self.latest_row['low_price'])
                self.Close = float(self.latest_row['close_price'])
                self.Volume = float(self.latest_row['volume'])
                self.Timestamp = self.latest_row.name # The timestamp is now the index
            else:
                self.Exists = False
                self._set_defaults()
        else:
            self.Exists = False
            self._set_defaults()

    def _set_defaults(self):
        """Helper to set properties to None when no data is available."""
        self.latest_row = None
        self.Open = self.High = self.Low = self.Close = self.Volume = self.Timestamp = None

    def History(self, lookback_period: str) -> pd.DataFrame:
        if not self.Exists:
            return pd.DataFrame()

        end_date = self._time
        start_date = end_date - pd.to_timedelta(lookback_period)
        
        # Filter the already asset-specific DataFrame, which is much faster
        hist_df = self._df.loc[(self._df.index >= start_date) & (self._df.index <= end_date)]
        return hist_df.copy()

    def __repr__(self) -> str:
        if not self.Exists:
            return f"AssetData(ticker='{self._ticker}', Exists=False)"
        return f"AssetData(ticker='{self._ticker}', Time='{self.Timestamp}', Close={self.Close})"


class MarketData:
    """
    A high-level and performant interface for accessing all market data.
    It pre-processes the data by grouping it by ticker upon initialization.
    """
    def __init__(self, market_data_df: pd.DataFrame, current_time: datetime):
        self._time = current_time
        self._cache = {}

        if market_data_df is not None and not market_data_df.empty:
            # PERFORMANCE OPTIMIZATION: Set timestamp as index and group by ticker ONCE.
            df = market_data_df.set_index('timestamp')
            self._grouped_data = dict(list(df.groupby('ticker')))
            self._unique_tickers = set(self._grouped_data.keys())
        else:
            self._grouped_data = {}
            self._unique_tickers = set()

    def __getitem__(self, ticker: str) -> AssetData:
        if ticker in self._cache:
            return self._cache[ticker]

        asset_specific_df = self._grouped_data.get(ticker, pd.DataFrame())
        asset = AssetData(ticker, asset_specific_df, self._time)
        self._cache[ticker] = asset
        return asset

    def __contains__(self, ticker: str) -> bool:
        return ticker in self._unique_tickers


class PortfolioManager:
    """
    Provides a clean, high-level interface to the current state of the portfolio.
    """
    def __init__(self, cash: float, total_value: float, positions_df: pd.DataFrame):
        self.cash = cash
        self.total_value = total_value

        # Ensure it is a valid empty DataFrame if None is passed, so .empty check never fails
        self.positions_df = positions_df if positions_df is not None else pd.DataFrame()

        if not self.positions_df.empty:
            self.positions = dict(zip(self.positions_df['ticker'], self.positions_df['quantity']))
        else:
            self.positions = {}

    def get_asset_value(self, ticker: str, current_price: float) -> float:
        quantity = self.positions.get(ticker, 0.0)
        return quantity * current_price

    def get_asset_weight(self, ticker: str, current_price: float) -> float:
        if self.total_value == 0:
            return 0.0
        asset_value = self.get_asset_value(ticker, current_price)
        return asset_value / self.total_value

    def __repr__(self) -> str:
        return f"PortfolioManager(TotalValue={self.total_value:,.2f}, Cash={self.cash:,.2f}, Positions={len(self.positions)})"


class StrategyContext:
    """
    The master context object passed to the strategy's OnData method on each time step.
    """
    def __init__(self, market_data_df, cash_df, positions_df, port_notional_df, current_time, executor, portfolio_config):
        self._executor = executor
        self._portfolio_config = portfolio_config
        self.time = current_time

        # Initialize the high-level helper classes
        self.Market = MarketData(market_data_df, current_time)

        cash_val = cash_df.iloc[0]['notional'] if cash_df is not None and not cash_df.empty else 0.0
        port_val = port_notional_df.iloc[0]['notional'] if port_notional_df is not None and not port_notional_df.empty else 0.0

        self.Portfolio = PortfolioManager(
            cash=cash_val,
            total_value=port_val,
            positions_df=positions_df
        )

    def buy(self, ticker: str, confidence: float = 1.0):
        self._trade(ticker, 'BUY', confidence)

    def sell(self, ticker: str, confidence: float = 1.0):
        self._trade(ticker, 'SELL', confidence)

    def _trade(self, ticker: str, signal_type: str, confidence: float):
        asset_data = self.Market[ticker]
        if not asset_data.Exists or asset_data.Close is None or asset_data.Close <= 0:
            print(f"Warning: Cannot trade {ticker}. No valid market data at {self.time}.")
            return

        self._executor.execute_trade(
            portfolio_id=self._portfolio_config['id'],
            ticker=ticker,
            signal_type=signal_type,
            confidence=confidence,
            arrival_price=asset_data.Close,
            cash=self.Portfolio.cash,
            # The executor expects a DataFrame to check buying power/exposure logic.
            positions=self.Portfolio.positions_df,
            port_notional=self.Portfolio.total_value,
            ticker_weight=self._portfolio_config['weights'].get(ticker, 0.0),
            timestamp=self.time
        )