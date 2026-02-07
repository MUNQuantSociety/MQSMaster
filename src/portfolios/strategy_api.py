# src/portfolios/strategy_api.py
import logging
from datetime import datetime
from typing import Optional
import pandas as pd
import pytz as pytz 
# This import registers the .toolkit accessor globally
try:
    from portfolios import toolkit
except ImportError:
    logging.warning("toolkit relative import failed; using absolute import.")
    try:
        from src.portfolios import toolkit
    except ImportError:
        logging.error("Failed to import toolkit from both relative and absolute paths.")
        raise

class AssetData:
    """
    Represents the market data for a single asset at a specific point in time.
    """
    def __init__(self, ticker: str, asset_specific_df: pd.DataFrame, current_time: Optional[datetime]):
        self._ticker = ticker
        self._df = asset_specific_df
        self._time = current_time

        if self._df.empty:
            self.Exists = False
            self._set_defaults()
            return

        effective_time = current_time
        if effective_time is None:
            try:
                effective_time = self._df.index.max()
            except Exception:
                effective_time = None

        # Obtain latest row up to effective_time (or overall last row if None)
        if effective_time is None:
            latest_data = self._df
        else:
            try:
                latest_data = self._df.loc[self._df.index <= effective_time]
            except TypeError:
                latest_data = self._df

        if latest_data.empty:
            self.Exists = False
            self._set_defaults()
            return

        self.latest_row = latest_data.iloc[-1]
        # Safely extract numeric fields; if any core price is missing, mark as non-existent
        def _to_float(val):
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        self.Open = _to_float(self.latest_row.get('open_price'))
        self.High = _to_float(self.latest_row.get('high_price'))
        self.Low = _to_float(self.latest_row.get('low_price'))
        self.Close = _to_float(self.latest_row.get('close_price'))
        self.Volume = _to_float(self.latest_row.get('volume'))
        if self.Close is None:
            self.Exists = False
            self._set_defaults()
            return
        self.Timestamp = self.latest_row.name
        self.Exists = True

    def _set_defaults(self):
        """Helper to set properties to None when no data is available."""
        self.latest_row = None
        self.Open = None
        self.High = None
        self.Low = None
        self.Close = None
        self.Volume = None
        self.Timestamp = None

    def History(self, lookback_period: str) -> pd.DataFrame:
        if self._df.empty:
            return pd.DataFrame()
        # Use effective time: if _time is None, use the latest timestamp in the data
        end_date = self._time
        if end_date is None:
            try:
                end_date = self._df.index.max()
            except Exception:
                return pd.DataFrame()


        try:
            start_date = end_date - pd.to_timedelta(lookback_period)
        except Exception:
            return pd.DataFrame()

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
        self.cash = float(cash)
        self.total_value = float(total_value)

        if positions_df is not None and not positions_df.empty:
            self.positions = dict(zip(positions_df['ticker'], positions_df['quantity']))
        else:
            self.positions = {}

    def get_asset_value(self, ticker: str, current_price: float) -> float:
        quantity = float(self.positions.get(ticker, 0.0))
        return quantity * current_price

    def get_asset_weight(self, ticker: str, current_price: float) -> float:
        total_val = float(self.total_value)
        if total_val == 0:
            return 0.0
        asset_value = float(self.get_asset_value(ticker, current_price))
        return asset_value / total_val

    def __repr__(self) -> str:
        return f"PortfolioManager(TotalValue={self.total_value:,.2f}, Cash={self.cash:,.2f}, Positions={len(self.positions)})"


class StrategyContext:
    """
    The master context object passed to the strategy's OnData method on each time step.
    It encapsulates MarketData, PortfolioManager, and provides trade execution methods"""
    def __init__(self, market_data_df, cash_df, positions_df, port_notional_df, current_time, executor, portfolio_config):
        self._executor = executor
        self._portfolio_config = portfolio_config
        self._positions_df = positions_df
        effective_time = current_time
        timezone = pytz.timezone('America/New_York')
        if effective_time is None:
            if market_data_df is not None and not getattr(market_data_df, 'empty', True) and 'timestamp' in market_data_df.columns:
                try:
                    effective_time = pd.to_datetime(market_data_df['timestamp']).max()
                except Exception:
                    effective_time = datetime.now(timezone)
            else:
                effective_time = datetime.now(timezone)
        self.time = effective_time

        # Initialize the high-level helper classes
        self.Market = MarketData(market_data_df, effective_time)

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
            logging.warning(
                "Skip trade: no valid market data for %s at %s (Exists=%s, Close=%s)",
                ticker,
                self.time,
                asset_data.Exists,
                asset_data.Close,
            )
            return

        self._executor.execute_trade(
            portfolio_id=self._portfolio_config['id'],
            ticker=ticker,
            signal_type=signal_type,
            confidence=confidence,
            arrival_price=asset_data.Close,
            cash=self.Portfolio.cash,
            positions=self._positions_df,
            port_notional=self.Portfolio.total_value,
            ticker_weight=self.Portfolio.get_asset_weight(ticker, asset_data.Close),
            timestamp=self.time
        )
