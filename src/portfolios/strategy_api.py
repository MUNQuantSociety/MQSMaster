# src/portfolios/strategy_api.py
import logging
from datetime import datetime
from typing import Optional
import pandas as pd
import pytz as pytz 
# This import registers the .toolkit accessor globally.
# Prefer the `src.*` path so MarketData is the same class object as imported
# elsewhere (tests, src/main_backtest.py); falling through to the bare
# `portfolios.*` path otherwise would create a duplicate MarketData class
# and break `isinstance` checks.
try:
    from src.portfolios import toolkit
    from src.portfolios.market_data_api import MarketData
except ImportError:
    try:
        from portfolios import toolkit
        from portfolios.market_data_api import MarketData
    except ImportError:
        logging.error("Failed to import toolkit from both src.* and bare paths.")
        raise

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
