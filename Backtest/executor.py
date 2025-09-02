# backtest/executor.py

import pandas as pd
import logging
import math
from datetime import datetime
from typing import Optional, Dict, List
from collections import defaultdict

class BacktestExecutor:
    """
    A backtest executor that manages a single, unified portfolio.
    It perfectly mirrors the "Direct Fractional Order" logic of the live 
    tradeExecutor to ensure 1:1 backtest accuracy.
    """

    def __init__(self, initial_capital: float, tickers: List[str]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tickers = tickers
        
        # --- Unified Portfolio State ---
        self.cash = initial_capital
        self.positions: Dict[str, float] = {ticker: 0.0 for ticker in tickers}
        self.latest_prices: Dict[str, float] = {ticker: 0.0 for ticker in tickers}
        self.trade_log: List[Dict] = []
        
        self.logger.info(
            f"BacktestExecutor initialized with {initial_capital:.2f} capital "
            f"for tickers: {tickers}"
        )

    def update_price(self, ticker: str, price: float):
        """Updates the latest known price for a ticker."""
        if ticker in self.latest_prices:
            self.latest_prices[ticker] = price

    def get_port_notional(self) -> float:
        """Calculates the total current value of the portfolio."""
        positions_value = sum(
            self.positions[ticker] * self.latest_prices.get(ticker, 0.0)
            for ticker in self.tickers
        )
        return self.cash + positions_value

    def get_position_value(self, ticker: str) -> float:
        """
        Calculates the notional value of a single ticker's position.
        """
        return self.positions.get(ticker, 0.0) * self.latest_prices.get(ticker, 0.0)

    def get_data_feeds(self) -> Dict[str, pd.DataFrame]:
        """
        Generates the portfolio state dataframes required by the strategy.
        """
        cash_df = pd.DataFrame([{'notional': self.cash}])
        positions_list = [
            {'ticker': ticker, 'quantity': quantity}
            for ticker, quantity in self.positions.items()
        ]
        positions_df = pd.DataFrame(positions_list)
        port_notional_df = pd.DataFrame([{'notional': self.get_port_notional()}])

        return {
            'CASH_EQUITY': cash_df,
            'POSITIONS': positions_df,
            'PORT_NOTIONAL': port_notional_df
        }

    def get_trade_logs(self) -> Dict[str, List[Dict]]:
        """
        Returns trade logs grouped by ticker for reporting compatibility.
        """
        grouped_logs = defaultdict(list)
        for log_entry in self.trade_log:
            grouped_logs[log_entry['ticker']].append(log_entry)
        return dict(grouped_logs)

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
        signal_type = signal_type.upper()
        if signal_type not in ('BUY', 'SELL', 'HOLD'):
            return

        confidence = max(0.0, min(1.0, confidence))
        if signal_type == 'HOLD' or confidence == 0.0:
            return
            
        exec_price = self.latest_prices.get(ticker, 0.0)
        if exec_price <= 0:
            return

        current_quantity = self.positions.get(ticker, 0.0)
        current_notional_value = current_quantity * exec_price
        max_target_notional = port_notional * ticker_weight
        
        direct_order_notional = max_target_notional * confidence

        quantity_to_trade = 0

        if signal_type == 'BUY':
            final_trade_notional = min(direct_order_notional, self.cash)
            room_before_cap = max(0, max_target_notional - current_notional_value)
            final_trade_notional = min(final_trade_notional, room_before_cap)
            quantity_to_trade = math.floor(final_trade_notional / exec_price)
            
            if quantity_to_trade > 0:
                self.cash -= (quantity_to_trade * exec_price)
                self.positions[ticker] += quantity_to_trade

        elif signal_type == 'SELL':
            final_trade_notional = min(direct_order_notional, current_notional_value)
            quantity_to_trade = math.floor(final_trade_notional / exec_price)

            if quantity_to_trade > 0:
                self.cash += (quantity_to_trade * exec_price)
                self.positions[ticker] -= quantity_to_trade
        
        if quantity_to_trade > 0:
            self.trade_log.append({
                "timestamp": timestamp,
                "portfolio_id": portfolio_id,
                "ticker": ticker,
                "signal_type": signal_type,
                "confidence": confidence,
                "shares": quantity_to_trade,
                "fill_price": exec_price,
                "cash_after": self.cash
            })
            return {'status': 'success', 'quantity': quantity_to_trade, 'updated_cash': self.cash}