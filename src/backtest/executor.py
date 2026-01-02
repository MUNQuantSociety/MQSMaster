import logging
import math
from typing import Dict, List

import pandas as pd


class BacktestExecutor:
    """
    A backtest executor that manages a single, unified portfolio,
    supporting long/short positions with a realistic margin model that mirrors live trading constraints.
    """

    def __init__(
        self,
        initial_capital: float,
        tickers: List[str],
        leverage: float = 2.0,
        slippage: float = 0.0,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tickers = tickers
        self.leverage = leverage
        self.slippage = slippage

        # --- Unified Portfolio State ---
        self.cash = initial_capital
        self.positions: Dict[str, float] = {ticker: 0.0 for ticker in tickers}
        self.latest_prices: Dict[str, float] = {ticker: 0.0 for ticker in tickers}
        self.trade_log: List[Dict] = []

        self.logger.info(
            f"BacktestExecutor initialized with {initial_capital:.2f} capital, "
            f"leverage={leverage}, slippage={slippage}, for tickers: {tickers}"
        )

    def _apply_slippage(self, price: float, signal_type: str) -> float:
        """
        Applies slippage to the execution price based on the trade direction.
        - For BUY orders, the price is increased.
        - For SELL orders, the price is decreased.
        """
        if signal_type == "BUY":
            return price * (1 + self.slippage)
        elif signal_type == "SELL":
            return price * (1 - self.slippage)
        return price

    def update_price(self, ticker: str, price: float):
        """Updates the latest known price for a ticker."""
        if ticker in self.latest_prices:
            self.latest_prices[ticker] = price

    def get_port_notional(self) -> float:
        """Calculates the total current equity of the portfolio."""
        positions_value = sum(
            self.positions[ticker] * self.latest_prices.get(ticker, 0.0)
            for ticker in self.tickers
        )
        return self.cash + positions_value

    def get_position_value(self, ticker: str) -> float:
        """Calculates the notional value of a single ticker's position."""
        return self.positions.get(ticker, 0.0) * self.latest_prices.get(ticker, 0.0)

    def get_data_feeds(self) -> Dict[str, pd.DataFrame]:
        """Generates the portfolio state dataframes required by the strategy."""
        cash_df = pd.DataFrame([{"notional": self.cash}])
        positions_list = [
            {"ticker": ticker, "quantity": quantity}
            for ticker, quantity in self.positions.items()
        ]
        positions_df = pd.DataFrame(positions_list)
        port_notional_df = pd.DataFrame([{"notional": self.get_port_notional()}])

        return {
            "CASH_EQUITY": cash_df,
            "POSITIONS": positions_df,
            "PORT_NOTIONAL": port_notional_df,
        }

    def _calculate_buying_power(self, portfolio_equity: float) -> float:
        """Calculates the available buying power based on a margin model."""
        gross_position_value = sum(
            abs(self.positions[ticker] * self.latest_prices.get(ticker, 0.0))
            for ticker in self.tickers
        )
        buying_power = (portfolio_equity * self.leverage) - gross_position_value
        return max(0, buying_power)

    def execute_trade(
        self,
        portfolio_id,
        ticker,
        signal_type,
        confidence,
        arrival_price,
        cash,
        positions,
        port_notional,
        ticker_weight,
        timestamp,
    ):
        try:
            cash = float(cash)
            port_notional = float(port_notional)
            arrival_price = float(arrival_price)
            confidence = float(confidence)
            ticker_weight = float(ticker_weight)
        except (ValueError, TypeError) as e:
            self.logger.error(f"Numeric conversion failed: {e}")
            return

        signal_type = signal_type.upper()
        if signal_type not in ("BUY", "SELL", "HOLD"):
            self.logger.warning(
                f"Invalid signal type '{signal_type}' for {ticker}. Must be BUY, SELL, or HOLD."
            )
            return

        confidence = max(0.0, min(1.0, confidence))
        if signal_type == "HOLD" or confidence == 0.0:
            return

        exec_price = self._apply_slippage(arrival_price, signal_type)
        if exec_price <= 0:
            self.logger.warning(
                f"Cannot execute trade for {ticker}: Invalid execution price of {exec_price} after slippage."
            )
            return

        # --- Unified Sizing & Margin Logic (Reconciled with Live Executor) ---
        current_quantity = self.positions.get(ticker, 0.0)
        current_notional_value = current_quantity * exec_price

        # If ticker_weight is 0 (no current position), default to equal-weight allocation
        if ticker_weight == 0.0:
            tickers_list = self.tickers
            if not tickers_list or len(tickers_list) == 0:
                self.logger.error("No tickers list available for fallback allocation.")
                return
            ticker_weight = 1.0 / len(tickers_list)
        target_notional = port_notional * ticker_weight
        # A SELL signal targets a negative (short) position
        if signal_type == "SELL":
            target_notional *= -1

        adjustment_notional = target_notional - current_notional_value
        desired_trade_notional = adjustment_notional * confidence

        # Ignore trades smaller than $1.00 notional
        if abs(desired_trade_notional) < 1.0:
            return

        # --- Constraint Application (Mirrors Live Logic) ---
        # Buying power constrains BOTH new buys and new shorts.
        buying_power = self._calculate_buying_power(port_notional)

        # For buys, we are also constrained by the actual cash available.
        if desired_trade_notional > 0:  # This is a BUY operation
            tradable_notional = min(
                abs(desired_trade_notional), buying_power, self.cash
            )
        else:  # This is a SELL/SHORT operation
            tradable_notional = min(abs(desired_trade_notional), buying_power)

        if tradable_notional < 1.0:
            return

        quantity_to_trade = math.floor(tradable_notional / exec_price)

        if quantity_to_trade <= 0:
            return

        # --- Execute the Trade ---
        trade_value = quantity_to_trade * exec_price

        if desired_trade_notional > 0:  # Finalizing a BUY
            self.cash -= trade_value
            self.positions[ticker] += quantity_to_trade
        else:  # Finalizing a SELL
            self.cash += trade_value
            self.positions[ticker] -= quantity_to_trade

        self.trade_log.append(
            {
                "timestamp": timestamp,
                "portfolio_id": portfolio_id,
                "ticker": ticker,
                "signal_type": signal_type,
                "confidence": confidence,
                "shares": quantity_to_trade,
                "fill_price": exec_price,
                "cash_after": self.cash,
            }
        )

        return {
            "status": "success",
            "quantity": quantity_to_trade,
            "updated_cash": self.cash,
        }

    def dump_trade_log(self):
        """
        Emit all recorded trades to the executor's logger.

        This method should typically be called once after a backtest run has
        completed, so that detailed trade output does not interfere with
        progress bars (for example, those produced by ``tqdm``) or other
        console output written during execution.

        The method returns nothing; it iterates over ``self.trade_log`` and
        logs a formatted message for each recorded trade.
        """
        for entry in self.trade_log:
            self.logger.info(
                "Executed %s: %s| %s @ %.2f$ Cash after trade: %.2f$",
                entry["ticker"],
                entry["signal_type"],
                entry["shares"],
                entry["fill_price"],
                entry["cash_after"],
            )
