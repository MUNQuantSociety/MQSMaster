"""
Specifies the BacktestExecutor class. Used for managing a portfolio.
"""
import logging
from typing import Any, Literal

import pandas as pd

from src.backtest.cost_model import CostModel


class BacktestExecutor:
    """
    A backtest executor that manages a single, unified portfolio,
    supporting long/short positions with a realistic margin model that mirrors
    live trading constraints.
    """

    def __init__(
        self,
        initial_capital: float,
        tickers: list[str],
        leverage: float = 2.0,
        slippage: float = 0.0,
        cost_model: CostModel | None = None,
        adv_lookup: dict[str, float] | None = None,
        sigma_lookup: dict[str, float] | None = None,
    ):
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.tickers: list[str] = tickers
        self.leverage: float = leverage
        self.slippage: float = slippage
        # Legacy constant slippage stays available as a fallback (when cost_model is None).
        self.cost_model: CostModel | None = cost_model
        self.adv_lookup: dict[str, float] = dict(adv_lookup or {})
        self.sigma_lookup: dict[str, float] = dict(sigma_lookup or {})

        # --- Unified Portfolio State ---
        self.cash: float = initial_capital
        self.positions: dict[str, float] = {ticker: 0.0 for ticker in tickers}
        self.latest_prices: dict[str, float] = {ticker: 0.0 for ticker in tickers}
        self.trade_log: list[dict[Any, Any]] = []

        self.logger.info(
            "BacktestExecutor initialized with %.2f capital, leverage=%.2f, slippage=%.2f, "
            + "cost_model=%s, for tickers: %s",
            initial_capital,
            leverage,
            slippage,
            'on' if self.cost_model is not None else 'off',
            tickers
        )

    def _apply_slippage(
        self,
        price: float,
        signal_type: str,
        ticker: str | None = None,
        trade_notional: float = 0.0,
    ) -> float:
        """
        Applies execution cost to the price.
        If a CostModel is configured, uses it (fixed + spread + sqrt impact).
        Otherwise falls back to the legacy constant-multiplier slippage.
        """
        if self.cost_model is not None and ticker is not None:
            adv = float(self.adv_lookup.get(ticker, 0.0))
            sigma = float(self.sigma_lookup.get(ticker, 0.0))
            return self.cost_model.apply_to_price(
                mid_price=price,
                side=signal_type,
                trade_notional=abs(trade_notional),
                adv_notional=adv,
                sigma_daily=sigma,
                ticker=ticker,
            )
        if signal_type == "BUY":
            return price * (1 + self.slippage)

        if signal_type == "SELL":
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

    def get_data_feeds(self) -> dict[str, pd.DataFrame]:
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
        portfolio_id: int,
        ticker: str,
        signal_type: Literal['BUY', 'SELL'],
        confidence: float,
        arrival_price: float,
        cash: float,
        positions: list[Any],
        port_notional: float,
        ticker_weight: float,
        timestamp,
    ) -> dict[str, str | int | float]:
        """
        Executes a trade.

        @returns a dictionary:
          - 'status' == 'success': Returns the necessary data
          - 'status' == 'skipped': Returns 'reason' for skipping, not an issue
          - 'status' == 'error': Returns 'reason' for the error
        """
        # Default to equal-weight allocation if unspecified
        if ticker_weight == 0.0 and len(self.tickers) > 0:
            ticker_weight = 1.0 / len(self.tickers)
        elif ticker_weight == 0.0:
            ticker_weight = 1.0

        confidence = max(0.001, min(1.0, confidence))

        # First-pass approximation of trade notional so the cost model can size impact.
        approx_notional = abs(port_notional * ticker_weight * confidence)
        exec_cost = self._apply_slippage(
            arrival_price, signal_type, ticker=ticker, trade_notional=approx_notional,
        )
        if exec_cost <= 0:
            return {
                "status": "error",
                "reason": f"Invalid execution price after slippage: {exec_cost}"
            }

        # --- Unified Sizing & Margin Logic (Reconciled with Live Executor) ---
        current_notional = self.positions.get(ticker, 0.0) * exec_cost
        target_notional = port_notional * ticker_weight

        # A SELL signal targets a negative (short) position
        if signal_type == "SELL":
            target_notional *= -1

        target_trade_notional = (target_notional - current_notional) * confidence

        # Ignore trades smaller than $1.00 notional
        if abs(target_trade_notional) < 1.0:
            return {
                "status": "skipped",
                "reason": "Desired notional below $1.00."
            }

        # --- Constraint Application (Mirrors Live Logic) ---
        # Buying power constrains BOTH new buys and new shorts.
        buying_power = self._calculate_buying_power(port_notional)

        # For buys, we are also constrained by the actual cash available.
        if target_trade_notional > 0:  # This is a BUY operation
            tradable_notional = min(
                abs(target_trade_notional), buying_power, self.cash
            )
        else:  # This is a SELL/SHORT operation
            tradable_notional = min(abs(target_trade_notional), buying_power)

        trade_qty = tradable_notional // exec_cost

        if tradable_notional < 1.0 or trade_qty <= 0:
            return {
                "status": "skipped",
                "reason": "Missing tradable notional or no quantity to trade."
            }

        # --- Execute the Trade ---
        if signal_type == 'BUY':
            self.cash -= trade_qty * exec_cost
            self.positions[ticker] += trade_qty
        elif signal_type == 'SELL':
            self.cash += trade_qty * exec_cost
            self.positions[ticker] -= trade_qty
        self.latest_prices[ticker] = arrival_price

        self.trade_log.append(
            {
                "timestamp": timestamp,
                "portfolio_id": portfolio_id,
                "ticker": ticker,
                "signal_type": signal_type,
                "confidence": confidence,
                "shares": trade_qty,
                "fill_price": exec_cost,
                "cash_after": self.cash,
            }
        )

        return {
            "status": "success",
            "quantity": trade_qty,
            "updated_cash": self.cash,
        }

    def dump_trade_log(self) -> list[str]:
        """
        Generate formatted trade log entries and return them as a list of strings.
        """
        trade_logs: list[str] = []
        for entry in self.trade_log:
            ts = entry["timestamp"]
            ts_str: str = (
                ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)
            )
            msg = (
                f"[{entry.get('portfolio_id', 'unknown')}] "
                f"{ts_str} - "
                f"{entry['ticker']} | {entry['signal_type']} "
                f"{entry['shares']} @ {entry['fill_price']:.2f}$ "
                f"cash={entry['cash_after']:.2f}$"
            )
            trade_logs.append(msg)
        return trade_logs
