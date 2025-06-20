from datetime import datetime, timezone
from typing import Optional
import logging

class SingleTickerExecutor:
    """
    Tracks partial share buys/sells for a single ticker. Simulates trades
    and maintains the state of a single-asset sub-portfolio.
    """

    def __init__(self, initial_capital: float = 100000.0):
        if initial_capital < 0:
            raise ValueError("Initial capital cannot be negative.")
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.quantity = 0.0
        self.avg_price = 0.0
        self.trade_log: list[dict] = []
        self.latest_price = 0.0
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"SingleTickerExecutor initialized with capital {self.cash:.2f}")

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
        # If strategy passed an arrival_price, update internal price
        if arrival_price is not None:
            self.latest_price = arrival_price

        self.logger.debug(
            f"Received {signal_type.upper()} signal for {ticker} — "
            f"conf={confidence:.2f}, price={self.latest_price:.2f}, "
            f"cash={self.cash:.2f}, holdings={self.quantity:.4f}"
        )

        tolerance = 1e-9
        if self.latest_price <= tolerance:
            self.logger.warning(f"{ticker}: latest_price is zero or invalid; skipping trade.")
            return

        signal = signal_type.upper()
        if signal not in ("BUY", "SELL"):
            self.logger.error(f"{ticker}: invalid signal_type '{signal_type}'; must be BUY or SELL.")
            return

        confidence = max(0.0, min(1.0, confidence))
        log_ts = timestamp or datetime.now(timezone.utc)

        if signal == "BUY":
            intended = confidence * self.get_portfolio_value()
            invest = min(intended, self.cash)
            if invest <= tolerance:
                self.logger.debug(f"{ticker}: no cash available to invest; skipping BUY.")
                return

            shares = invest / self.latest_price
            # update running average price
            total_cost = self.quantity * self.avg_price
            total_cost += invest
            self.quantity += shares
            self.avg_price = total_cost / self.quantity
            self.cash -= invest

            self.trade_log.append({
                "timestamp": log_ts,
                "portfolio_id": portfolio_id,
                "ticker": ticker,
                "signal_type": "BUY",
                "confidence": confidence,
                "shares": shares,
                "fill_price": self.latest_price,
                "cash_after": self.cash
            })
            self.logger.info(f"Executed BUY {shares:.4f} of {ticker} @ {self.latest_price:.2f}")

        else:  # SELL
            if self.quantity <= tolerance:
                self.logger.debug(f"{ticker}: no holdings to sell; skipping SELL.")
                return

            shares = confidence * self.quantity
            shares = min(shares, self.quantity)
            if shares <= tolerance:
                self.logger.debug(f"{ticker}: calculated zero shares to sell; skipping.")
                return

            proceeds = shares * self.latest_price
            self.quantity -= shares
            self.cash += proceeds
            if self.quantity == 0:
                self.avg_price = 0.0

            self.trade_log.append({
                "timestamp": log_ts,
                "portfolio_id": portfolio_id,
                "ticker": ticker,
                "signal_type": "SELL",
                "confidence": confidence,
                "shares": shares,
                "fill_price": self.latest_price,
                "cash_after": self.cash
            })
            self.logger.info(f"Executed SELL {shares:.4f} of {ticker} @ {self.latest_price:.2f}")

    def get_portfolio_value(self) -> float:
        value = self.cash + self.quantity * max(self.latest_price, 0.0)
        return value
