# data_infra/tradingOps/backtest/single_ticker_executor.py

from datetime import datetime, timezone # Added timezone for UTC consistency
from typing import Optional # Import Optional
import logging # Import logging

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
        # Add a logger instance
        self.logger = logging.getLogger(self.__class__.__name__)


    # *** MODIFIED: Add optional timestamp argument ***
    def execute_trade(self,
                      portfolio_id: str,
                      ticker: str,
                      signal_type: str,
                      confidence: float,
                      timestamp: Optional[datetime] = None): # Added timestamp
        """
        Simulates executing a trade for this single ticker.

        - BUY invests up to confidence * initial_capital, limited by available cash.
        - SELL sells confidence * current holdings.
        - Logs the trade using the provided timestamp or falls back to UTC now.
        """
        # Use a small tolerance for floating point checks
        tolerance = 1e-9

        if self.latest_price <= tolerance:
            self.logger.debug(f"{ticker}: Skipping {signal_type} - No valid latest price ({self.latest_price}).")
            return

        signal_type = signal_type.upper()
        if signal_type not in ('BUY', 'SELL'):
            self.logger.error(f"Invalid signal type '{signal_type}' received for {ticker}.")
            return

        # Clamp confidence just in case
        confidence = max(0.0, min(1.0, confidence))

        # *** MODIFIED: Determine log timestamp ***
        # Use provided timestamp, otherwise use current UTC time for consistency
        log_ts = timestamp or datetime.now(timezone.utc)
        # *** END MODIFICATION ***


        # --- BUY Logic ---
        if signal_type == 'BUY':
            # Calculate intended investment based on *initial* capital and confidence
            intended_investment = confidence * self.initial_capital
            # Investment cannot exceed available cash
            actual_investment = min(intended_investment, self.cash)

            if actual_investment <= tolerance:
                self.logger.debug(f"{ticker}: Skipping BUY - No cash ({self.cash:.2f}) or zero investment.")
                return

            shares_to_buy = actual_investment / self.latest_price
            if shares_to_buy <= tolerance:
                 self.logger.debug(f"{ticker}: Skipping BUY - Calculated shares too small ({shares_to_buy}).")
                 return

            # Update average price (weighted average)
            total_cost_before = self.quantity * self.avg_price
            total_cost_new = total_cost_before + actual_investment
            new_qty = self.quantity + shares_to_buy
            # Avoid division by zero if new_qty is somehow still zero (shouldn't happen here)
            new_avg = total_cost_new / new_qty if new_qty > tolerance else 0.0

            # Update state
            self.quantity = new_qty
            self.avg_price = new_avg
            self.cash -= actual_investment

            # Log the trade
            self.trade_log.append({
                'timestamp': log_ts, # Use determined log_ts
                'portfolio_id': portfolio_id,
                'ticker': ticker,
                'signal_type': 'BUY',
                'confidence': confidence,
                'shares': shares_to_buy,
                'fill_price': self.latest_price,
                'cash_after': self.cash
            })
            self.logger.debug(f"Executed BUY {shares_to_buy:.4f} {ticker} @ {self.latest_price:.4f}")

        # --- SELL Logic ---
        elif signal_type == 'SELL':
            if self.quantity <= tolerance:
                self.logger.debug(f"{ticker}: Skipping SELL - No current holdings ({self.quantity}).")
                return

            # Calculate shares to sell based on *current* quantity and confidence
            shares_to_sell = confidence * self.quantity
            # Ensure we don't sell more than we have (due to float precision)
            shares_to_sell = min(shares_to_sell, self.quantity)

            if shares_to_sell <= tolerance:
                self.logger.debug(f"{ticker}: Skipping SELL - Calculated shares too small ({shares_to_sell}).")
                return

            proceeds = shares_to_sell * self.latest_price

            # Update state
            new_qty = self.quantity - shares_to_sell
            self.quantity = new_qty if new_qty > tolerance else 0.0 # Set to 0 if very close
            self.cash += proceeds
            # Reset average price if position is closed
            if self.quantity == 0.0:
                self.avg_price = 0.0

            # Log the trade
            self.trade_log.append({
                'timestamp': log_ts, # Use determined log_ts
                'portfolio_id': portfolio_id,
                'ticker': ticker,
                'signal_type': 'SELL',
                'confidence': confidence,
                'shares': shares_to_sell,
                'fill_price': self.latest_price,
                'cash_after': self.cash
            })
            self.logger.debug(f"Executed SELL {shares_to_sell:.4f} {ticker} @ {self.latest_price:.4f}")


    def get_portfolio_value(self) -> float:
        """ Calculate current value = cash + (quantity * latest_price). """
        # Use max with 0 to avoid potential negative value if latest_price dips unexpectedly below 0
        holdings_value = self.quantity * max(self.latest_price, 0.0)
        return self.cash + holdings_value