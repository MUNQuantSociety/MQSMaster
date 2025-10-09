import logging
from typing import Dict
import numpy as np

from src.portfolios.indicators.base import Indicator
from src.portfolios.portfolio_BASE.strategy import BasePortfolio
from src.portfolios.strategy_api import StrategyContext

class CrossoverRmiStrategy(BasePortfolio):
    """
    A showcase strategy demonstrating advanced use of the StrategyContext API,
    including portfolio-level risk management, data transformation with the
    .toolkit accessor, and dynamic position logic.
    """
    def __init__(self, db_connector, executor, debug=False, config_dict=None, backtest_start_date=None):
        super().__init__(db_connector, executor, debug, config_dict, backtest_start_date)
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")

        # --- 1. Define All Indicators ---
        indicator_definitions = {
            "fast_sma": ("SimpleMovingAverage", {"period": 20}),
            "slow_sma": ("SimpleMovingAverage", {"period": 50}),
            "rmi": ("RelativeMomentumIndex", {"period": 14, "momentum_period": 4})
        }
        self.RegisterIndicatorSet(indicator_definitions)
        
        # --- State Tracking ---
        self._previous_sma_values: Dict[str, Dict[str, float]] = {
            ticker: {"fast": 0.0, "slow": 0.0} for ticker in self.tickers
        }

    def OnData(self, context: StrategyContext):
        # --- 1. Showcase PortfolioManager for top-down risk management ---
        portfolio = context.Portfolio

        # Risk Rule: Don't open new long positions if cash is less than 10% of total value
        is_risk_off = portfolio.cash < (portfolio.total_value * 0.10)
        if is_risk_off:
            self.logger.warning("Risk-Off Mode: Cash is low. No new long positions will be opened.")

        # --- 2. Loop through assets for trading logic ---
        for ticker in self.tickers:
            asset = context.Market[ticker]
            
            # Get indicator instances
            fast_sma = self.fast_sma[ticker]
            slow_sma = self.slow_sma[ticker]
            rmi = self.rmi[ticker]

            # Wait until all indicators are ready
            if not all([asset.Exists, fast_sma.IsReady, slow_sma.IsReady, rmi.IsReady]):
                continue

            # --- 3. Showcase History and the .toolkit accessor ---
            # Fetch history to calculate volatility
            history = asset.History("30d")
            returns = history['close_price'].pct_change().dropna()

            # Use the toolkit to remove outliers before calculating volatility
            winsorized_returns = returns.toolkit.winsorize(limits=[0.05, 0.05])
            volatility = winsorized_returns.std()
            
            # --- 4. Get Current and Previous Indicator Values ---
            fast_current = fast_sma.Current
            slow_current = slow_sma.Current
            fast_previous = self._previous_sma_values[ticker]["fast"]
            slow_previous = self._previous_sma_values[ticker]["slow"]
            rmi_value = rmi.Current
            

            current_position = portfolio.positions.get(ticker, 0)
            current_weight = portfolio.get_asset_weight(ticker, asset.Close)

            # --- 5. Implement Trading Logic using Context ---
            is_bullish_crossover = fast_previous < slow_previous and fast_current > slow_current
            is_bearish_crossover = fast_previous > slow_previous and fast_current < slow_current
            
            # Volatility Stop-Loss: Exit if price drops 3 std deviations below the fast SMA
            stop_loss_level = fast_current - (3 * volatility * fast_current)
            is_stop_loss_triggered = asset.Close < stop_loss_level

            # BUY LOGIC
            if is_bullish_crossover and rmi_value > 10 and current_position == 0:
                # Only buy if not in risk-off mode and we have room to increase position weight
                if not is_risk_off and current_weight < 0.25: # Max 25% weight
                    context.buy(ticker, confidence=1.0)
            
            # SELL LOGIC
            elif (is_bearish_crossover or is_stop_loss_triggered) and current_position > 0:
                context.sell(ticker, confidence=1.0)
            
            # Update the previous values for the next OnData call
            self._previous_sma_values[ticker]["fast"] = fast_current
            self._previous_sma_values[ticker]["slow"] = slow_current