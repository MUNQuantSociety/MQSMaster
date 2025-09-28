import logging
from typing import Dict

from src.portfolios.indicators.base import Indicator
from src.portfolios.portfolio_BASE.strategy import BasePortfolio
from src.portfolios.strategy_api import StrategyContext


class MomentumStrategy(BasePortfolio):
    def __init__(self, db_connector, executor, debug=False, config_dict=None, backtest_start_date=None):
        super().__init__(db_connector, executor, debug, config_dict, backtest_start_date)
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")

        indicator_definitions = { # Format: "indicator_variable_name": ("IndicatorName", {params})
            "sma_fast": ("SimpleMovingAverage", {"period": 10}),
            "sma_slow": ("SimpleMovingAverage", {"period": 30})
        }
        
        self.RegisterIndicatorSet(indicator_definitions)
        
    def OnData(self, context: StrategyContext):
        for ticker in self.tickers:
            fast_sma = self.sma_fast[ticker]
            slow_sma = self.sma_slow[ticker]

            if not slow_sma.IsReady:
                self.logger.debug(f"Indicators for {ticker} are not ready yet.")
                continue

            fast_value = fast_sma.Current
            slow_value = slow_sma.Current
            
            current_position = context.Portfolio.positions.get(ticker, 0)
            
            if fast_value > slow_value and current_position == 0:
                context.buy(ticker, confidence=1.0)

            elif fast_value < slow_value and current_position > 0:
                context.sell(ticker, confidence=1.0)