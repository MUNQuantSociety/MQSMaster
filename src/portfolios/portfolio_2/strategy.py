import json
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
            "sma_fast": ("SimpleMovingAverage", {"period": 7}),
            "sma_slow": ("SimpleMovingAverage", {"period": 14}),
            "rmi": ("RelativeMomentumIndex", {"period": 3, "momentum_period": 7}),
        }

        self.RegisterIndicatorSet(indicator_definitions)
        
    def OnData(self, context: StrategyContext):
        for ticker in self.tickers:
            fast = self.sma_fast[ticker]
            slow = self.sma_slow[ticker]
            rmi = self.rmi[ticker]

            if not (fast.IsReady and slow.IsReady and rmi.IsReady):
                continue

            fast_v = fast.Current
            slow_v = slow.Current
            rmi_v = rmi.Current
            position = context.Portfolio.positions.get(ticker, 0)

            bullish = fast_v > slow_v
            bearish = fast_v < slow_v
            oversold = rmi_v is not None and rmi_v < 20
            overbought = rmi_v is not None and rmi_v > 80

            # Entry logic
            if position < 50:
                if bullish and oversold:
                    context.buy(ticker, confidence=2.0)
                elif bullish and rmi_v is not None and rmi_v > 50:
            # momentum confirmation
                    context.buy(ticker, confidence=0.8)

            # Exit logic
            elif position > 0:
                if bearish or overbought:
                    context.sell(ticker, confidence=1.0)