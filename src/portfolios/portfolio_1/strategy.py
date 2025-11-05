import logging
from typing import Dict
from src.portfolios.portfolio_BASE.strategy import BasePortfolio
from src.portfolios.strategy_api import StrategyContext


class VolMomentum(BasePortfolio):
    def __init__(self, db_connector, executor, debug=False, config_dict=None, backtest_start_date=None):
        super().__init__(db_connector, executor, debug, config_dict, backtest_start_date)
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")
        self.config_dict = config_dict or {}

        # strategy parameters
        self.momentum_lookback = self.config_dict.get("lookback_days", 20)
        self.vol_lookback = self.config_dict.get("volatility_lookback_days", 60)
        self.vol_multiplier = self.config_dict.get("volatility_multiplier", 1.5)

        # register indicators - dummy indicators not actually used
        indicator_definitions = {
            "returns": ("RelativeMomentumIndex", {"period": 14}), 
            "volatility": ("RelativeStrengthIndex", {"period": 14}) 
        }
        self.RegisterIndicatorSet(indicator_definitions)

    def OnData(self, context: StrategyContext):
        for ticker in self.tickers:
            returns = self.returns[ticker]
            vol = self.volatility[ticker]

            if not (returns.IsReady and vol.IsReady):
                continue

            momentum = returns.Current
            threshold = vol.Current * self.vol_multiplier
            position = context.Portfolio.positions.get(ticker, 0)

            if momentum > threshold and position == 0:
                context.buy(ticker, confidence=1.0)

            elif momentum < -threshold and position > 0:
                context.sell(ticker, confidence=1.0)