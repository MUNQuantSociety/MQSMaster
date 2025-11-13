import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from src.portfolios.indicators.base import Indicator
from src.portfolios.portfolio_BASE.strategy import BasePortfolio
from src.portfolios.strategy_api import StrategyContext

class TrendRotateStrategy(BasePortfolio):
    def __init__(self, db_connector, executor, debug=False, config_dict=None, backtest_start_date=None):
        super().__init__(db_connector, executor, debug, config_dict, backtest_start_date)
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")
        self.config_dict = config_dict or {}

        # strategy parameters
        self.trend_period = self.config_dict.get("trend_period", 50)
        self.beta_period = self.config_dict.get("beta_lookback", 90)
        # for risk on and risk off (i think)
        self.group_map = self.config_dict.get("asset_groups", {})

        # add indicators here
        indicator_definitions = {
            "trend": ("Portfolio4Dummy", {"period": self.trend_period}),
            "beta": ("Portfolio4Dummy", {"period": self.beta_period}),
        }

        self.RegisterIndicatorSet(indicator_definitions)
        self.logger.info("TrendRotateStrategy initialized.")
        
    def _asset_is_trending(self, ticker) -> bool:
        trend_ind = self.trend[ticker]
        if not trend_ind.IsReady:
            return False
        return trend_ind.Price > trend_ind.Current  # price > SMA → trending up
    
    def OnData(self, context: StrategyContext):
        # trend level 

        # rotation 
        risk_on = 
        risk_off = 

