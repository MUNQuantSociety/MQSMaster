import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from src.portfolios.portfolio_BASE.strategy import BasePortfolio
from src.portfolios.strategy_api import StrategyContext

class TrendRotateStrategy(BasePortfolio):
    def __init__(self, db_connector, executor, debug=False, config_dict=None, backtest_start_date=None):
        super().__init__(db_connector, executor, debug, config_dict, backtest_start_date)
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")
        self.config_dict = config_dict or {}

        # strategy parameters
        self.fast_period = self.config_dict.get("fast_trend_period", 20)
        self.slow_period = self.config_dict.get("slow_trend_period", 50)

        # asset groups
        self.group_map = self.config_dict.get("asset_groups", {})
        self.risk_on = self.group_map.get("risk_on", [])
        self.risk_off = self.group_map.get("risk_off", [])
        if not self.risk_on or not self.risk_off:
            self.logger.warning("Config missing risk_on or risk_off groups.")

        # indicators
        indicator_definitions = {
            "trend_fast": ("SimpleMovingAverage", {"period": self.fast_period}),
            "trend_slow": ("SimpleMovingAverage", {"period": self.slow_period}),
        }
        self.RegisterIndicatorSet(indicator_definitions)

    def OnData(self, context: StrategyContext):
       # ensure config has portfolio_id (does not work without this)
        config_with_id = self.config_dict.copy()
        config_with_id['id'] = self.portfolio_id
        context._portfolio_config = config_with_id

        positions = context.Portfolio.positions or {}
        port_notional = context.Portfolio.total_value
        if not port_notional or port_notional <= 0:
            self.logger.warning("Invalid or zero portfolio notional.")
            return

       # check trend
        risk_on_trending = [
            t for t in self.risk_on
            if self.trend_fast[t].IsReady
            and self.trend_slow[t].IsReady
            and self.trend_fast[t].Current > self.trend_slow[t].Current
        ]

        risk_off_trending = [
            t for t in self.risk_off
            if self.trend_fast[t].IsReady
            and self.trend_slow[t].IsReady
            and self.trend_fast[t].Current > self.trend_slow[t].Current
        ]

        # select active group
        if len(risk_on_trending) >= len(risk_off_trending):
            active_group = self.risk_on
            trending = risk_on_trending
        else:
            active_group = self.risk_off
            trending = risk_off_trending

        if not active_group:
            self.logger.warning("Active group empty. Skipping.")
            return

        target_weight = 1.0 / len(active_group)

        # prep simulated weights
        simulated_weights = {}
        latest_prices = {
            t: context.Market[t].Close if context.Market[t].Exists else None
            for t in self.tickers
        }

        for ticker in self.tickers:
            price = latest_prices[ticker]
            qty = float(positions.get(ticker, 0))
            current_val = qty * price if price else 0
            simulated_weights[ticker] = current_val / port_notional if port_notional else 0

        # rotation execution
        for ticker in self.tickers:
            price = latest_prices[ticker]
            if price is None or price <= 0:
                continue

            qty = float(positions.get(ticker, 0))
            current_weight = simulated_weights[ticker]

            is_in_active = ticker in active_group
            is_trending = ticker in trending
            should_hold = is_in_active and is_trending

            if should_hold:
                # increase position if underweight
                if current_weight < target_weight * 0.95:
                    context.buy(ticker, confidence=1.0)
                    simulated_weights[ticker] += (target_weight - current_weight)

                # decrease if overweight
                elif current_weight > target_weight * 1.05:
                    context.sell(ticker, confidence=0.6)
                    simulated_weights[ticker] -= (current_weight - target_weight)

            else:
                # exit if currently holding
                if qty > 0:
                    context.sell(ticker, confidence=1.0)
                    simulated_weights[ticker] = 0.0

#hello testing world
