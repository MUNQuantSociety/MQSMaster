import logging
from src.portfolios.indicators.base import Indicator
from src.portfolios.portfolio_BASE.strategy import BasePortfolio
from src.portfolios.strategy_api import StrategyContext


class VolMomentum(BasePortfolio):
    def __init__(self, db_connector, executor, debug=False, config_dict=None, backtest_start_date=None):
        super().__init__(db_connector, executor, debug, config_dict, backtest_start_date)
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")
        # Format: "indicator_variable_name": ("IndicatorName", {params})
        indicator_definitions = {
            "roc": ("RateOfChange", {"period": 20}),
        }
        self.RegisterIndicatorSet(indicator_definitions)

    def OnData(self, context: StrategyContext):
        """Generates BUY, SELL, and HOLD signals based on momentum and volatility, updates cash available for trade, and then calls the trade execution logic for each signal."""
        portfolio = context.Portfolio
        is_risk_off = portfolio.cash < (float(portfolio.total_value) * 0.10)
        if is_risk_off:
            self.logger.info("Risk-Off Mode: Cash is low. No new long positions will be opened.")

        #? A loop to iterate through each ticker and generate signals based on momentum and volatility.
        for ticker in self.tickers:
            asset = context.Market[ticker]
            roc = self.roc[ticker]
            vol_multiplier = 1.5  # This can be adjusted or made configurable

            if not all([asset.Exists, roc.IsReady]):
                continue

            return_history = asset.History("60d")
            returns = return_history['close_price'].pct_change(60).dropna()
            volatility = returns.std() * (252 ** 0.5)

            momentum = roc.Current
            threshold = volatility * vol_multiplier
            position = portfolio.positions.get(ticker, 0)


            bullish = momentum > threshold
            bearish = momentum < -threshold

            if bullish and not is_risk_off:
                is_risk_off = False

            weight = 0.2 if bullish else 0.0
            asset_weight = 0.0
            if asset.Exists:
                asset_weight = portfolio.get_asset_weight(ticker, asset.Close)
            if asset_weight <= weight:
                target_weight = True
            else:
                target_weight = False
                
            if (bullish and target_weight) or position < 0: # Max 25% weight
                self.logger.debug(f"[{ticker}] BUY signal: momentum ({momentum:.4f}) > threshold ({threshold:.4f}), position={position}")
                context.buy(ticker, confidence=1.0)

            elif position > 0 and (bearish or target_weight is False or is_risk_off):
                self.logger.debug(f"[{ticker}] SELL signal: momentum ({momentum:.4f}) < threshold ({threshold:.4f}), position={position}")
                context.sell(ticker, confidence=1.0)