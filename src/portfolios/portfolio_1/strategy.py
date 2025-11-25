from datetime import datetime
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


    def on_data(self,
                                   dataframes_dict: Dict[str, pd.DataFrame],
                                   current_time: Optional[datetime] = None):
        """Generates BUY, SELL, and HOLD signals based on momentum and volatility, updates cash available for trade, and then calls the trade execution logic for each signal."""
    
        #? Get DataFrames if none or empty exit strategy
        market_data = dataframes_dict.get('MARKET_DATA')
        cash_available = dataframes_dict.get('CASH_EQUITY')
        positions = dataframes_dict.get('POSITIONS')
        port_notional = dataframes_dict.get('PORT_NOTIONAL')

        if market_data is None or market_data.empty:
            return
        context = StrategyContext(
            market_data_df=market_data,
            cash_df=cash_available,
            positions_df=positions,
            port_notional_df=port_notional,
            current_time=current_time,
            executor=self.executor,
            portfolio_config=self.config_dict
        )
        #? Stamp current time of trade
        trade_ts = current_time or datetime.now().astimezone()

        #? Get current cash available
        current_cash_in_loop = cash_available.iloc[0]['notional'] if not cash_available.empty else 0.0

        #? A loop to iterate through each ticker and generate signals based on momentum and volatility.
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