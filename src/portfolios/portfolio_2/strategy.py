import logging
from src.portfolios.indicators.base import Indicator
from src.portfolios.portfolio_BASE.strategy import BasePortfolio
from src.portfolios.strategy_api import StrategyContext


class MomentumStrategy(BasePortfolio):
    def __init__(self, db_connector, executor, debug=False, config_dict=None, backtest_start_date=None):
        super().__init__(db_connector, executor, debug, config_dict, backtest_start_date)
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")

        indicator_definitions = { # Format: "indicator_variable_name": ("IndicatorName", {params})
            "sma_fast": ("SimpleMovingAverage", {"period": 14}),
            "sma_slow": ("SimpleMovingAverage", {"period": 28}),
            "rmi": ("RelativeMomentumIndex", {"period": 14, "momentum_period": 14}),
            "rsi": ("RelativeStrengthIndex", {"period": 14}),
            "dma": ("DisplacedMovingAverage", {"period": 14, "displacement": 7}),
            #"stochastic": ("StochasticOscillator", {"k_period": 14, "d_period": 3}),
            #"macd": ("MACD", {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
        }

        self.RegisterIndicatorSet(indicator_definitions)
        
    def OnData(self, context: StrategyContext):
        # Loop over each ticker and apply strategy logic
        #? Indicator references: add more as needed
        for ticker in self.tickers:
            fast = self.sma_fast[ticker]
            slow = self.sma_slow[ticker]
            rmi = self.rmi[ticker]
            rsi = self.rsi[ticker]
            dma = self.dma[ticker]
            #stochastic = self.stochastic[ticker]
            #macd = self.macd[ticker]

            if not (fast.IsReady and slow.IsReady and rmi.IsReady and rsi.IsReady and rsi.IsReady and dma.IsReady):
                continue

            #* ---- Additional indicators can be enabled here ----
            fast_v = fast.Current
            slow_v = slow.Current
            rmi_v = rmi.Current
            rsi_v = rsi.Current
            dma_v = dma.Current
            #stochastic_v = stochastic.Current
            #macd_v = macd.Current

            position = context.Portfolio.positions.get(ticker, 0)
            if any(v is None for v in 
            #? add additional indicators here after enabling them above
                   [
                    position,
                    fast_v,
                    slow_v,
                    rmi_v,
                    rsi_v,
                    dma_v,
                    #stochastic_v, macd_v
                    ]
                   ):
                self.logger.warning(
                    f"Skipping {ticker} due to None indicator values: fast={fast_v}, slow={slow_v}, rmi={rmi_v}, rsi={rsi_v}"
                )
                continue
#*---------------------------------------------------
#* add indicator logic here in the appropriate section
#*---------------------------------------------------

#? Moving Average strategy logic
            sma_bullish = fast_v > slow_v
            sma_bearish = fast_v < slow_v
            dma_bullish = dma_v > slow_v
            dma_bearish = dma_v < slow_v

#? Momentum/Oscillator logic
            rsi_oversold = 10 < rsi_v < 30
            rsi_overbought = 90 > rsi_v > 70
            rmi_oversold = 10 < rmi_v < 30
            rmi_overbought = 90 > rmi_v > 70
            #stochastic_oversold = stochastic_v < 20
            #stochastic_overbought = stochastic_v > 80
            #macd_bullish = macd_v > 0
            #macd_bearish = macd_v < 0
#*---------------------------------------------------
            #* Combine signals for entries and exits
            #* computes the and of both sets of indicator conditions
#*---------------------------------------------------
#? momentum/oscillator indicators
            oversold = [
                x for x in [
                    rsi_oversold,
                    rmi_oversold
                    #stochastic_oversold, macd_bearish
                    ]
                if x
            ]
            overbought = [
                x for x in [
                    rsi_overbought,
                    rmi_overbought
                    #stochastic_overbought, macd_bullish
                    ]
                if x
            ]
#? Moving Average indicators
            bullish = [
                x for x in [
                    sma_bullish,
                    dma_bullish
                    ]
                if x
            ]
            bearish = [
                x for x in [
                    sma_bearish,
                    dma_bearish
                    ]
                if x
            ]
            #?Entry logic
            if position < 10:
                if bullish and oversold:
                    context.buy(ticker, confidence=1.0)
                elif bullish:
                    context.buy(ticker, confidence=0.8)

            #? Exit logic
            elif position > 0:
                if bearish or overbought:
                    context.sell(ticker, confidence=1.0)
                elif bearish:
                    context.sell(ticker, confidence=0.5)