<<<<<<< HEAD
import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from portfolios.portfolio_BASE.strategy import BasePortfolio
from typing import Dict, Optional

class RegimeAdaptiveStrategy(BasePortfolio):
    """
    Adaptive strategy that switches between momentum and mean-reversion (VWAP/ATR fades)
    based on volatility regime (using VIX or rolling volatility).
    """

    def __init__(self, db_connector, executor, debug=False):
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, 'r') as f:
            config_data = json.load(f)

        super().__init__(db_connector=db_connector,
                         executor=executor,
                         debug=debug,
                         config_dict=config_data)

        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")
        self.interval_seconds = self.poll_interval
        self.last_decision_time = {}
        self.trade_history = {}  # used for confidence decay
        self.market_open_start = timedelta(hours=9, minutes=30)
        self.market_open_end = timedelta(hours=10, minutes=0)

        # Load VIX data
        vix_path = os.path.join(os.path.dirname(__file__), "vix_1min_full_2023.csv")
        self.vix_df = pd.read_csv(vix_path, parse_dates=["datetime"])
        self.vix_df.set_index("datetime", inplace=True)

        self.logger.info("Strategy initialized: Regime-Adaptive")

    def compute_rolling_volatility(self, df: pd.DataFrame, window: int = 20) -> float:
        if len(df) < window:
            return 0.0
        returns = df['close_price'].pct_change().dropna()
        return returns.rolling(window).std().iloc[-1]
    
    def compute_vwap(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        return (df['close_price'] * df['volume']).sum() / df['volume'].sum()
    
    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period:
            return 0.0
        high_low = df['high_price'] - df['low_price']
        high_close = (df['high_price'] - df['close_price'].shift( )).abs()
        low_close = (df['low_price'] - df['close_price'].shift()).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr.iloc[-1]
    
    def is_high_volatility(self, timestamp: datetime, ticker_data: pd.DataFrame) -> bool:
        # 1. Market open → always high volatility
        time_of_day = timestamp.time()
        if self.market_open_start <= timedelta(hours=time_of_day.hour, minutes=time_of_day.minute) <= self.market_open_end:
            return True

        # 2. Try VIX first
        vix_value = self.vix_df['close'].asof(timestamp) if timestamp in self.vix_df.index else None
        if vix_value is not None:
            return vix_value > 17  # <- customizable threshold

        # 3. Fallback: use rolling price volatility
        rolling_vol = self.compute_rolling_volatility(ticker_data)
        return rolling_vol > 0.01  # <- also customizable
    
    def fade_signal(self, df: pd.DataFrame, latest_price: float) -> str:
        vwap = self.compute_vwap(df)
        atr = self.compute_atr(df)

        upper_band = vwap + atr
        lower_band = vwap - atr

        if latest_price > upper_band:
            return "SELL"
        elif latest_price < lower_band:
            return "BUY"
        else:
            return "HOLD"
        
    def momentum_signal(self, df: pd.DataFrame) -> str:
        df_sorted = df.sort_values('timestamp')
        if len(df_sorted) < 5:
            return "HOLD"

        prev_price = df_sorted['close_price'].iloc[-6]
        curr_price = df_sorted['close_price'].iloc[-1]

        if prev_price == 0:
            return "HOLD"

        change = (curr_price - prev_price) / prev_price

        if change > 0.002:
            return "BUY"
        elif change < -0.002:
            return "SELL"
        else:
            return "HOLD"
        
    def calculate_confidence(self, ticker: str) -> float:
        """
        Use a decaying confidence score based on recent trade outcomes.
        Score = base * decay^failures
        """
        base_conf = 0.2
        decay = 0.85 #needs to be determined
        history = self.trade_history.get(ticker, [])
        failures = sum(1 for h in history[-5:] if not h)

        return round(base_conf * (decay ** failures), 4)

    def update_trade_history(self, ticker: str, success: bool):
        if ticker not in self.trade_history:
            self.trade_history[ticker] = []
        self.trade_history[ticker].append(success)
        if len(self.trade_history[ticker]) > 10:
            self.trade_history[ticker] = self.trade_history[ticker][-10:]



    
    def generate_signals_and_trade(self,
                                   dataframes_dict: Dict[str, pd.DataFrame],
                                   current_time: Optional[datetime] = None):

        market_data = dataframes_dict.get('MARKET_DATA')
        cash_available = dataframes_dict.get('CASH_EQUITY')
        positions = dataframes_dict.get('POSITIONS')
        port_notional = dataframes_dict.get('PORT_NOTIONAL')

        if market_data is None or market_data.empty:
            return

        trade_ts = current_time or datetime.now().astimezone()
        current_cash = cash_available.iloc[0]['notional'] if not cash_available.empty else 0.0

        for ticker in self.tickers:
            try:
                last_decision = self.last_decision_time.get(ticker)
                if last_decision and (trade_ts - last_decision) < timedelta(seconds=self.interval_seconds):
                    continue

                ticker_data = market_data[market_data['ticker'] == ticker]
                if len(ticker_data) < 6:
                    continue  # not enough data for momentum or ATR

                recent_data = ticker_data.sort_values('timestamp').tail(20).copy()
                latest_price = recent_data['close_price'].iloc[-1]

                is_high_vol = self.is_high_volatility(trade_ts, recent_data)
                if is_high_vol:
                    signal = self.fade_signal(recent_data, latest_price)
                else:
                    signal = self.momentum_signal(recent_data)

                confidence = self.calculate_confidence(ticker)

                if signal != "HOLD":
                    quantity = positions[positions['ticker'] == ticker]['quantity'].iloc[0] \
                        if not positions.empty and not positions[positions['ticker'] == ticker].empty else 0.0

                    trade_result = self.executor.execute_trade(
                        portfolio_id=self.portfolio_id,
                        ticker=ticker,
                        signal_type=signal,
                        confidence=confidence,
                        arrival_price=latest_price,
                        cash=current_cash,
                        positions=quantity,
                        port_notional=port_notional.iloc[0]['notional'] if not port_notional.empty else 0.0,
                        ticker_weight=self.portfolio_weights.get(ticker, 1.0 / len(self.tickers)),
                        timestamp=trade_ts
                    )

                    if trade_result and trade_result.get('status') == 'success':
                        current_cash = trade_result['updated_cash']
                        self.update_trade_history(ticker, success=True)
                    else:
                        self.update_trade_history(ticker, success=False)

                    self.last_decision_time[ticker] = trade_ts

            except Exception as e:
                self.logger.exception(f"[{ticker}] Error during trading decision: {e}")
=======
import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional

# Imports from your project structure
from src.portfolios.portfolio_BASE.strategy import BasePortfolio
from src.portfolios.strategy_api import StrategyContext

class RegimeAdaptiveStrategy(BasePortfolio):
    """
    Adaptive strategy that switches between momentum and mean-reversion (VWAP/ATR fades)
    based on the VIX, using the OnData framework.
    
    All logic is contained within __init__ and OnData.
    """

    def __init__(self, db_connector, executor, debug=False, config_dict=None, backtest_start_date=None):
        
        # --- Base Class Initialization ---
        if config_dict is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                
        super().__init__(db_connector, executor, debug, config_dict, backtest_start_date)
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")

        # --- Strategy Properties ---
        self.interval_seconds = self.poll_interval
        self.last_decision_time = {} # Cooldown timer per ticker
        self.trade_history = {}      # History for confidence decay
        
        self.market_open_start = timedelta(hours=9, minutes=30)
        self.market_open_end = timedelta(hours=10, minutes=0)

        #*---------------------------------------------------
        #* 1. DEFINE YOUR INDICATORS HERE
        #*---------------------------------------------------
        indicator_definitions = { 
            "vwap": ("VWAP", {"period": 20}),
            "atr": ("AverageTrueRange", {"period": 14}),
            "momentum_pct": ("RateOfChange", {"period": 5}) 
        }

        self.RegisterIndicatorSet(indicator_definitions)
        
        self.logger.info("RegimeAdaptiveStrategy initialized for OnData framework.")
        self.logger.info(f"Registered indicators: {list(indicator_definitions.keys())}")


    def OnData(self, context: StrategyContext):
        """
        This method is called for each new data point (e.g., each minute bar).
        All strategy logic is contained here.
        """
        
        # --- 1. Get Current State from Context ---
        try:
            trade_ts = context.time
            current_cash = context.Portfolio.cash
            positions_dict = context.Portfolio.positions
            port_notional = context.Portfolio.total_value
        except Exception as e:
            self.logger.error(f"Error accessing context properties: {e}")
            return

        # --- 2. Determine Volatility Regime from VIX ---
        try:
            vix_asset = context.Market['^VIX']
            
            if not vix_asset.Exists or vix_asset.Close is None:
                self.logger.warning(f"No VIX data found at {trade_ts}. Skipping cycle.")
                return
            vix_value = vix_asset.Close 
        except Exception as e:
            self.logger.error(f"Error getting VIX data: {e}")
            return
            
        # --- 3. Loop Through Tickers and Apply Logic ---
        for ticker in self.tickers:
            
            if ticker == '^VIX':
                continue

            try:
                # --- 3a. Cooldown Check ---
                last_decision = self.last_decision_time.get(ticker)
                if last_decision and (trade_ts - last_decision) < timedelta(seconds=self.interval_seconds):
                    continue

                # --- 3b. Get Indicators & Market Data ---
                vwap_ind = self.vwap[ticker]
                atr_ind = self.atr[ticker]
                momentum_ind = self.momentum_pct[ticker]
                
                asset_data = context.Market[ticker]

                # --- 3c. Check if all data is ready ---
                indicators_to_check = [vwap_ind, atr_ind, momentum_ind]
                if not all(ind.IsReady for ind in indicators_to_check):
                    continue
                
                if not asset_data.Exists:
                    self.logger.warning(f"No market data for {ticker} at {trade_ts}")
                    continue

                # --- 3d. Get Current Values ---
                vwap_v = vwap_ind.Current
                atr_v = atr_ind.Current
                momentum_v = momentum_ind.Current
                latest_price = asset_data.Close
                quantity = positions_dict.get(ticker, 0.0)

                all_values = [vwap_v, atr_v, momentum_v, latest_price, quantity]
                if any(v is None for v in all_values):
                    self.logger.warning(f"Skipping {ticker} due to None values.")
                    continue

                #*---------------------------------------------------
                #* INLINED: is_high_volatility()
                #*---------------------------------------------------
                time_of_day = trade_ts.time()
                is_market_open = self.market_open_start <= timedelta(hours=time_of_day.hour, minutes=time_of_day.minute) <= self.market_open_end
                is_high_vol = is_market_open or (vix_value > 10) # VIX threshold

                
                #*---------------------------------------------------
                #* INLINED: fade_signal() and momentum_signal()
                #*---------------------------------------------------
                signal = "HOLD"
                if is_high_vol:
                    # Regime: High Volatility -> Use Fade (Mean Reversion)
                    upper_band = vwap_v + atr_v
                    lower_band = vwap_v - atr_v

                    if latest_price > upper_band:
                        signal = "SELL"
                    elif latest_price < lower_band:
                        signal = "BUY"
                else:
                    # Regime: Low Volatility -> Use Momentum
                    if momentum_v > 0.002: # 0.2% positive change
                        signal = "BUY"
                    elif momentum_v < -0.002: # 0.2% negative change
                        signal = "SELL"

                
                #*---------------------------------------------------
                #* INLINED: calculate_confidence()
                #*---------------------------------------------------
                base_conf = 0.2
                decay = 0.85
                history = self.trade_history.get(ticker, [])
                failures = sum(1 for h in history[-5:] if not h) # Count recent failures
                confidence = round(base_conf * (decay ** failures), 4)


                # --- 3e. Execute Trade (if signal) ---
                if signal != "HOLD":
                    
                    trade_result = self.executor.execute_trade(
                        portfolio_id=self.portfolio_id,
                        ticker=ticker,
                        signal_type=signal,
                        confidence=confidence,
                        arrival_price=latest_price,
                        cash=current_cash,
                        positions=quantity,
                        port_notional=port_notional,
                        ticker_weight=self.portfolio_weights.get(ticker, 0.0),
                        timestamp=trade_ts
                    )

                    #*---------------------------------------------------
                    #* INLINED: update_trade_history()
                    #*---------------------------------------------------
                    success = False
                    if trade_result and trade_result.get('status') == 'success':
                        success = True
                        current_cash = trade_result['updated_cash'] 
                    
                    if ticker not in self.trade_history:
                        self.trade_history[ticker] = []
                    
                    self.trade_history[ticker].append(success)
                    
                    if len(self.trade_history[ticker]) > 10:
                        self.trade_history[ticker] = self.trade_history[ticker][-10:]
                    #*---------------------------------------------------

                    self.last_decision_time[ticker] = trade_ts

            except Exception as e:
                self.logger.exception(f"[{ticker}] Error during OnData decision loop: {e}")
>>>>>>> fallRefactor
