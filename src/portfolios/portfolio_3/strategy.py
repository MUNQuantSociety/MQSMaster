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
