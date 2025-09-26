import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from portfolios.portfolio_BASE.strategy import BasePortfolio
from typing import Dict, Optional

class VolMomentum(BasePortfolio):
    """
    A simple momentum strategy that buys assets with strong positive returns
    over a lookback period and sells assets with strong negative returns.
    The threshold for buying or selling is dynamically adjusted based on volatility.
    """
    def __init__(self, db_connector, executor, debug=False):
        """
        Initializes the VolMomentum strategy.
        """
        child_dir = os.path.dirname(__file__)
        config_path = os.path.join(child_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found for VolMomentum at {config_path}")

        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load or parse config.json: {e}") from e

        super().__init__(db_connector=db_connector,
                         executor=executor,
                         debug=debug,
                         config_dict=config_data)

        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")

        # --- Strategy-Specific Parameters ---
        self.momentum_lookback_days = self.lookback_days
        self.volatility_lookback_days = 60 # Lookback period for volatility calculation
        self.volatility_multiplier = 1.5 # Multiplier for setting the dynamic threshold
        self.interval_seconds = self.poll_interval
        self.last_decision_time = {}

        self.logger.info(f"VolMomentum portfolio '{self.portfolio_id}' initialized.")
        self.logger.info(
            f"Strategy Parameters: Momentum Lookback = {self.momentum_lookback_days} days, "
            f"Volatility Lookback = {self.volatility_lookback_days} days, "
            f"Volatility Multiplier = {self.volatility_multiplier}, "
            f"Trade Interval = {self.interval_seconds} seconds"
        )


    def generate_signals_and_trade(self,
                                   dataframes_dict: Dict[str, pd.DataFrame],
                                   current_time: Optional[datetime] = None):
        """Generates BUY, SELL, and HOLD signals based on momentum and volatility, updates cash available for trade, and then calls the trade execution logic for each signal."""

        #* Get DataFrames if none or empty exit strategy
        market_data = dataframes_dict.get('MARKET_DATA')
        cash_available = dataframes_dict.get('CASH_EQUITY')
        positions = dataframes_dict.get('POSITIONS')
        port_notional = dataframes_dict.get('PORT_NOTIONAL')

        if market_data is None or market_data.empty:
            return

        #* Stamp current time of trade
        trade_ts = current_time or datetime.now().astimezone()

        #* Get current cash available
        current_cash_in_loop = cash_available.iloc[0]['notional'] if not cash_available.empty else 0.0

        #? A loop to iterate through each ticker and generate signals based on momentum and volatility.
        for ticker in self.tickers:
            #? check if we have traded this ticker recently based on interval_seconds=600(10 mins).
            #? This prevents overtrading within short intervals.
            try:
                last_decision = self.last_decision_time.get(ticker)
                if last_decision and (trade_ts - last_decision) < timedelta(seconds=self.interval_seconds):
                    continue
                # Filtered market data for the current ticker; contains rows with columns such as 'timestamp', 'close_price', etc. for this ticker.
                ticker_data = market_data[market_data['ticker'] == ticker]
                if ticker_data.empty:
                    continue
                #? Ensure we have enough data for both momentum and volatility calculations
                if len(ticker_data) < 2:
                    continue
                #? Calculate daily returns and check that we have enough data points(min 2) to compute percent change
                daily_prices = ticker_data.resample('D', on='timestamp')['close_price'].last()
                returns = daily_prices.pct_change().dropna()

                if len(returns) < 2:
                    continue
                #? Calculate volatility for dynamic momentum threshold(volatility(Standard Deviation of summed returns) * multiplier(1.5)).
                volatility = returns.std()
                dynamic_momentum_threshold = volatility * self.volatility_multiplier

                if len(ticker_data) < 2:
                    continue
                #? Calculate momentum over the lookback period
                latest_price = ticker_data['close_price'].iloc[-1]
                lookback_price = ticker_data['close_price'].iloc[0]
                #? Avoid division by zero(base case)
                if lookback_price == 0:
                    momentum_return = 0.0
                else:
                    #? Calculate momentum return(change in price over lookback price)
                    momentum_return = (latest_price - lookback_price) / lookback_price
                #* Generate signals based on momentum and dynamic threshold
                #? If momentum return is greater than dynamic threshold, volatility is positively increasing, signal is BUY
                #? If momentum return is less than negative dynamic threshold, volatility is decreasing, signal is SELL
                if momentum_return > dynamic_momentum_threshold:
                    signal = 'BUY'
                elif momentum_return < -dynamic_momentum_threshold:
                    signal = 'SELL'
                else:
                    signal = 'HOLD'
                # update the generated signal
                self.last_decision_time[ticker] = trade_ts
                #?if signal is BUY or SELL update position DF
                if signal in ['BUY', 'SELL']:
                    ticker_pos_series = positions[positions['ticker'] == ticker]['quantity'] if not positions.empty else pd.Series(dtype=float)
                    current_quantity = ticker_pos_series.iloc[0] if not ticker_pos_series.empty else 0.0

                    #? Execute trade and update cash available
                    #? The executor handles the actual trade logic and updates the database accordingly.
                    #? We pass all necessary parameters to ensure the trade can be executed correctly.
                    trade_result = self.executor.execute_trade(
                        portfolio_id=self.portfolio_id,
                        ticker=ticker,
                        signal_type=signal,
                        confidence=1.0,
                        arrival_price=latest_price,
                        cash=current_cash_in_loop,
                        positions=current_quantity,
                        port_notional=port_notional.iloc[0]['notional'] if not port_notional.empty else 0.0,
                        ticker_weight=self.portfolio_weights.get(ticker, 1.0 / len(self.tickers)),
                        timestamp=trade_ts
                    )
                    #? If trade is successful, update current cash available for next iteration
                    if trade_result and trade_result.get('status') == 'success':
                        current_cash_in_loop = trade_result['updated_cash']

            except Exception as e:
                self.logger.exception(f"[{ticker}] An error occurred during signal generation: {e}")