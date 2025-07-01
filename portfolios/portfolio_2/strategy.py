import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from portfolios.portfolio_BASE.strategy import BasePortfolio
from typing import Dict, Optional

class MomentumThresholdStrategy(BasePortfolio):
    """
    A refined momentum strategy:
    - Uses historical and current price to decide direction.
    - Only acts if price moves more than a defined threshold.
    - Uses larger timeframes to reduce noise.
    """

    def __init__(self, db_connector, executor, debug=False):
        """
        Initializes the strategy from a config file.
        """
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading config.json: {e}") from e

        super().__init__(db_connector=db_connector,
                         executor=executor,
                         debug=debug,
                         config_dict=config_data)

        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")
        self.threshold = config_data.get("momentum_threshold", 0.005)  # Â±0.5% default
        self.interval_seconds = self.poll_interval
        self.last_decision_time = {}

        self.logger.info(f"Strategy initialized: Threshold = {self.threshold:.2%}, Interval = {self.interval_seconds}s")


    def generate_signals_and_trade(self,
                                   dataframes_dict: Dict[str, pd.DataFrame],
                                   current_time: Optional[datetime] = None):
        """
        Executes the strategy: compares current price to last price,
        and trades if change > threshold.
        """
        market_data = dataframes_dict.get('MARKET_DATA')
        cash_available = dataframes_dict.get('CASH_EQUITY')
        positions = dataframes_dict.get('POSITIONS')
        port_notional = dataframes_dict.get('PORT_NOTIONAL')

        if market_data is None or market_data.empty:
            return

        trade_ts = current_time or datetime.now().astimezone()

        for ticker in self.tickers:
            try:
                # Enforce polling interval
                last_decision = self.last_decision_time.get(ticker)
                if last_decision and (trade_ts - last_decision) < timedelta(seconds=self.interval_seconds):
                    continue

                ticker_data = market_data[market_data['ticker'] == ticker]
                if len(ticker_data) < 2:
                    continue

                # Most recent two prices
                latest_price = ticker_data['close_price'].iloc[-1]
                previous_price = ticker_data['close_price'].iloc[-2]

                if previous_price == 0:
                    price_change = 0.0
                else:
                    price_change = (latest_price - previous_price) / previous_price

                if price_change > self.threshold:
                    signal = "BUY"
                elif price_change < -self.threshold:
                    signal = "SELL"
                else:
                    signal = "HOLD"

                self.last_decision_time[ticker] = trade_ts

                if signal in ["BUY", "SELL"]:
                    ticker_pos_series = positions[positions['ticker'] == ticker]['quantity'] if not positions.empty else pd.Series(dtype=float)
                    current_quantity = ticker_pos_series.iloc[0] if not ticker_pos_series.empty else 0.0

                    self.executor.execute_trade(
                        portfolio_id=self.portfolio_id,
                        ticker=ticker,
                        signal_type=signal,
                        confidence=1.0,
                        arrival_price=latest_price,
                        cash=cash_available.iloc[0]['notional'] if not cash_available.empty else 0.0,
                        positions=current_quantity,
                        port_notional=port_notional.iloc[0]['notional'] if not port_notional.empty else 0.0,
                        ticker_weight=self.portfolio_weights.get(ticker, 1.0 / len(self.tickers)),
                        timestamp=trade_ts
                    )

            except Exception as e:
                self.logger.exception(f"[{ticker}] Error during trading decision: {e}")