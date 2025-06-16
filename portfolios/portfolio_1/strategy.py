import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from portfolios.portfolio_BASE.strategy import BasePortfolio
from typing import List, Dict, Optional, Union


class SimpleMeanReversion(BasePortfolio):
    def __init__(self, db_connector, executor, debug=False):
        child_dir = os.path.dirname(__file__)
        config_path = os.path.join(child_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.json in {child_dir}")

        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding config.json: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load config.json: {e}") from e

        super().__init__(db_connector=db_connector,
                         executor=executor,
                         debug=debug,
                         config_dict=config_data)

        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")
        self.strategy_lookback_minutes = 30
        self.logger.info(f"SimpleMeanReversion portfolio '{self.portfolio_id}' initialized.")

    def generate_signals_and_trade(self,
                                   dataframes_dict: Dict[str, pd.DataFrame],
                                   current_time: Optional[datetime] = None):

        market_data = dataframes_dict.get('MARKET_DATA', None)
        cash_available = dataframes_dict.get('CASH_EQUITY', None)
        positions = dataframes_dict.get('POSITIONS', None)
        port_notional = dataframes_dict.get('PORT_NOTIONAL', None)

        if market_data is None or market_data.empty:
            self.logger.debug("Strategy received no valid market data.")
            return

        df = market_data

        for ticker in self.tickers:
            try:
                ticker_data = df[df['ticker'] == ticker].copy()
                ticker_data.set_index('timestamp', inplace=True, drop=False)
                ticker_data = ticker_data[~ticker_data.index.duplicated(keep='last')].sort_index()

                trade_ts = current_time if current_time is not None else datetime.now()
                window_start = trade_ts - timedelta(minutes=self.strategy_lookback_minutes)

                df_window = ticker_data[(ticker_data['timestamp'] >= window_start) & 
                                        (ticker_data['timestamp'] <= trade_ts)].copy()

                if df_window.empty:
                    self.logger.debug(f"{ticker}: No data in lookback window.")
                    continue

                mean_price = df_window['close_price'].mean()
                latest_row = ticker_data.iloc[-1]
                latest_price = latest_row['close_price']
                latest_ts_used = latest_row['timestamp']

                if pd.isna(latest_price) or pd.isna(mean_price):
                    self.logger.warning(f"{ticker}: Invalid price or mean encountered.")
                    continue

                signal = None
                if latest_price < mean_price:
                    signal = 'BUY'
                elif latest_price > mean_price:
                    signal = 'SELL'
                else:
                    self.logger.debug(f"{ticker}: No trade signal (price ≈ mean).")
                    continue

                self.executor.execute_trade(
                    self.portfolio_id,
                    ticker,
                    signal_type=signal,
                    confidence=1.0,
                    arrival_price=latest_price,
                    cash=cash_available.iloc[0]['notional'] if not cash_available.empty else 0.0,
                    positions=positions[positions['ticker'] == ticker]['quantity'].iloc[0] if not positions.empty and ticker in positions['ticker'].values else 0,
                    port_notional=port_notional.iloc[0]['notional'] if not port_notional.empty else 0.0,
                    ticker_weight=self.portfolio_weights.get(ticker, 1.0 / len(self.tickers)),
                    timestamp=trade_ts
                )

                self.logger.debug(f"{ticker}: Executed {signal} at {latest_price:.2f} (mean={mean_price:.2f})")

            except Exception as e:
                self.logger.exception(f"{ticker}: Error in signal generation or execution: {e}")
