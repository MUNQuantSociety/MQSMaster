# portfolios/portfolio_3/strategy.py

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from portfolios.portfolio_BASE.strategy import BasePortfolio
from typing import List, Dict, Optional, Union

class SAMPLE_PORTFOLIO(BasePortfolio):
    """
    Simple Mean Reversion Strategy:
    """
    def __init__(self, db_connector, executor, debug=False):
        child_dir = os.path.dirname(__file__)
        config_path = os.path.join(child_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.json in {child_dir}")
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load/parse config.json: {e}") from e

        super().__init__(db_connector=db_connector,
                         executor=executor,
                         debug=debug,
                         config_dict=config_data)


        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")

    def generate_signals_and_trade(self,
                                     dataframes_dict: Dict[str, pd.DataFrame],
                                     current_time: Optional[datetime] = None):

        market_data = dataframes_dict.get('MARKET_DATA', None)
        cash_available = dataframes_dict.get('CASH_EQUITY', None)
        positions = dataframes_dict.get('POSITIONS', None)
        port_notional = dataframes_dict.get('PORT_NOTIONAL', None)
        

        if market_data is None:
            self.logger.debug("Strategy received no valid market data.") # Optional: reduce noise
            return
        
        df = market_data

        for ticker in self.tickers:
            ticker_data = df[df['ticker'] == ticker].copy()

            try:
                ticker_data.set_index('timestamp', inplace=True, drop=False)
                ticker_data = ticker_data[~ticker_data.index.duplicated(keep='last')].sort_index()

                start_time = trade_ts - timedelta(hours=3)

                # 2. Filter the DataFrame
                df_last_3_hours = ticker_data[
                    (ticker_data['timestamp'] >= start_time)
                    & 
                    (ticker_data['timestamp'] <= trade_ts)
                ].copy()

                mean = df_last_3_hours.mean(numeric_only=True)

                latest_price = ticker_data.sort_values(by='timestamp', ascending=False).iloc[0]['close_price']

                if latest_price < mean:
                    signal = 'BUY'
                else:
                    signal = 'SELL'

                trade_ts = current_time if current_time is not None else datetime.now()

                self.executor.execute_trade(self.portfolio_id,
                                   ticker,
                                   signal_type=signal,
                                   confidence=1, # Assuming confidence is 1 for simplicity,
                                   arrival_price=latest_price,
                                   cash = cash_available.iloc[0]['notional'],
                                   positions=positions[positions['ticker'] == ticker]['quantity'].iloc[0] if not positions.empty else 0,
                                   port_notional=port_notional.iloc[0]['notional'],
                                   ticker_weight=self.portfolio_weights[ticker],
                                   timestamp=trade_ts
                                   )

            except Exception as e:
                self.logger.debug(f"{ticker}: Error in portfolio_1 {trade_ts}: {e}")