# portfolio_1.py

import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
from portfolios.portfolio_BASE.strategy import BasePortfolio
import pytz


class SAMPLE_PORTFOLIO(BasePortfolio):
    """
    Simple Mean Reversion Strategy for Portfolio 1:
    - For each ticker, look back 3 hours from current time (or backtest time),
      compute mean of close_price, compare latest price to mean, generate BUY/SELL.
    """
    def __init__(self, db_connector, executor, debug=False):
        # Load config.json from this directory
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
        self.strategy_lookback_hours = 3
        self.logger.info(f"SAMPLE_PORTFOLIO '{self.portfolio_id}' initialized with tickers: {self.tickers}")

    def generate_signals_and_trade(self,
                                   dataframes_dict: Dict[str, pd.DataFrame],
                                   current_time: Optional[datetime] = None):
        """
        dataframes_dict keys: 'MARKET_DATA', 'CASH_EQUITY', 'POSITIONS', 'PORT_NOTIONAL'
        current_time: provided by backtest runner; if None, use datetime.now()
        """
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
                if ticker_data.empty:
                    self.logger.debug(f"{ticker}: No market data rows.")
                    continue

                # Ensure timestamp column is datetime and sorted
                ticker_data['timestamp'] = pd.to_datetime(ticker_data['timestamp'], errors='coerce')
                ticker_data = ticker_data.dropna(subset=['timestamp', 'close_price'])
                ticker_data = ticker_data.sort_values('timestamp')
                ticker_data = ticker_data[~ticker_data['timestamp'].duplicated(keep='last')]
                ticker_data = ticker_data.set_index('timestamp', drop=False)

                tz = ticker_data.index.tz
                if tz is None:
                    self.logger.warning(f"{ticker}: Timestamp column has no timezone info. Assuming UTC.")
                    tz = pytz.UTC

                # Create timezone-AWARE datetime objects
                trade_ts = current_time if current_time is not None else datetime.now(tz)
                if trade_ts.tzinfo is None: # If current_time was passed without tz
                    trade_ts = tz.localize(trade_ts)

                # Compute lookback window
                window_start = trade_ts - timedelta(hours=self.strategy_lookback_hours)
                df_window = ticker_data[
                    (ticker_data['timestamp'] >= window_start) &
                    (ticker_data['timestamp'] <= trade_ts)
                ].copy()

                if df_window.empty:
                    self.logger.debug(f"{ticker}: No data in last {self.strategy_lookback_hours} hours before {trade_ts}.")
                    continue

                # Compute mean of close_price in window
                mean_price = df_window['close_price'].mean()

                # Latest price: take the last row by timestamp <= trade_ts
                # If exact match of timestamp isn't present, idxmax ensures latest available
                latest_idx = df_window['timestamp'].idxmax()
                latest_row = df_window.loc[latest_idx]
                latest_price = latest_row['close_price']

                if pd.isna(latest_price) or pd.isna(mean_price):
                    self.logger.warning(f"{ticker}: NaN encountered (latest_price={latest_price}, mean={mean_price}).")
                    continue

                # Decide signal
                if latest_price < mean_price:
                    signal = 'BUY'
                else:
                    signal = 'SELL'

                # Gather execution params
                # cash_available and port_notional assumed DataFrame with column 'notional'
                cash_amt = 0.0
                if cash_available is not None and not cash_available.empty:
                    # assume column name 'notional'
                    try:
                        cash_amt = float(cash_available.iloc[0].get('notional', 0.0))
                    except Exception:
                        cash_amt = 0.0

                pos_qty = 0
                if positions is not None and not positions.empty:
                    # find row for this ticker
                    try:
                        row = positions[positions['ticker'] == ticker]
                        if not row.empty:
                            pos_qty = float(row.iloc[0].get('quantity', 0.0))
                    except Exception:
                        pos_qty = 0

                port_not = 0.0
                if port_notional is not None and not port_notional.empty:
                    try:
                        port_not = float(port_notional.iloc[0].get('notional', 0.0))
                    except Exception:
                        port_not = 0.0

                weight = None
                if self.portfolio_weights:
                    weight = self.portfolio_weights.get(ticker)
                if weight is None:
                    # fallback equal weight
                    weight = 1.0 / len(self.tickers) if self.tickers else 0.0

                # Execute trade via executor
                self.executor.execute_trade(
                    self.portfolio_id,
                    ticker,
                    signal_type=signal,
                    confidence=1.0,
                    arrival_price=latest_price,
                    cash=cash_amt,
                    positions=pos_qty,
                    port_notional=port_not,
                    ticker_weight=weight,
                    timestamp=trade_ts
                )

                self.logger.debug(f"{ticker}: Executed {signal} at {latest_price:.2f} (mean {mean_price:.2f}) at {trade_ts}")

            except Exception as e:
                self.logger.exception(f"{ticker}: Error in portfolio_1 generate_signals_and_trade at {datetime.now()}: {e}")
