import pandas as pd
from datetime import datetime, timedelta
from portfolios.portfolio_BASE.strategy import BasePortfolio

class SimpleMeanReversion(BasePortfolio):
    def generate_signals_and_trade(self, market_data: list[dict]):
        df = pd.DataFrame(market_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp', 'ticker', 'close_price'])

        today = datetime.now().date()
        min_time = datetime.now() - timedelta(minutes=30)
        df = df[df['timestamp'].dt.date == today]
        df = df[df['timestamp'] >= min_time]

        for ticker in self.tickers:
            ticker_df = df[df['ticker'] == ticker]
            if len(ticker_df) < 2:
                continue

            mean_price = ticker_df['close_price'].mean()
            latest_price = ticker_df.iloc[-1]['close_price']

            if latest_price < mean_price:
                self.execute_trade(ticker, 'BUY', confidence=1.0)
            else:
                self.logger.debug(f"{ticker} no signal. Price {latest_price:.2f} >= Mean {mean_price:.2f}")