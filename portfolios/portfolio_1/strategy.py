from portfolios.portfolio_BASE.strategy import BasePortfolio

class SimpleMomentum(BasePortfolio):
    def generate_signals_and_trade(self, market_data):
        by_ticker = {}
        for row in market_data:
            by_ticker.setdefault(row['ticker'], []).append(row)

        for ticker, rows in by_ticker.items():
            if len(rows) < 2:
                continue
            latest = rows[0]
            prev = rows[1]

            if latest['close_price'] > prev['close_price'] * 1.01:
                self.execute_trade(ticker, 'BUY', 1)
            elif latest['close_price'] < prev['close_price'] * 0.99:
                self.execute_trade(ticker, 'SELL', 1)
