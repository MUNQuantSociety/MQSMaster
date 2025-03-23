# BUY/SELL functionality sits here.

from data_infra.database.tradeAPI import tradeAPI

class tradeExecutor():
    def __init__(self):
        self.tradeAPI = tradeAPI()

    def executor(self, portfolio_id, ticker, signal, quantity):
        latest_price = self.tradeAPI.get_latest_price(ticker)

        if signal == "BUY":
            cash_on_hand = self.tradeAPI.get_portfolio_cash(portfolio_id)
            if cash_on_hand >= latest_price * quantity:
                self.tradeAPI.commit_trade(portfolio_id, ticker, signal, quantity, latest_price):

        elif signal == "SELL":
            current_quantity =  db.get_holding_quantity(portfolio_id, ticker)
            if current_quantity >= quantity:
                self.tradeAPI.commit_trade(portfolio_id, ticker, signal, quantity, latest_price):

