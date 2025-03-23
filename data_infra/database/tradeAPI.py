from data_infra.database.MQSDBConnector import MQSDBConnector

class tradeAPI():
    def __init__(self):
        self.db = MQSDBConnector()
        self.INITIAL_PORTFOLIO_VALUE = 100000

    def get_portfolio_cash(self, portfolio_id):
        response = self.db.read_db(sql = f'''
            SELECT
                {self.INITIAL_PORTFOLIO_VALUE} - SUM(CASE WHEN tel.side = 'BUY' THEN tel.quantity * tel.price_last ELSE 0 END) 
                + SUM(CASE WHEN tel.side = 'SELL' THEN tel.quantity * tel.price_last ELSE 0 END) AS cash
            FROM
                trade_execution_logs tel
            WHERE
                tel.portfolio_id = '{portfolio_id}'
            GROUP BY
                tel.portfolio_id;''')
        
        if response['status'] == "success":
            if response['data']:
                return float(response['data'][0]['cash'])
            else:
                raise Exception(f"Portfolio {portfolio_id} does not exist")
        else:
            raise Exception(response["message"])
    
    def get_latest_price(self, ticker):
        response =  self.db.read_db(sql = f'''
            SELECT close_price
            FROM market_data
            WHERE ticker = '{ticker}'
            ORDER BY timestamp DESC
            LIMIT 1;''')
        
        if response['status'] == "success":
            if response['data']:
                return float(response['data'][0]['close_price'])
            else:
                raise Exception(f"Ticker {ticker} does not exist")
        else:
            raise Exception(response["message"])
    
    def get_holding_quantity(self, portfolio_id, ticker):
        response = self.db.read_db(sql = f'''
            SELECT 
                COALESCE(
                    SUM(CASE WHEN side = 'BUY' THEN quantity ELSE 0 END) - 
                    SUM(CASE WHEN side = 'SELL' THEN quantity ELSE 0 END),
                    0
                ) AS current_quantity
            FROM 
                trade_execution_logs
            WHERE 
                portfolio_id = '{portfolio_id}'
                AND ticker = '{ticker}';''')
        
        if response['status'] == "success":
            return float(response['data'][0]['current_quantity'])
        else:
            raise Exception(response["message"])
        
        
    def commit_trade(self, portfolio_id, ticker, signal, quantity, price):
        data = {
            "portfolio_id": portfolio_id,
            "ticker": ticker,
            "exec_timestamp": "NOW()",
            "side": signal,
            "quantity": quantity,
            "price_last": price,
            "notional": quantity * price,
            "notional_local": quantity * price,
            "currency": 'USD',
            "fx_rate": 1.0,
            "created_at": "NOW()"
        }
        self.db.inject_to_db("trade_execution_logs", data)