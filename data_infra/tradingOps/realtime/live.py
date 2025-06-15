from datetime import datetime
import logging
import math
from data_infra.database.schemaDefinitions import MQSDBConnector

class tradeExecutor:
    def __init__(self):
        self.table = MQSDBConnector()

    # --- Trading Methods ---
    def execute_trade(self,
                  portfolio_id,
                  ticker, 
                  signal_type,
                  confidence,
                  arrival_price,
                  cash,
                  positions,
                  port_notional,
                  ticker_weight,
                  timestamp):
        """
        Calculates the trade size and prepares for order placement.
        Args:
            ticker: Target ticker symbol.
            signal_type: 'BUY' or 'SELL'.
            confidence: Strategy confidence (0.0 to 1.0) scaling the trade size.
            arrival_price: The price at which the trade is expected to execute.
            cash: Current cash available in the portfolio.
            positions: Current quantity (number of shares) held of the ticker.
            port_notional: Current total portfolio value (holdings + cash).
            ticker_weight: Target maximum weight for the ticker in the portfolio.
            timestamp: The simulation time (used for backtest logging/arrival time).
        """

        signal_type = signal_type.upper()
        if signal_type not in ('BUY', 'SELL', 'HOLD'):
            self.logger.error(f"Invalid signal type '{signal_type}' for {ticker}.")
            print(f"Invalid signal type '{signal_type}' for {ticker}.")
            return
            
        # Clamp confidence to be between 0.0 and 1.0
        confidence = max(0.0, min(1.0, confidence))

        if signal_type == 'HOLD' or confidence == 0.0:
            return # No trade to execute

        # --- Core Position Sizing Logic (in Dollars) ---

        # 1. Calculate current and maximum desired dollar values for the ticker
        current_notional_value = positions * arrival_price
        max_target_notional_value = port_notional * ticker_weight

        quantity_to_trade = 0

        if signal_type == 'BUY':
            # Determine the maximum dollar amount we *could* buy to reach the weight limit
            available_notional = max(0, max_target_notional_value - current_notional_value)
            
            # The actual desired buy amount is the least of portfolio notional scaled by confidence
            target_buy_notional = max_target_notional_value * confidence

            buy_notional = min(available_notional, target_buy_notional)
            
            # We can't spend more cash than we have
            actual_buy_notional = min(buy_notional, cash)
            
            # Convert the final dollar amount to an integer number of shares
            quantity_to_trade = math.floor(actual_buy_notional / arrival_price)

            updated_cash = cash - actual_buy_notional
            updated_quantity = positions + quantity_to_trade

        elif signal_type == 'SELL':
            # Determine the maximum dollar amount we *could* sell (our entire holding)
            # The desired sell amount is scaled by confidence
            available_notional = current_notional_value
        
            target_sell_notional = (max_target_notional_value * confidence)

            sell_notional = min(available_notional, target_sell_notional)

            # We can't sell more shares than we own
            actual_sell_notional = min(sell_notional, available_notional)
            
            # Convert the dollar amount to an integer number of shares
            quantity_to_trade = math.floor(actual_sell_notional / arrival_price)

            updated_cash = cash + actual_sell_notional
            updated_quantity = positions - quantity_to_trade

        update_database(self, portfolio_id, ticker, signal_type, quantity_to_trade, updated_cash, updated_quantity, arrival_price, timestamp)


    def update_database(self, portfolio_id, ticker, signal_type, quantity_to_trade, updated_cash, updated_quantity, arrival_price, timestamp):
        
        Update the following tables:
        cash_equity_book:
            timestamp IMESTAMP WITH TIME ZONE NOT NULL,
            date DATE NOT NULL,
            portfolio_id VARCHAR(50) NOT NULL,
            currency VARCHAR(10) NOT NULL,
            notional NUMERIC NOT NULL,

        positions_book:
            portfolio_id VARCHAR(50) NOT NULL,
            ticker VARCHAR(10) NOT NULL,
            quantity NUMERIC NOT NULL,
            updated_at TIMESTAMP DEFAULT NOW(),
        
        trade_execution_logs:
            portfolio_id VARCHAR(50),
            ticker VARCHAR(10),
            exec_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            side VARCHAR(4) NOT NULL,  -- e.g. 'BUY'/ 'SELL'/ HOLD
            quantity NUMERIC NOT NULL,
            price_last NUMERIC NOT NULL,
            notional NUMERIC,
            notional_local NUMERIC,
            currency VARCHAR(10),
            fx_rate NUMERIC,
        


    def get_current_price(self, ticker):
        """
        Retrieve the current price for a ticker using the latest close_price from market_data.
        """
        sql = """
            SELECT close_price
            FROM market_data
            WHERE ticker = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """
        result = self.table.execute_query(sql, values=(ticker,), fetch=True)
        if result['status'] == 'success' and result['data']:
            return float(result['data'][0]['close_price'])
        else:
            logging.warning(f"No price found for ticker {ticker}, defaulting to 0.")
            return 0.0
        

    def liquidate(self, portfolio_id):
        """
        Liquidate all positions:
          1. Retrieve all positions with a positive quantity from the positions table.
          2. For each ticker, execute a SELL for the full available amount.
          3. The SELL method will update trade_execution_logs, cash_equity_book, and positions.
        """
        sql_get_positions = """
            SELECT ticker, quantity
            FROM positions
            WHERE portfolio_id = %s AND quantity > 0
        """
        pos_result = self.table.execute_query(sql_get_positions, values=(portfolio_id,), fetch=True)
        if pos_result['status'] != 'success':
            logging.error(f"Failed to retrieve positions for portfolio {portfolio_id}")
            return

        exec_time = datetime.now() # Use a consistent timestamp for all liquidations
        for pos in pos_result['data']:
            ticker = pos['ticker']
            available_quantity = float(pos['quantity'])
            if available_quantity > 0:
                # Call the sell method for each ticker to update all tables.
                self.sell(portfolio_id, ticker, available_quantity, exec_time)

        logging.info(f"All positions for portfolio {portfolio_id} have been liquidated.")