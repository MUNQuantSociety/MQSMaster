from datetime import datetime
import logging
import math
from data_infra.database.schemaDefinitions import MQSDBConnector
from data_infra.authentication.apiAuth import APIAuth
from data_infra.marketData.fmpMarketData import FMPMarketData

class tradeExecutor:
    def __init__(self):
        self.table = MQSDBConnector()
        self.api_auth = APIAuth()
        self.fmp_api_key = self.api_auth.get_fmp_api_key()
        self.marketData= FMPMarketData()

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

        return(self.update_database(self, portfolio_id, ticker, signal_type, quantity_to_trade, updated_cash, updated_quantity, arrival_price, timestamp))


    def update_database(self, portfolio_id, ticker, signal_type, quantity_to_trade, 
                 updated_cash, updated_quantity, arrival_price, timestamp):
        """
        Update database tables after trade execution using individual queries
        """
        # Extract date from timestamp
        date_part = timestamp.date()
        # Calculate trade notional
        trade_notional = abs(quantity_to_trade * arrival_price)
        
        try:
            # 1. Update cash_equity_book
            cash_query = """
                INSERT INTO cash_equity_book (timestamp, date, portfolio_id, currency, notional)
                VALUES (%s, %s, %s, %s, %s)
            """
            cash_values = (timestamp, date_part, portfolio_id, 'USD', updated_cash)
            cash_result = self.table.execute_query(cash_query, values=cash_values, fetch=False)
            
            if cash_result['status'] != 'success':
                self.logger.error(f"Cash update failed: {cash_result.get('message', 'Unknown error')}")
        
            # 2. Update positions_book (upsert)
            position_query = """
                INSERT INTO positions_book (portfolio_id, ticker, quantity, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (portfolio_id, ticker)
                DO UPDATE SET 
                    quantity = EXCLUDED.quantity,
                    updated_at = EXCLUDED.updated_at
            """
            position_values = (portfolio_id, ticker, updated_quantity, timestamp)
            position_result = self.table.execute_query(position_query, values=position_values, fetch=False)
            
            if position_result['status'] != 'success':
                self.logger.error(f"Position update failed: {position_result.get('message', 'Unknown error')}")
        
            # 3. Insert trade log
            trade_log_query = """
                INSERT INTO trade_execution_logs (
                    portfolio_id, ticker, exec_timestamp, side, quantity, 
                    price_last, notional, notional_local, currency, fx_rate
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            trade_log_values = (
                portfolio_id, ticker, timestamp, signal_type, quantity_to_trade,
                arrival_price, trade_notional, trade_notional, 'USD', 1.0
            )
            trade_result = self.table.execute_query(trade_log_query, values=trade_log_values, fetch=False)
            
            if trade_result['status'] != 'success':
                self.logger.error(f"Trade log failed: {trade_result.get('message', 'Unknown error')}")
            
            # If all successful
            if (cash_result['status'] == 'success' and 
                position_result['status'] == 'success' and 
                trade_result['status'] == 'success'):
                self.logger.info(f"Trade updated: {signal_type} {quantity_to_trade} {ticker} @ {arrival_price}")
            else:
                self.logger.warning("Partial database update completed with errors")
                
        except Exception as e:
            self.logger.exception("Failed to update database in trade_executor")

    def get_current_price(self, ticker):
        """
        Fetch real-time stock price for a single ticker using FMP API.
        
        Args:
            ticker (str): Ticker symbol to get price for
            
        Returns:
            float: Current price of the ticker, or 0.0 if not found
        """
        # Base URL for the quote endpoint
        url = "https://financialmodelingprep.com/api/v3/quote/"
        
        # Parameters including the ticker symbol and API key
        params = {
            "symbol": ticker,
            "apikey": self.fmp_api_key
        }
        
        try:
            # Make API request using the _make_request method
            data = self.marketData._make_request(url, params)
            
            # Response format: [{'symbol': 'AAPL', 'price': 172.35, ...}]
            if isinstance(data, list) and data and 'price' in data[0]:
                return float(data[0]['price'])
            
            self.logger.warning(f"No price found for ticker {ticker}, defaulting to 0.")
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Price fetch failed for {ticker}: {str(e)}")
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