from datetime import datetime
import logging
import math
from data_infra.database.schemaDefinitions import MQSDBConnector
from data_infra.authentication.apiAuth import APIAuth
from data_infra.marketData.fmpMarketData import FMPMarketData

# --- Best Practice: Configure logging once at the application entry point ---
# This basic configuration will show log messages of INFO level and higher.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class tradeExecutor:
    def __init__(self, db_connector: MQSDBConnector):
        """Initializes the tradeExecutor and its components."""
        self.dbconn = db_connector
        self.api_auth = APIAuth()
        self.fmp_api_key = self.api_auth.get_fmp_api_key()
        self.marketData = FMPMarketData()
        # Correctly initialize the logger for the class instance
        self.logger = logging.getLogger(__name__)
        self.logger.info("tradeExecutor initialized.")


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
        Calculates trade size and executes it by updating the database.
        
        Note: Core position sizing logic has been preserved as requested.
        """
        signal_type = signal_type.upper()
        if signal_type not in ('BUY', 'SELL', 'HOLD'):
            # Use the logger exclusively for application-level errors
            self.logger.error(f"Invalid signal type '{signal_type}' for {ticker}. No trade executed.")
            return

        # Clamp confidence to be between 0.0 and 1.0
        confidence = max(0.0, min(1.0, confidence))

        if signal_type == 'HOLD' or confidence == 0.0:
            self.logger.info(f"Signal is HOLD or confidence is 0 for {ticker}. No trade executed.")
            return

        # --- Core Position Sizing Logic (in Dollars) ---
        current_notional_value = positions * arrival_price
        max_target_notional_value = port_notional * ticker_weight
        
        quantity_to_trade = 0
        updated_cash = cash
        updated_quantity = positions

        if signal_type == 'BUY':
            # Determine the max dollar amount to reach the weight limit
            available_notional_to_buy = max(0, max_target_notional_value - current_notional_value)
            
            # Scale the target purchase by the confidence factor
            target_buy_notional = max_target_notional_value * confidence

            # The notional to buy is the lesser of the available room and the desired target
            buy_notional = min(available_notional_to_buy, target_buy_notional)
            
            # Ensure the purchase does not exceed available cash
            actual_buy_notional = min(buy_notional, cash)
            
            # Convert final dollar amount to an integer number of shares
            quantity_to_trade = math.floor(actual_buy_notional / arrival_price)

            # Calculate portfolio changes based on the actual trade
            updated_cash = cash - (quantity_to_trade * arrival_price)
            updated_quantity = positions + quantity_to_trade

        elif signal_type == 'SELL':
            # The maximum we can sell is our current holding
            available_notional_to_sell = current_notional_value
        
            # Scale the target sale by the confidence factor
            target_sell_notional = max_target_notional_value * confidence

            # The notional to sell is the lesser of what we have and what we want to sell
            sell_notional = min(available_notional_to_sell, target_sell_notional)
            
            # Convert final dollar amount to an integer number of shares
            quantity_to_trade = math.floor(sell_notional / arrival_price)

            # Calculate portfolio changes based on the actual trade
            updated_cash = cash + (quantity_to_trade * arrival_price)
            updated_quantity = positions - quantity_to_trade

        # --- Final check before execution ---
        if quantity_to_trade == 0:
            self.logger.info(f"Calculated trade quantity for {ticker} is 0. No database update needed.")
            return

        # Corrected method call: 'self' is passed implicitly
        return self.update_database(
            portfolio_id, ticker, signal_type, quantity_to_trade, 
            updated_cash, updated_quantity, arrival_price, timestamp
        )


    def update_database(self, portfolio_id, ticker, signal_type, quantity_to_trade, 
                     updated_cash, updated_quantity, arrival_price, timestamp):
        """
        Update database tables after trade execution within a single transaction.
        If any operation fails, all changes are rolled back.
        """
        # Extract date and calculate trade notional
        date_part = timestamp.date()
        trade_notional = abs(quantity_to_trade * arrival_price)

        conn = None
        try:
            # 1. Get a single connection from the pool for the entire transaction.
            conn = self.dbconn.get_connection()
            if not conn:
                self.logger.error("Failed to get a database connection from the pool.")
                return

            # Use a 'with' block for the cursor to ensure the transaction is closed properly.
            with conn.cursor() as cursor:
                # 2. Execute all database write operations using the same cursor.

                # Update cash_equity_book
                cash_query = """
                    INSERT INTO cash_equity_book (timestamp, date, portfolio_id, currency, notional)
                    VALUES (%s, %s, %s, %s, %s)
                """
                cash_values = (timestamp, date_part, portfolio_id, 'USD', updated_cash)
                cursor.execute(cash_query, cash_values)
                
                # Update positions_book (upsert)
                position_query = """
                    INSERT INTO positions_book (portfolio_id, ticker, quantity, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (portfolio_id, ticker)
                    DO UPDATE SET 
                        quantity = EXCLUDED.quantity,
                        updated_at = EXCLUDED.updated_at
                """
                position_values = (portfolio_id, ticker, updated_quantity, timestamp)
                cursor.execute(position_query, position_values)
                
                # Insert trade log
                trade_log_query = """
                    INSERT INTO trade_execution_logs (
                        portfolio_id, ticker, exec_timestamp, side, quantity, 
                        price_last, notional, notional_local, currency, fx_rate
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                trade_log_values = (
                    portfolio_id, ticker, timestamp, signal_type, quantity_to_trade,
                    arrival_price, None, trade_notional, 'USD', None
                )
                cursor.execute(trade_log_query, trade_log_values)

            # 3. If all 'execute' calls succeed, commit the transaction.
            conn.commit()
            self.logger.info(f"Database successfully updated for trade: {signal_type} {quantity_to_trade} {ticker}")
            return {'status': 'success', 'quantity': quantity_to_trade}

        except Exception as e:
            # 4. If any operation fails, log the error and roll back the entire transaction.
            self.logger.exception("Database update transaction failed. Rolling back all changes.")
            if conn:
                try:
                    conn.rollback()
                except Exception as rollback_error:
                    self.logger.error(f"Failed to rollback transaction: {rollback_error}")
            return {'status': 'error', 'message': str(e)}

        finally:
            # 5. Always release the connection back to the pool.
            if conn:
                self.dbconn.release_connection(conn)

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
        pos_result = self.dbconn.execute_query(sql_get_positions, values=(portfolio_id,), fetch=True)
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