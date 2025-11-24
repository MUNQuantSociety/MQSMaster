from datetime import datetime
import logging
import math
import pandas as pd
from common.database.schemaDefinitions import MQSDBConnector
from common.auth.apiAuth import APIAuth
from orchestrator.marketData.fmpMarketData import FMPMarketData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class tradeExecutor:
    def __init__(self, db_connector: MQSDBConnector, leverage: float = 2.0):
        """Initializes the tradeExecutor and its components."""
        self.dbconn = db_connector
        self.api_auth = APIAuth()
        self.fmp_api_key = self.api_auth.get_fmp_api_key()
        self.marketData = FMPMarketData()
        self.logger = logging.getLogger(__name__)
        self.leverage = leverage
        self.logger.info(f"tradeExecutor initialized with leverage={self.leverage}.")

    def _calculate_buying_power(self, portfolio_equity: float, positions_df: pd.DataFrame, 
                                current_ticker: str, current_ticker_price: float) -> float:
        if positions_df.empty:
            return portfolio_equity * self.leverage

        gross_position_value = 0.0
        for _, row in positions_df.iterrows():
            ticker = row['ticker']
            quantity = row['quantity']
            price = 0.0

            if ticker == current_ticker:
                price = current_ticker_price
            else:
                price = self.get_current_price(ticker)
            
            # --- ROBUSTNESS CHECK ---
            if price <= 0:
                self.logger.critical(
                    f"Could not fetch valid price for position {ticker} during buying power calculation. "
                    f"Temporarily halting new trades by returning zero buying power."
                )
                return 0.0 # Return 0 to prevent trading with an uncertain portfolio state

            gross_position_value += abs(quantity * price)

        buying_power = (portfolio_equity * self.leverage) - gross_position_value
        return max(0, buying_power)

    def execute_trade(self,
                      portfolio_id,
                      ticker,
                      signal_type,
                      confidence,
                      arrival_price,
                      cash,
                      positions, # This is a DataFrame
                      port_notional,
                      ticker_weight,
                      timestamp):
        """
        Calculates and executes a trade using a margin model that
        supports both long and short positions, with buying power constraints on all trades.
        """
        try:
            cash_val = float(cash)
            port_notional_val = float(port_notional)
            arrival_price_val = float(arrival_price)
            confidence_val = float(confidence)
            ticker_weight_val = float(ticker_weight)
        except (ValueError, TypeError) as e:
            self.logger.error(f"Numeric conversion failed: {e}")
            return

        signal_type = signal_type.upper()
        if signal_type not in ('BUY', 'SELL', 'HOLD'):
            return

        confidence_val = max(0.0, min(1.0, confidence_val))
        if signal_type == 'HOLD' or confidence_val == 0.0:
            return

        # 1. Calculate buying power before fetching the final exec_price
        buying_power = self._calculate_buying_power(port_notional_val, positions, ticker, arrival_price_val)

        # 2. Get the final execution price
        exec_price = self.get_current_price(ticker)
        if exec_price <= 0:
            self.logger.error(f"Could not fetch valid execution price for {ticker}. Aborting.")
            return
        
        slippage_bps = ((exec_price / arrival_price_val) - 1) * 10000 if arrival_price_val > 0 else 0


        # 3. Determine target notional
        current_pos_row = positions[positions['ticker'] == ticker]
        current_quantity = current_pos_row['quantity'].iloc[0] if not current_pos_row.empty else 0.0
        current_notional_value = current_quantity * exec_price
        
        target_notional = port_notional_val * ticker_weight_val
        if signal_type == 'SELL':
            target_notional *= -1

        adjustment_notional = target_notional - current_notional_value
        desired_trade_notional = adjustment_notional * confidence_val

        # --- CORRECTED: Unified Constraint & Sizing Logic ---
        if abs(desired_trade_notional) < 1.0: # Ignore trades smaller than $1.00
            return
        
        final_trade_notional = 0.0
        if desired_trade_notional > 0: # This is a BUY operation
            # Constrain by cash AND buying power
            final_trade_notional = min(desired_trade_notional, cash_val, buying_power)
        else: # This is a SELL/SHORT operation
            # Constrain by buying power
            final_trade_notional = min(abs(desired_trade_notional), buying_power)

        if final_trade_notional < 1.0:
            return

        quantity_to_trade = math.floor(final_trade_notional / exec_price)
        if quantity_to_trade == 0:
            return
            
        # --- 4. Execute and Update Database ---
        updated_cash = cash_val
        updated_quantity = current_quantity
        trade_value = quantity_to_trade * exec_price

        if desired_trade_notional > 0: # Finalizing a BUY
            updated_cash = cash_val - trade_value
            updated_quantity = current_quantity + quantity_to_trade
        elif desired_trade_notional < 0: # Finalizing a SELL
            updated_cash = cash_val + trade_value
            updated_quantity = current_quantity - quantity_to_trade
                
        return self.update_database(
            portfolio_id, ticker, signal_type, quantity_to_trade,
            updated_cash, updated_quantity, arrival_price_val, exec_price, slippage_bps, timestamp
        )


    def update_database(self, portfolio_id, ticker, signal_type, quantity_to_trade, 
                         updated_cash, updated_quantity, arrival_price, exec_price, slippage_bps, timestamp):
        """
        Update database tables after trade execution within a single transaction.
        The 'with conn:' block ensures atomic execution (all-or-nothing).
        """
        date_part = timestamp.date()
        trade_notional = abs(quantity_to_trade * exec_price)

        conn = None
        try:
            conn = self.dbconn.get_connection()
            if not conn:
                self.logger.error("Failed to get a database connection from the pool.")
                return {'status': 'error', 'message': 'Database connection failed'}

            # TRANSACTION START
            # Using 'with conn:' automatically starts a transaction.
            # If the block finishes without error, it auto-commits.
            # If an exception is raised anywhere inside, it auto-rollbacks.
            with conn:
                with conn.cursor() as cursor:
                    # 1. Update cash_equity_book
                    cash_query = """
                        INSERT INTO cash_equity_book (timestamp, date, portfolio_id, currency, notional)
                        VALUES (%s, %s, %s, %s, %s)
                    """
                    cash_values = (timestamp, date_part, portfolio_id, 'USD', updated_cash)
                    cursor.execute(cash_query, cash_values)

                    # 2. Update positions_book
                    position_query = """
                        INSERT INTO positions_book (portfolio_id, ticker, quantity, updated_at)
                        VALUES (%s, %s, %s, %s)
                    """
                    position_values = (portfolio_id, ticker, updated_quantity, timestamp)
                    cursor.execute(position_query, position_values)

                    # 3. Insert trade log
                    trade_log_query = """
                        INSERT INTO trade_execution_logs (
                            portfolio_id, ticker, exec_timestamp, side, quantity,
                            arrival_price, exec_price, slippage_bps,
                            notional, notional_local, currency, fx_rate
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    trade_log_values = (
                        portfolio_id, ticker, timestamp, signal_type, quantity_to_trade,
                        arrival_price, exec_price, slippage_bps,
                        None, trade_notional, 'USD', None
                    )
                    cursor.execute(trade_log_query, trade_log_values)
            
            # TRANSACTION END (Commit happens automatically here)

            self.logger.info(f"Database successfully updated for trade: {signal_type} {quantity_to_trade} {ticker} @ {exec_price:.2f}")
            return {'status': 'success', 'quantity': quantity_to_trade, 'updated_cash': updated_cash}

        except Exception as e:
            # If we are here, the 'with conn:' block already triggered a rollback.
            self.logger.exception("Database update transaction failed. Changes rolled back.")
            return {'status': 'error', 'message': str(e)}

        finally:
            if conn:
                self.dbconn.release_connection(conn)

    def get_current_price(self, ticker):
        """
        Fetch real-time stock price for a single ticker using FMP API.
        Returns float: Current price of the ticker, or 0.0 if not found or on error.
        """
        url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}"
        params = {"apikey": self.fmp_api_key}

        try:
            data = self.marketData._make_request(url, params)
            if isinstance(data, list) and data and 'price' in data[0] and data[0]['price'] is not None:
                return float(data[0]['price'])

            self.logger.warning(f"No valid price found for ticker {ticker} in API response. Response: {data}")
            return 0.0

        except Exception as e:
            self.logger.error(f"Price fetch failed for {ticker}: {e}")
            return 0.0
    
    
    def liquidate(self, portfolio_id):
        """
        Liquidates all long and short positions for a given portfolio.
        """
        self.logger.warning(f"Attempting to liquidate all positions for portfolio {portfolio_id}.")
        
        # This logic would need access to the full portfolio state (cash, positions, etc.)
        # A complete implementation is complex as it requires fetching the current state.
        # This is a conceptual fix for the logic.
        
        # 1. Fetch current positions from the database
        # 2. Fetch current cash and calculate port_notional
        
        # For each position:
        #   If quantity > 0 (long), create a SELL signal for the full quantity.
        #   If quantity < 0 (short), create a BUY signal for the absolute quantity.
        #   Call self.execute_trade() for each signal.

        # NOTE: The original `liquidate` function is flawed and needs a significant rewrite
        # to fetch the full portfolio state before it can generate the correct closing trades.
        # The provided code has a bug calling a non-existent `self.sell`.
        self.logger.error("The liquidate function is not fully implemented and contains logical errors.")