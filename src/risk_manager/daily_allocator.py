import json
import logging
from datetime import datetime
import os
import sys
from decimal import Decimal
import pytz # Import the pytz library

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from common.database.MQSDBConnector import MQSDBConnector
from data_management.marketData.fmpMarketData import FMPMarketData

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DailyAllocator:
    def __init__(self, config_path: str):
        self.db_connector = MQSDBConnector()
        self.market_data = FMPMarketData()
        self.db_timezone = pytz.timezone("America/New_York") # Define the target timezone
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.master_portfolio_id = self.config['master_portfolio_id']
            self.strategy_portfolios = self.config['portfolio_weights']
            self.currency = self.config['currency']
        except FileNotFoundError:
            logger.exception(f"Configuration file not found at {config_path}")
            sys.exit(1)
        except (KeyError, json.JSONDecodeError) as e:
            logger.exception(f"Error parsing configuration file: {e}")
            sys.exit(1)

    def _get_current_cash(self, portfolio_id: str) -> Decimal:
        # ... (no changes in this function)
        query = "SELECT notional FROM cash_equity_book WHERE portfolio_id = %s ORDER BY timestamp DESC LIMIT 1;"
        result = self.db_connector.execute_query(query, (portfolio_id,), fetch='one')
        
        if result and result.get('data'):
            first_row = result['data'][0]
            return Decimal(first_row['notional'])
        else:
            return Decimal('0.0')

    def _get_positions_value(self, portfolio_id: str) -> Decimal:
        # ... (no changes in this function)
        query = "SELECT ticker, quantity FROM positions_book WHERE portfolio_id = %s AND quantity > 0;"
        result = self.db_connector.execute_query(query, (portfolio_id,), fetch='all')

        if not result or not result.get('data'):
            return Decimal('0.0')

        positions = result['data']
        if not positions:
            return Decimal('0.0')

        tickers = [pos['ticker'] for pos in positions]
        
        prices = {}
        for ticker in tickers:
            price = self.market_data.get_current_price(ticker)
            if price is not None:
                prices[ticker] = price

        total_value = Decimal('0.0')
        for pos in positions:
            ticker = pos['ticker']
            quantity = Decimal(pos['quantity'])
            price = Decimal(prices.get(ticker, '0.0'))

            if price > Decimal('0.0'):
                total_value += quantity * price
            else:
                logger.warning(
                    f"Could not fetch a valid price for {ticker} in portfolio {portfolio_id}. "
                    "It will be valued at 0."
                )
                
        return total_value

    def _execute_internal_transfer(self, cursor, from_portfolio: str, to_portfolio: str, amount: Decimal, new_from_balance: Decimal, new_to_balance: Decimal):
        """
        Executes database writes for an internal transfer using a provided cursor.
        """
        if amount <= 0:
            return

        # Generate a timezone-aware timestamp
        exec_timestamp = datetime.now(self.db_timezone)
        date_part = exec_timestamp.date()
        
        cursor.execute("""
            INSERT INTO trade_execution_logs (
                portfolio_id, ticker, exec_timestamp, side, quantity, 
                arrival_price, exec_price, notional_local, currency, notional
            ) VALUES (%s, %s, %s, %s, %s, 1.0, 1.0, %s, %s, %s)
        """, (from_portfolio, f"{self.currency}_CASH", exec_timestamp, 'SELL', amount, amount, self.currency, None))

        cursor.execute("""
            INSERT INTO trade_execution_logs (
                portfolio_id, ticker, exec_timestamp, side, quantity, 
                arrival_price, exec_price, notional_local, currency, notional
            ) VALUES (%s, %s, %s, %s, %s, 1.0, 1.0, %s, %s, %s)
        """, (to_portfolio, f"{self.currency}_CASH", exec_timestamp, 'BUY', amount, amount, self.currency, None))

        cash_update_query = "INSERT INTO cash_equity_book (timestamp, date, portfolio_id, currency, notional) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(cash_update_query, (exec_timestamp, date_part, from_portfolio, self.currency, new_from_balance))
        cursor.execute(cash_update_query, (exec_timestamp, date_part, to_portfolio, self.currency, new_to_balance))
        
        logger.info(f"Prepared DB operations for transfer of {amount:.2f} from portfolio {from_portfolio} to {to_portfolio}")

    def initialize_new_portfolios(self):
        """Checks for and initializes new portfolios from the config file."""
        logger.info("Starting portfolio initialization check...")
        portfolio_ids_in_config = list(self.strategy_portfolios.keys())
        query = "SELECT DISTINCT portfolio_id FROM cash_equity_book WHERE portfolio_id = ANY(%s);"
        result = self.db_connector.execute_query(query, (portfolio_ids_in_config,), fetch='all')
        
        existing_portfolios = {row['portfolio_id'] for row in result['data']} if result and result.get('data') else set()
        new_portfolios = set(portfolio_ids_in_config) - existing_portfolios

        if not new_portfolios:
            logger.info("No new portfolios to initialize.")
            return

        conn = None
        try:
            conn = self.db_connector.get_connection()
            with conn.cursor() as cursor:
                for portfolio_id in new_portfolios:
                    logger.info(f"New portfolio detected: {portfolio_id}. Initializing with nominal funding.")
                    # Generate a timezone-aware timestamp
                    exec_timestamp = datetime.now(self.db_timezone)
                    cursor.execute("""
                        INSERT INTO trade_execution_logs (
                            portfolio_id, ticker, exec_timestamp, side, quantity, 
                            arrival_price, exec_price, notional_local, currency, notional
                        ) VALUES (%s, %s, %s, %s, 1.0, 1.0, 1.0, 1.0, %s, %s)
                    """, (portfolio_id, f"{self.currency}_CASH", exec_timestamp, 'BUY', self.currency, None))
                    
                    cursor.execute("""
                        INSERT INTO cash_equity_book (timestamp, date, portfolio_id, currency, notional)
                        VALUES (%s, %s, %s, %s, 1.0)
                    """, (exec_timestamp, exec_timestamp.date(), portfolio_id, self.currency))
            conn.commit()
            logger.info(f"Successfully initialized {len(new_portfolios)} new portfolios.")
        except Exception as e:
            logger.exception("Failed during portfolio initialization. Rolling back.")
            if conn: conn.rollback()
        finally:
            if conn: self.db_connector.release_connection(conn)

    def run_allocation(self):
        # ... (no changes in this function)
        logger.info("Starting daily capital allocation...")

        master_cash = self._get_current_cash(self.master_portfolio_id)
        
        portfolio_valuations = {}
        total_equity = master_cash
        for pid in self.strategy_portfolios.keys():
            cash = self._get_current_cash(pid)
            positions_value = self._get_positions_value(pid)
            portfolio_total = cash + positions_value
            total_equity += portfolio_total
            portfolio_valuations[pid] = {'cash': cash, 'positions': positions_value, 'total': portfolio_total}
            logger.info(f"Strategy Portfolio ({pid}) cash: {cash:.2f}, positions: {positions_value:.2f}, total: {portfolio_total:.2f}")

        logger.info(f"Master Portfolio ({self.master_portfolio_id}) initial cash: {master_cash:.2f}")
        logger.info(f"Total System Equity calculated: {total_equity:.2f}")

        conn = None
        try:
            conn = self.db_connector.get_connection()
            with conn.cursor() as cursor:
                in_memory_master_cash = master_cash
                for pid, weight in self.strategy_portfolios.items():
                    weight = Decimal(str(weight))
                    current_total_value = portfolio_valuations[pid]['total']
                    target_value = total_equity * weight
                    adjustment = target_value - current_total_value
                    
                    logger.info(f"Portfolio {pid}: Current Value={current_total_value:.2f}, Target Value={target_value:.2f}, Adjustment={adjustment:.2f}")
                    if adjustment.is_zero():
                        continue
                    
                    cash_to_transfer = abs(adjustment)
                    if adjustment > 0:
                        new_master_balance = in_memory_master_cash - cash_to_transfer
                        new_portfolio_balance = portfolio_valuations[pid]['cash'] + cash_to_transfer
                        self._execute_internal_transfer(cursor, self.master_portfolio_id, pid, cash_to_transfer, new_master_balance, new_portfolio_balance)
                        in_memory_master_cash = new_master_balance
                        
                    elif adjustment < 0:
                        new_master_balance = in_memory_master_cash + cash_to_transfer
                        new_portfolio_balance = portfolio_valuations[pid]['cash'] - cash_to_transfer
                        self._execute_internal_transfer(cursor, pid, self.master_portfolio_id, cash_to_transfer, new_portfolio_balance, new_master_balance)
                        in_memory_master_cash = new_master_balance

            conn.commit()
            logger.info("Daily capital allocation transaction completed successfully.")
        except Exception as e:
            logger.exception("Failed during capital allocation transaction. Rolling back.")
            if conn: conn.rollback()
        finally:
            if conn: self.db_connector.release_connection(conn)


if __name__ == '__main__':
    # (No changes to the main execution block)
    config_file_path = os.path.join(project_root, 'portfolios', 'portfolio_manager_config.json')
    
    allocator = DailyAllocator(config_path=config_file_path)
    
    allocator.initialize_new_portfolios()
    
    allocator.run_allocation()