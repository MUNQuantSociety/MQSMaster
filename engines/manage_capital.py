# engines/manage_capital.py

import argparse
import logging
from datetime import datetime
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from data_infra.database.MQSDBConnector import MQSDBConnector

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
MASTER_PORTFOLIO_ID = "0"
CASH_TICKER = "USD_CASH"
CURRENCY = "USD"

def get_current_cash(db_connector: MQSDBConnector, portfolio_id: str) -> float:
    """Fetches the most recent cash balance for a given portfolio."""
    # This function correctly queries 'cash_equity_book' which uses 'notional' for its balance. No changes needed here.
    query = """
        SELECT notional FROM cash_equity_book
        WHERE portfolio_id = %s
        ORDER BY timestamp DESC
        LIMIT 1;
    """
    try:
        result = db_connector.execute_query(query, (portfolio_id,), fetch='one')
        if result and result['data']:
            return float(result['data'][0]['notional'])
        return 0.0  # No cash balance found, assume 0
    except Exception as e:
        logger.exception(f"Failed to fetch cash for portfolio {portfolio_id}: {e}")
        return 0.0


def update_capital(db_connector: MQSDBConnector, amount: float, action: str):
    """Adds or withdraws capital from the master portfolio."""
    action = action.upper()
    if action not in ['ADD', 'WITHDRAW']:
        logger.error(f"Invalid action: {action}. Must be 'ADD' or 'WITHDRAW'.")
        return

    conn = None
    try:
        conn = db_connector.get_connection()
        if not conn:
            logger.error("Failed to get a database connection.")
            return

        with conn.cursor() as cursor:
            # Get current cash balance
            current_balance = get_current_cash(db_connector, MASTER_PORTFOLIO_ID)

            # Determine trade side and calculate new balance
            if action == 'ADD':
                side = 'BUY'
                new_balance = current_balance + amount
            else: # WITHDRAW
                if amount > current_balance:
                    logger.error(f"Withdrawal amount {amount} exceeds current balance {current_balance}.")
                    return
                side = 'SELL'
                new_balance = current_balance - amount

            exec_timestamp = datetime.now()
            date_part = exec_timestamp.date()

            # 1. Insert into trade_execution_logs
            # Only notional_local is touched. notional (CAD value) is set to NULL
            # because no FX rate is implemented.
            trade_log_query = """
                INSERT INTO trade_execution_logs (
                    portfolio_id, ticker, exec_timestamp, side, quantity,
                    arrival_price, exec_price, slippage_bps, notional, notional_local, currency
                ) VALUES (%s, %s, %s, %s, %s, 1.0, 1.0, 0, %s, %s, %s);
            """
            trade_log_values = (
                MASTER_PORTFOLIO_ID, CASH_TICKER, exec_timestamp, side, amount,
                None, amount, CURRENCY # Set notional to NULL, and notional_local to the transaction amount
            )
            cursor.execute(trade_log_query, trade_log_values)
            logger.info(f"Logged funding transaction: {side} {amount} {CASH_TICKER} for portfolio {MASTER_PORTFOLIO_ID}")


            # 2. Insert new balance into cash_equity_book
            # This table correctly uses 'notional' for its balance as it has a currency column. No changes needed.
            cash_query = """
                INSERT INTO cash_equity_book (timestamp, date, portfolio_id, currency, notional)
                VALUES (%s, %s, %s, %s, %s);
            """
            cash_values = (exec_timestamp, date_part, MASTER_PORTFOLIO_ID, CURRENCY, new_balance)
            cursor.execute(cash_query, cash_values)
            logger.info(f"Updated cash book for portfolio {MASTER_PORTFOLIO_ID} to new balance: {new_balance:.2f}")

        conn.commit()
        logger.info(f"Capital management operation '{action}' for amount {amount} completed successfully.")

    except Exception as e:
        logger.exception("Database transaction failed. Rolling back changes.")
        if conn:
            conn.rollback()
    finally:
        if conn:
            db_connector.release_connection(conn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Manage capital for the trading system.")
    parser.add_argument('--action', type=str, required=True, choices=['ADD', 'WITHDRAW'],
                        help="The action to perform: ADD or WITHDRAW capital.")
    parser.add_argument('--amount', type=float, required=True,
                        help="The amount of capital to add or withdraw.")

    args = parser.parse_args()

    if args.amount <= 0:
        print("Error: Amount must be positive.")
        sys.exit(1)

    db_conn = MQSDBConnector()
    update_capital(db_conn, args.amount, args.action)