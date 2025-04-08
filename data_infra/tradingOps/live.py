from datetime import datetime
import logging
from data_infra.brokerAPI.brokerClient import brokerClient
from data_infra.database.schemaDefinitions import MQSDBConnector

class tradeExecutor:
    def __init__(self):
        self.table = MQSDBConnector()
        self.broker_client = brokerClient()  

    # --- Helper Methods ---

    def _get_cash_balance(self, portfolio_id):
        """Retrieve the latest cash balance (notional) for the portfolio."""
        sql_cash = """
            SELECT *
            FROM cash_equity_book
            WHERE portfolio_id = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """
        cash_result = self.table.execute_query(sql_cash, values=(portfolio_id,), fetch=True)
        if cash_result['status'] != 'success' or not cash_result['data']:
            logging.error(f"Could not retrieve cash_equity_book for portfolio {portfolio_id}")
            return None
        return float(cash_result['data'][0]['notional'])

    def _update_cash_balance(self, portfolio_id, new_balance):
        """Update the cash_equity_book with a new notional value."""
        return self.table.update_data(
            table='cash_equity_book',
            data={'notional': new_balance},
            conditions={'portfolio_id': portfolio_id}
        )

    def _update_positions(self, portfolio_id, ticker, quantity_change):
        """
        Update the positions table by incrementing (or decrementing) the quantity for a given ticker.
        Use a positive quantity_change for BUY and negative for SELL.
        """
        sql_update_position = """
            INSERT INTO positions (portfolio_id, ticker, quantity)
            VALUES (%s, %s, %s)
            ON CONFLICT (portfolio_id, ticker)
            DO UPDATE SET quantity = positions.quantity + EXCLUDED.quantity,
                          updated_at = NOW()
        """
        return self.table.execute_query(sql_update_position, values=(portfolio_id, ticker, quantity_change))

    def _insert_trade_log(self, portfolio_id, ticker, side, quantity, price):
        """Insert a trade log record into trade_execution_logs."""
        trade_data = {
            'portfolio_id': portfolio_id,
            'ticker': ticker,
            'side': side,
            'quantity': quantity,
            'price_last': price,
            'exec_timestamp': datetime.now()
        }
        return self.table.inject_to_db('trade_execution_logs', trade_data)

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

    # --- Trading Methods ---

    def execute_trade(self, portfolio_id, action, ticker, quantity):
        """
        Execute a trade based on the action (BUY or SELL).
        """
        if action == 'BUY':
            self.buy(portfolio_id, ticker, quantity)
        elif action == 'SELL':
            self.sell(portfolio_id, ticker, quantity)
        else:
            logging.error(f"Invalid action: {action}. Must be BUY or SELL.")

    def buy(self, portfolio_id, ticker, quantity):
        """
        Execute a BUY trade:
          1. Check available cash.
          2. Retrieve current price and calculate total cost.
          3. Insert a BUY record in trade_execution_logs.
          4. Update cash_equity_book.
          5. Update positions table (increment the holding).
        """
        cash = self._get_cash_balance(portfolio_id)
        if cash is None:
            return

        price = self.get_current_price(ticker)
        if price == 0:
            logging.error(f"Current price for {ticker} is 0. Cannot execute BUY.")
            return

        total_cost = price * quantity
        if total_cost > cash:
            logging.error(
                f"Insufficient funds to buy {quantity} shares of {ticker}. "
                f"Required: {total_cost}, Available: {cash}"
            )
            return

        # Insert trade log.
        result = self._insert_trade_log(portfolio_id, ticker, 'BUY', quantity, price)
        if result['status'] != 'success':
            logging.error(f"Failed to insert BUY record: {result['message']}")
            return

        # Update cash balance.
        new_cash_balance = cash - total_cost
        cash_update = self._update_cash_balance(portfolio_id, new_cash_balance)
        if cash_update['status'] != 'success':
            logging.error(f"Failed to update cash_equity_book: {cash_update['message']}")
            return

        # Update positions table.
        pos_update = self._update_positions(portfolio_id, ticker, quantity)
        if pos_update['status'] != 'success':
            logging.error(f"Failed to update positions for BUY: {pos_update['message']}")
            return

        logging.info(
            f"BUY executed: {quantity} shares of {ticker} at {price}. "
            f"New cash balance: {new_cash_balance}"
        )

    def sell(self, portfolio_id, ticker, quantity):
        """
        Execute a SELL trade:
          1. Retrieve current available shares from positions.
          2. Verify enough shares are available.
          3. Retrieve current price and calculate total proceeds.
          4. Insert a SELL record in trade_execution_logs.
          5. Update cash_equity_book.
          6. Update positions table (decrement the holding).
        """
        # Get current share holding from positions.
        sql_get_position = """
            SELECT quantity
            FROM positions
            WHERE portfolio_id = %s AND ticker = %s
        """
        pos_result = self.table.execute_query(sql_get_position, values=(portfolio_id, ticker), fetch=True)
        if pos_result['status'] != 'success' or not pos_result['data']:
            logging.error(f"No position record found for ticker {ticker} in portfolio {portfolio_id}")
            return

        current_shares = float(pos_result['data'][0]['quantity'])
        if quantity > current_shares:
            logging.error(f"Not enough shares to sell {quantity} of {ticker}. Available: {current_shares}")
            return

        price = self.get_current_price(ticker)
        if price == 0:
            logging.error(f"Current price for {ticker} is 0. Cannot execute SELL.")
            return

        total_proceeds = price * quantity

        # Insert trade log.
        result = self._insert_trade_log(portfolio_id, ticker, 'SELL', quantity, price)
        if result['status'] != 'success':
            logging.error(f"Failed to insert SELL record: {result['message']}")
            return

        # Update cash balance.
        cash = self._get_cash_balance(portfolio_id)
        if cash is None:
            return
        new_cash_balance = cash + total_proceeds
        cash_update = self._update_cash_balance(portfolio_id, new_cash_balance)
        if cash_update['status'] != 'success':
            logging.error(f"Failed to update cash_equity_book: {cash_update['message']}")
            return

        # Update positions table (decrement the holding).
        pos_update = self._update_positions(portfolio_id, ticker, -quantity)
        if pos_update['status'] != 'success':
            logging.error(f"Failed to update positions for SELL: {pos_update['message']}")
            return

        logging.info(
            f"SELL executed: {quantity} shares of {ticker} at {price}. "
            f"New cash balance: {new_cash_balance}"
        )

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

        for pos in pos_result['data']:
            ticker = pos['ticker']
            available_quantity = float(pos['quantity'])
            if available_quantity > 0:
                # Call the sell method for each ticker to update all tables.
                self.sell(portfolio_id, ticker, available_quantity)

        logging.info(f"All positions for portfolio {portfolio_id} have been liquidated.")
