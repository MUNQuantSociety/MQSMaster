# pnl/pnl_calculator.py

import time
import logging
from datetime import datetime
from decimal import Decimal

# Add the project root to the Python path if running as a standalone script
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from data_infra.database.MQSDBConnector import MQSDBConnector

class PnLCalculator:
    """
    Calculates and updates Total, Unrealized, and Realized PnL for all portfolios.
    This script runs in a continuous loop to provide real-time PnL updates by
    tracking net capital flow and position cost basis.
    """
    def __init__(self, db_connector: MQSDBConnector, poll_interval: int = 60):
        """
        Initializes the PnLCalculator.

        Args:
            db_connector: An instance of MQSDBConnector.
            poll_interval (int): The interval in seconds to wait between PnL calculations.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_connector = db_connector
        self.poll_interval = poll_interval
        self.running = True

    def _get_portfolio_data(self) -> dict:
        """
        Fetches the latest cash, positions, and market prices for all portfolios.
        """
        # Get latest cash notional for each portfolio
        cash_query = """
        WITH latest_cash AS (
            SELECT portfolio_id, notional, currency,
                   ROW_NUMBER() OVER(PARTITION BY portfolio_id ORDER BY timestamp DESC) as rn
            FROM cash_equity_book
        )
        SELECT portfolio_id, notional, currency FROM latest_cash WHERE rn = 1;
        """
        cash_result = self.db_connector.execute_query(cash_query, fetch='all')
        portfolio_cash = {item['portfolio_id']: {'notional': Decimal(item['notional']), 'currency': item['currency']} for item in cash_result.get('data', [])}

        # Get all current positions
        positions_query = "SELECT portfolio_id, ticker, quantity FROM positions_book WHERE quantity > 0;"
        positions_result = self.db_connector.execute_query(positions_query, fetch='all')
        
        portfolio_positions = {}
        all_tickers = set()
        for pos in positions_result.get('data', []):
            pid = pos['portfolio_id']
            if pid not in portfolio_positions:
                portfolio_positions[pid] = []
            portfolio_positions[pid].append({'ticker': pos['ticker'], 'quantity': Decimal(pos['quantity'])})
            all_tickers.add(pos['ticker'])

        # Get latest market price for all relevant tickers
        market_prices = {}
        if all_tickers:
            placeholders = ', '.join(['%s'] * len(all_tickers))
            market_data_query = f"""
            WITH latest_prices AS (
                SELECT ticker, close_price,
                       ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY timestamp DESC) as rn
                FROM market_data WHERE ticker IN ({placeholders})
            )
            SELECT ticker, close_price FROM latest_prices WHERE rn = 1;
            """
            market_data_result = self.db_connector.execute_query(market_data_query, tuple(all_tickers), fetch='all')
            market_prices = {item['ticker']: Decimal(item['close_price']) for item in market_data_result.get('data', [])}

        return {'cash': portfolio_cash, 'positions': portfolio_positions, 'prices': market_prices}

    def _get_net_capital_injected(self) -> dict:
        """
        Calculates the net capital added or withdrawn for each portfolio.
        Capital movements are identified by the 'USD_CASH' ticker.
        """
        query = """
            SELECT
                portfolio_id,
                SUM(CASE WHEN side = 'BUY' THEN notional_local ELSE -notional_local END) as net_capital
            FROM trade_execution_logs
            WHERE ticker = 'USD_CASH'
            GROUP BY portfolio_id;
        """
        result = self.db_connector.execute_query(query, fetch='all')
        return {item['portfolio_id']: Decimal(item['net_capital']) for item in result.get('data', [])}

    def _get_positions_cost_basis(self) -> dict:
        """
        Calculates the average cost basis for each ticker in each portfolio.
        This uses a weighted average of all BUY trades.
        NOTE: This is a simplification. A FIFO or specific-lot method would be more accurate
        but is significantly more complex to implement.
        """
        query = """
            SELECT
                portfolio_id,
                ticker,
                SUM(quantity * exec_price) / SUM(quantity) as avg_cost
            FROM trade_execution_logs
            WHERE side = 'BUY' AND ticker != 'USD_CASH'
            GROUP BY portfolio_id, ticker;
        """
        result = self.db_connector.execute_query(query, fetch='all')
        cost_basis = {}
        for item in result.get('data', []):
            pid = item['portfolio_id']
            if pid not in cost_basis:
                cost_basis[pid] = {}
            cost_basis[pid][item['ticker']] = Decimal(item['avg_cost'])
        return cost_basis

    def _calculate_and_update_pnl(self):
        """
        Calculates the PnL for each portfolio and updates the pnl_book table.
        """
        self.logger.info("Starting PnL calculation cycle.")
        
        # Fetch all required data in batch
        portfolio_data = self._get_portfolio_data()
        net_capital_data = self._get_net_capital_injected()
        cost_basis_data = self._get_positions_cost_basis()

        all_portfolio_ids = set(portfolio_data['cash'].keys()) | set(portfolio_data['positions'].keys())
        pnl_updates = []

        for portfolio_id in all_portfolio_ids:
            # --- 1. Calculate Unrealized PnL ---
            positions_market_value = Decimal('0.0')
            positions_cost_basis = Decimal('0.0')
            unrealized_pnl = Decimal('0.0')

            if portfolio_id in portfolio_data['positions']:
                for position in portfolio_data['positions'][portfolio_id]:
                    ticker = position['ticker']
                    quantity = position['quantity']
                    
                    current_price = portfolio_data['prices'].get(ticker)
                    avg_cost = cost_basis_data.get(portfolio_id, {}).get(ticker)

                    if current_price:
                        positions_market_value += quantity * current_price
                    else:
                        self.logger.warning(f"No market price for {ticker} in portfolio {portfolio_id}.")

                    if avg_cost:
                        positions_cost_basis += quantity * avg_cost
                    else:
                        self.logger.warning(f"No cost basis for {ticker} in portfolio {portfolio_id}.")
            
            unrealized_pnl = positions_market_value - positions_cost_basis

            # --- 2. Calculate Total PnL ---
            cash_balance = portfolio_data['cash'].get(portfolio_id, {}).get('notional', Decimal('0.0'))
            current_valuation = cash_balance + positions_market_value
            net_capital = net_capital_data.get(portfolio_id, Decimal('0.0'))
            total_pnl = current_valuation - net_capital

            # --- 3. Calculate Realized PnL ---
            realized_pnl = total_pnl - unrealized_pnl
            
            exec_timestamp = datetime.now()
            pnl_updates.append({
                'portfolio_id': portfolio_id,
                'timestamp': exec_timestamp,
                'date': exec_timestamp.date(),
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'fx_rate': Decimal('1.0'),
                'currency': portfolio_data['cash'].get(portfolio_id, {}).get('currency', 'USD'),
                'notional': current_valuation, # Total portfolio value
            })

        if pnl_updates:
            pnl_insert_query = """
            INSERT INTO pnl_book (portfolio_id, timestamp, date, realized_pnl, unrealized_pnl, fx_rate, currency, notional)
            VALUES (%(portfolio_id)s, %(timestamp)s, %(date)s, %(realized_pnl)s, %(unrealized_pnl)s, %(fx_rate)s, %(currency)s, %(notional)s);
            """
            try:
                for update in pnl_updates:
                    self.db_connector.execute_query(pnl_insert_query, update)
                self.logger.info(f"Successfully updated PnL for {len(pnl_updates)} portfolios.")
            except Exception as e:
                self.logger.exception(f"Failed to update pnl_book: {e}")

    def run(self):
        self.logger.info("Starting PnL Calculator...")
        while self.running:
            try:
                start_time = time.time()
                self._calculate_and_update_pnl()
                elapsed_time = time.time() - start_time
                self.logger.info(f"PnL cycle finished in {elapsed_time:.2f}s. Sleeping for {self.poll_interval - elapsed_time:.2f}s.")
                time.sleep(max(0, self.poll_interval - elapsed_time))
            except KeyboardInterrupt:
                self.logger.info("PnL Calculator stopped by user.")
                self.running = False
            except Exception as e:
                self.logger.exception(f"An error occurred in the PnL Calculator run loop: {e}")
                time.sleep(self.poll_interval)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    db_connector = MQSDBConnector()
    pnl_calculator = PnLCalculator(db_connector=db_connector, poll_interval=60)
    pnl_calculator.run()