# pnl/pnl_calculator.py

import time
import logging
from datetime import datetime
from decimal import Decimal
from collections import deque

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data_infra.database.MQSDBConnector import MQSDBConnector

class PnLCalculator:
    """
    Calculates and updates Total, Unrealized, and Realized PnL for all portfolios.
    This script uses a FIFO (First-In, First-Out) accounting method for industry-standard
    accuracy in cost basis and PnL calculations.
    """
    def __init__(self, db_connector: MQSDBConnector, poll_interval: int = 60):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_connector = db_connector
        self.poll_interval = poll_interval
        self.running = True

    def _get_latest_portfolio_state(self) -> dict:
        """
        Fetches the latest portfolio state (cash and positions) for all portfolios
        atomically to prevent race conditions.
        """
        # This single, unified query fetches the latest cash balance and an aggregated
        # list of current positions for every portfolio in one atomic operation.
        # It uses a FULL OUTER JOIN to correctly handle portfolios that may only have
        # cash or only have positions.
        atomic_state_query = """
        WITH latest_cash AS (
            -- 1. Get the most recent cash balance for each portfolio.
            --    The 'id' column is used as a tie-breaker for records with the same timestamp.
            SELECT
                portfolio_id,
                notional,
                currency,
                ROW_NUMBER() OVER(PARTITION BY portfolio_id ORDER BY "timestamp" DESC, id DESC) as rn
            FROM
                cash_equity_book
        ),
        cash_snapshot AS (
            -- Filter to get only the single latest entry for each portfolio
            SELECT portfolio_id, notional, currency FROM latest_cash WHERE rn = 1
        ),
        positions_snapshot AS (
            -- 2. Get all current (non-zero quantity) positions for each portfolio,
            --    aggregated into a single JSON array.
            SELECT
                portfolio_id,
                json_agg(json_build_object('ticker', ticker, 'quantity', quantity)) AS positions
            FROM (
                -- Subquery to get the most recent state of each position
                SELECT DISTINCT ON (portfolio_id, ticker)
                    portfolio_id,
                    ticker,
                    quantity
                FROM
                    positions_book
                ORDER BY
                    portfolio_id, ticker, updated_at DESC
            ) AS latest_positions
            WHERE
                quantity != 0
            GROUP BY
                portfolio_id
        )
        -- 3. Combine cash and position snapshots atomically.
        --    COALESCE is used to correctly retrieve the portfolio_id from either side of the join.
        SELECT
            COALESCE(cs.portfolio_id, ps.portfolio_id) AS portfolio_id,
            cs.notional AS cash_notional,
            cs.currency,
            ps.positions
        FROM
            cash_snapshot cs
        FULL OUTER JOIN
            positions_snapshot ps ON cs.portfolio_id = ps.portfolio_id;
        """

        state_result = self.db_connector.execute_query(atomic_state_query, fetch='all')

        portfolio_cash = {}
        portfolio_positions = {}
        all_tickers = set()

        # 4. Parse the results of the unified query
        for portfolio_state in state_result.get('data', []):
            pid = portfolio_state['portfolio_id']

            # Populate cash information if it exists
            if portfolio_state.get('cash_notional') is not None:
                portfolio_cash[pid] = {
                    'notional': Decimal(portfolio_state['cash_notional']),
                    'currency': portfolio_state['currency']
                }

            # Populate positions information if it exists, and collect tickers
            if portfolio_state.get('positions') is not None:
                positions_list = portfolio_state['positions']
                portfolio_positions[pid] = []
                for pos in positions_list:
                    portfolio_positions[pid].append({
                        'ticker': pos['ticker'],
                        'quantity': Decimal(pos['quantity'])
                    })
                    all_tickers.add(pos['ticker'])

        # 5. Get latest market prices for all relevant tickers (this can remain separate)
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

    def _calculate_fifo_pnl_and_cost_basis(self, portfolio_id: str, ticker: str) -> dict:
        """
        Calculates realized PnL and the cost basis of current holdings for a single
        ticker using the FIFO method.
        """
        query = """
            SELECT exec_timestamp, side, quantity, exec_price
            FROM trade_execution_logs
            WHERE portfolio_id = %s AND ticker = %s AND ticker != 'USD_CASH'
            ORDER BY exec_timestamp ASC;
        """
        trades_result = self.db_connector.execute_query(query, (portfolio_id, ticker), fetch='all')
        trades = trades_result.get('data', [])

        buy_lots = deque()
        realized_pnl_for_ticker = Decimal('0.0')

        for trade in trades:
            side = trade['side']
            quantity = Decimal(trade['quantity'])
            price = Decimal(trade['exec_price'])

            if side == 'BUY':
                buy_lots.append({'quantity': quantity, 'price': price})
            elif side == 'SELL':
                quantity_to_sell = quantity
                while quantity_to_sell > 0 and buy_lots:
                    oldest_lot = buy_lots[0]
                    
                    if oldest_lot['quantity'] <= quantity_to_sell:
                        # Sell the entire oldest lot
                        realized_pnl_for_ticker += oldest_lot['quantity'] * (price - oldest_lot['price'])
                        quantity_to_sell -= oldest_lot['quantity']
                        buy_lots.popleft()
                    else:
                        # Sell a portion of the oldest lot
                        realized_pnl_for_ticker += quantity_to_sell * (price - oldest_lot['price'])
                        oldest_lot['quantity'] -= quantity_to_sell
                        quantity_to_sell = 0
        
        # Calculate cost basis of remaining (unrealized) lots
        current_cost_basis = Decimal('0.0')
        total_quantity = Decimal('0.0')
        for lot in buy_lots:
            current_cost_basis += lot['quantity'] * lot['price']
            total_quantity += lot['quantity']

        return {
            'realized_pnl': realized_pnl_for_ticker,
            'cost_basis': current_cost_basis,
            'quantity': total_quantity
        }

    def _calculate_and_update_pnl(self):
        """
        Calculates PnL for each portfolio and updates the pnl_book table.
        """
        self.logger.info("Starting PnL calculation cycle.")
        
        state = self._get_latest_portfolio_state()
        all_portfolio_ids = set(state['cash'].keys()) | set(state['positions'].keys())
        pnl_updates = []

        for portfolio_id in all_portfolio_ids:
            total_realized_pnl = Decimal('0.0')
            total_unrealized_pnl = Decimal('0.0')
            total_market_value = Decimal('0.0')
            
            # 1. Calculate PnL from equity positions
            if portfolio_id in state['positions']:
                for position in state['positions'][portfolio_id]:
                    ticker = position['ticker']
                    
                    # Skip cash-like instruments from trade-based PnL calc
                    if ticker == 'USD_CASH':
                        continue

                    fifo_results = self._calculate_fifo_pnl_and_cost_basis(portfolio_id, ticker)
                    total_realized_pnl += fifo_results['realized_pnl']

                    current_price = state['prices'].get(ticker)
                    if current_price:
                        market_value = fifo_results['quantity'] * current_price
                        total_market_value += market_value
                        total_unrealized_pnl += market_value - fifo_results['cost_basis']
                    else:
                        self.logger.warning(f"No market price for {ticker} in portfolio {portfolio_id}. Cannot calculate unrealized PnL.")

            # 2. Finalize portfolio-level numbers
            cash_balance = state['cash'].get(portfolio_id, {}).get('notional', Decimal('0.0'))
            current_valuation = cash_balance + total_market_value
            
            exec_timestamp = datetime.now()
            pnl_updates.append({
                'portfolio_id': portfolio_id,
                'timestamp': exec_timestamp,
                'date': exec_timestamp.date(),
                'realized_pnl': total_realized_pnl,
                'unrealized_pnl': total_unrealized_pnl,
                'fx_rate': Decimal('1.0'), # Assuming USD for now
                'currency': state['cash'].get(portfolio_id, {}).get('currency', 'USD'),
                'notional': current_valuation
            })

        # 3. Perform bulk insert into the database
        if pnl_updates:
            self.logger.info(f"Updating PnL for {len(pnl_updates)} portfolios.")
            # **FIXED**: Removed ON CONFLICT logic as requested.
            # This will perform a simple bulk insert.
            result = self.db_connector.bulk_inject_to_db(
                'pnl_book', 
                pnl_updates
            )
            if result['status'] == 'error':
                 self.logger.error(f"Failed to bulk update pnl_book: {result['message']}")
            else:
                 self.logger.info(f"Successfully processed PnL updates: {result['message']}")


    def run(self):
        self.logger.info("Starting PnL Calculator...")
        while self.running:
            try:
                start_time = time.time()
                self._calculate_and_update_pnl()
                elapsed_time = time.time() - start_time
                self.logger.info(f"PnL cycle finished in {elapsed_time:.2f}s. Sleeping for {max(0, self.poll_interval - elapsed_time):.2f}s.")
                time.sleep(max(0, self.poll_interval - elapsed_time))
            except KeyboardInterrupt:
                self.logger.info("PnL Calculator stopped by user.")
                self.running = False
            except Exception as e:
                self.logger.exception(f"An error occurred in the PnL Calculator run loop: {e}")
                time.sleep(self.poll_interval)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    db_conn = MQSDBConnector()
    pnl_calculator = PnLCalculator(db_connector=db_conn, poll_interval=60)
    pnl_calculator.run()
