# main.py
# Use main_backtest.py for backtesting purposes.
# main.py is reserved for live trading purposes.

import logging

import os
import sys
REPO_ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.insert(0, REPO_ROOT)

from data_infra.database.MQSDBConnector import MQSDBConnector
from data_infra.tradingOps.realtime.live import tradeExecutor # For live trading
from Backtest.runner import BacktestRunner  # This is the backtest runner

# Import portfolio classes, not instances
from portfolios.portfolio_1.strategy import SimpleMomentum
from portfolios.portfolio_2.strategy import SimpleMomentum2

from engines.run_engine import RunEngine
from engines.backtest_engine import BacktestEngine  # This is the backtest engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    """
    Main entry point for the MQS Trading System.
    """

    dbconn = MQSDBConnector()
    live_trade_executor = tradeExecutor(dbconn)

    run_engine = RunEngine(db_connector=dbconn, executor=live_trade_executor, debug=False)
    # Pass a list of the portfolio classes you want to run
    run_engine.load_portfolios([
        SimpleMomentum,
        SimpleMomentum2
    ])

    
    # Start all loaded portfolios. This will block until Ctrl+C is pressed.
    run_engine.run()
    
if __name__ == '__main__':
    main()