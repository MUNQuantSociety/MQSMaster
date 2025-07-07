# main.py
# Use main_backtest.py for backtesting purposes.
# main.py is reserved for live trading purposes.

import logging

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data_infra.database.MQSDBConnector import MQSDBConnector
from data_infra.tradingOps.realtime.live import tradeExecutor # For live trading
# Import portfolio classes, not instances
from portfolios.portfolio_1.strategy import VolMomentum
from portfolios.portfolio_2.strategy import MomentumStrategy

from engines.run_engine import RunEngine

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
        VolMomentum,
        MomentumStrategy
    ])

    
    # Start all loaded portfolios. This will block until Ctrl+C is pressed.
    run_engine.run()
    
if __name__ == '__main__':
    main()