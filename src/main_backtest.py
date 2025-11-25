# This is where backtests are setup.
# Simply add you portfolio class to the list of portfolio classes in the `setup` method, line 28.

# High level overview of how to set up a backtest:

# 1. Load your portfolio through imports such as `from portfolios.portfolio_1.strategy import VolMomentum`
# 2. Set up the backtest engine with the database connector and backtest executor.
# 3. Call the `setup` method with the portfolio class, start date, end date, and initial capital.
# 4. Call the `run` method to execute the backtest.
# src/main_backtest.py

import logging
from common.database.MQSDBConnector import MQSDBConnector
from portfolios.portfolio_2.strategy import MomentumStrategy
from portfolios.portfolio_1.strategy import VolMomentum
from portfolios.portfolio_4.strategy import TrendRotateStrategy
from portfolios.portfolio_3.strategy import RegimeAdaptiveStrategy
from backtest.backtest_engine import BacktestEngine

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#added backtest setup parameters to main
def main(portfolio_classes= None,
         start_date="2024-11-01",
         end_date="2025-10-01",
         initial_capital=1000000.0,
         slippage=0):
    """
    Main entry point for the MQS Trading System backtests.
    """
    if portfolio_classes is None:
        portfolio_classes = [TrendRotateStrategy]

    try:
        dbconn = MQSDBConnector()
        
        backtest_engine = BacktestEngine(db_connector=dbconn, backtest_executor=None)

        backtest_engine.setup(
            portfolio_classes=portfolio_classes,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            slippage=slippage # 0.1 basis point
        )
        
        backtest_engine.run()

        return backtest_engine

    finally:
        logging.info("===== DONE =====")

if __name__ == '__main__':
    main()