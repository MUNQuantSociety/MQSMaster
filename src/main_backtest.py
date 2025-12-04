#MQS Trading System - Main Backtest Module
'''
How to run a backtest:
1. Load your portfolio through imports such as:
    `from portfolios.portfolio_n.strategy import StrategyClass`.
2. Setup main args with start date, end date, initial capital, and slippage.
3. Add class to classes list comment out unused strategies for faster testing.
4. Run python -m src.main_backtest
'''

import logging
from common.database.MQSDBConnector import MQSDBConnector
from portfolios.portfolio_1.strategy import VolMomentum
from portfolios.portfolio_2.strategy import MomentumStrategy
from portfolios.portfolio_3.strategy import RegimeAdaptiveStrategy
from portfolios.portfolio_4.strategy import TrendRotateStrategy
from portfolios.portfolio_dummy.strategy import CrossoverRmiStrategy
from backtest.backtest_engine import BacktestEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main(portfolio_classes= None,
         start_date="2025-01-01",
         end_date="2025-01-31",
         initial_capital=1000000.0,
         slippage=0):
    """
    Main entry point for the MQS Trading System backtests.
    comment/uncomment the classes in the portfolio_classes list to run different strategies.
    """
    portfolio_classes= [
        VolMomentum,
        MomentumStrategy,
        RegimeAdaptiveStrategy,
        TrendRotateStrategy,
        CrossoverRmiStrategy,

    ]

#DO NOT CHANGE BELOW THIS LINE
#======================================================
    try:
        dbconn = MQSDBConnector()
        
        backtest_engine = BacktestEngine(
            db_connector=dbconn,
            backtest_executor=None)

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