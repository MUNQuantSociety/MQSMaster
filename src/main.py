# main.py
# Entry point for the MQS Trading System.
# main.py is reserved for live trading purposes.


import logging
from common.database.MQSDBConnector import MQSDBConnector
from live_trading.executor import tradeExecutor
from portfolios.portfolio_1.strategy import VolMomentum
from portfolios.portfolio_2.strategy import MomentumStrategy
from portfolios.portfolio_3.strategy import RegimeAdaptiveStrategy
from portfolios.portfolio_dummy.strategy import CrossoverRmiStrategy
from live_trading.engine import RunEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    """
    Main entry point for the MQS Trading System.
    comment/uncomment portfolio classes in the portfolio_classes list to run different strategies.
    """
    portfolio_classes= [
        VolMomentum,
        MomentumStrategy,
#        RegimeAdaptiveStrategy,
#        CrossoverRmiStrategy,
    ]

#DO NOT CHANGE BELOW THIS LINE
#======================================================
    db_conn = None
    try:
        db_conn = MQSDBConnector()
        logging.info("Database connector initialized.")

        live_executor = tradeExecutor(db_conn)
        logging.info("Live executor initialized.")

        run_engine = RunEngine(db_connector=db_conn, executor=live_executor)
        logging.info("Run engine initialized.")

        run_engine.load_portfolios(portfolio_classes)
        logging.info("Run engine setup complete.")

        run_engine.run()

    except Exception as e:
        logging.critical(f"A critical error occurred in the main application loop: {e}", exc_info=True)
    finally:
        logging.info("MQS Trading System is shutting down.")

if __name__ == '__main__':
    main()