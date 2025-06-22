# This is the complete main.py file with profiling included.

import logging
import cProfile
import pstats

from data_infra.database.MQSDBConnector import MQSDBConnector

# Import portfolio classes, not instances
from portfolios.portfolio_1.strategy import SAMPLE_PORTFOLIO
from portfolios.portfolio_2.strategy import SimpleMeanReversion

from engines.backtest_engine import BacktestEngine

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    """
    Main entry point for the MQS Trading System.
    """
    
    # 1. Set up the profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # --- Original main logic starts here ---
    try:
        dbconn = MQSDBConnector()
        
        # In a backtest, the executor is handled by the BacktestRunner, so passing None here is correct.
        backtest_engine = BacktestEngine(db_connector=dbconn, backtest_executor=None)

        backtest_engine.setup(
            portfolio_classes=[SAMPLE_PORTFOLIO],
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=1000000.0
        )
        
        # Run the backtests (this is the part we are profiling)
        backtest_engine.run()

    finally:
        # 2. Stop the profiler and print the results, even if errors occur
        profiler.disable()
        logging.info("===== PROFILING RESULTS =====")
        # Sort stats by cumulative time and print the top 20 results
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(20)


if __name__ == '__main__':
    main()