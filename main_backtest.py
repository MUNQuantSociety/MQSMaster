# This is where backtests are setup.
# Simply add you portfolio class to the list of portfolio classes in the `setup` method.

# High level overview of how to set up a backtest:

# 1. Load your portfolio through imports such as `from portfolios.portfolio_1.strategy import SimpleMomentum`
# 2. Set up the backtest engine with the database connector and backtest executor.
# 3. Call the `setup` method with the portfolio class, start date, end date, and initial capital.
# 4. Call the `run` method to execute the backtest.

import logging
import cProfile
import pstats

from data_infra.database.MQSDBConnector import MQSDBConnector

# Import portfolio classes, not instances
from portfolios.portfolio_1.strategy import SimpleMomentum
from portfolios.portfolio_2.strategy import MomentumThresholdStrategy

from engines.backtest_engine import BacktestEngine

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    """
    Main entry point for the MQS Trading System.
    """
    # 1. Set up the profiler to analyse the performance of the backtest
    profiler = cProfile.Profile()
    profiler.enable()

    # --- Original main logic starts here ---
    try:
        dbconn = MQSDBConnector()
        
        # In a backtest, backtest_executor is none.
        backtest_engine = BacktestEngine(db_connector=dbconn, backtest_executor=None)

        backtest_engine.setup(
            portfolio_classes=[MomentumThresholdStrategy],
            start_date="2023-01-01",
            end_date="2024-01-01",
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
        stats.print_stats(6)


if __name__ == '__main__':
    main()