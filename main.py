# main.py

import logging
from data_infra.database.MQSDBConnector import MQSDBConnector
from data_infra.tradingOps.realtime.live import tradeExecutor # For live trading
from Backtest.runner import BacktestRunner  # Assuming this is the backtest runner

# Import portfolio classes, not instances
from portfolios.portfolio_1.strategy import SAMPLE_PORTFOLIO
from portfolios.portfolio_2.strategy import SimpleMeanReversion

from engines.run_engine import RunEngine
from engines.backtest_engine import BacktestEngine  # Assuming this is the backtest engine

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
        SAMPLE_PORTFOLIO,
        SimpleMeanReversion
    ])

    
    # Start all loaded portfolios. This will block until Ctrl+C is pressed.
    run_engine.run()
    
    
    # Replace with actual backtest executor if available
    backtest_engine= BacktestEngine(db_connector=dbconn,backtest_executor=None)  # Replace None with actual backtest executor if available

    """backtest_engine.setup(
        portfolio_classes=[SAMPLE_PORTFOLIO,SimpleMeanReversion],
        start_date="2023-01-01",
        end_date="2023-12-31",
        initial_capital=1000000.0
    )
    # Run the backtests
    #backtest_engine.run()
    """
    # If you want to run backtests, uncomment the above lines and set up the backtest engine accordingly.

if __name__ == '__main__':
    main()