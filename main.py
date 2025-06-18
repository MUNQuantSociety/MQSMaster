from data_infra.database.MQSDBConnector import MQSDBConnector
from portfolios.portfolio_2.strategy import SimpleMeanReversion
from data_infra.tradingOps.realtime.live import tradeExecutor
from engines.run_engine import RunEngine
from engines.backtest_engine import BacktestEngine



def main():

    # Current Old Implementation
    dbconn = MQSDBConnector()
    tradeExecutor = tradeExecutor(dbconn)
    portfolio_2 = SimpleMeanReversion(db_connector=dbconn, executor=tradeExecutor)
    portfolio_2.run()

    print("Backtest executed successfully.")

    """
    New Implementation Example
    For real-time trading:

    Instead of initializing multiple portfolios directly like portfolio_1 = SAMPLE_PORTFOLIO_1(dbconn, tradeExecutor, debug=False), ...
    you can use the RunEngine to manage multiple portfolios which will initialize them and run them concurrently.

    runner = RunEngine(dbconn, tradeExecutor)
    runner.load_portfolios([portfolio_1, portfolio_2])
    runner.run()

    # Or backtest:
    backtest = BacktestEngine(dbconn, backtestExecutor)
    backtest.setup([portfolio_1], start_date="2025-01-01", end_date="2025-06-01")
    backtester.run()
    
    """

if __name__ == '__main__':
    main()