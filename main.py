from data_infra.database.MQSDBConnector import MQSDBConnector
from portfolios.portfolio_2.strategy import SimpleMeanReversion
from engines.run_engine import RunEngine
from engines.backtest_engine import BacktestEngine



def main():
    db = MQSDBConnector()
    portfolio_2 = SimpleMeanReversion(db_connector=db, executor=None)
    portfolio_2.backtest()

    print("Backtest executed successfully.")

    """
    strategy = SAMPLE_PORTFOLIO(db, executor, debug=False)

    # Live run:
    runner = RunEngine(strategy)
    runner.start()

    # Or backtest:
    backtester = BacktestEngine(strategy, start_date="2025-01-01", end_date="2025-06-01")
    backtester.run()
    
    """

if __name__ == '__main__':
    main()