from data_infra.database.MQSDBConnector import MQSDBConnector
from portfolios.portfolio_2.strategy import SimpleMeanReversion


def main():
    db = MQSDBConnector()
    portfolio_2 = SimpleMeanReversion(db_connector=db, executor=None)
    weights_2 = {
        "AAPL": 0.4,
        "TSLA": 0.3,
        "NVDA": 0.3
    }
    portfolio_2.backtest(weights_2)


    print("Backtest executed successfully.")

if __name__ == '__main__':
    main()