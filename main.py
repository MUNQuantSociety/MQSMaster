from data_infra.database.MQSDBConnector import MQSDBConnector
from portfolios.portfolio_2.strategy import SimpleMeanReversion


def main():
    db = MQSDBConnector()
    portfolio_1 = SimpleMeanReversion(db_connector=db, executor=None)

    portfolio_1.backtest()


    print("Backtest executed successfully.")

if __name__ == '__main__':
    main()