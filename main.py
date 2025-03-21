from data_infra.tradingOps.main import tradeExecutor
from data_infra.database.MQSDBConnector import MQSDBConnector
from portfolios.portfolio_1.strategy import SimpleMomentum
# import more portfolios as needed

def main():
    db = MQSDBConnector()
    executor = tradeExecutor()

    portfolio_1 = SimpleMomentum(db_connector=db, executor=executor)
    portfolio_1.run()

if __name__ == '__main__':
    main()
