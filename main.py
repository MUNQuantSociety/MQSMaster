# TODO: Implement similar functionality.

from data_infra.tradingOps.main import tradeExecutor
from portfolios.portfolio_1.strategy import Portfolio_1


executor = tradeExecutor()
Portfolio_1.run(executor)