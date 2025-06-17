# MQSMaster/engines/backtest_engine.py
from datetime import datetime
from typing import Optional, Union
from data_infra.tradingOps.backtest.runner import BacktestRunner
from portfolios.portfolio_BASE.strategy import BasePortfolio

class BacktestEngine:
    """Replays historical data via BacktestRunner."""
    def __init__(self,
                 strategy: BasePortfolio,
                 start_date: Optional[Union[str, datetime]] = None,
                 end_date:   Optional[Union[str, datetime]] = None,
                 initial_capital_per_ticker: float = 100_000.0):
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital_per_ticker

    def run(self):
        runner = BacktestRunner(
            portfolio=self.strategy,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital_per_ticker=self.initial_capital
        )
        runner.run()
