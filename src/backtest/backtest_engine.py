# src/backtest/backtest_engine.py

from typing import List
import logging
from .runner import BacktestRunner
from portfolios.portfolio_BASE.strategy import BasePortfolio

from common.database.MQSDBConnector import MQSDBConnector


class BacktestEngine:
    """
    The BacktestEngine is responsible for orchestrating backtesting runs for
    one or more portfolios.
    """

    def __init__(self, db_connector: 'MQSDBConnector', backtest_executor=None):
        self.db_connector = db_connector
        self.backtest_executor = backtest_executor
        self.logger = logging.getLogger(self.__class__.__name__)
        self.portfolio_classes: List['BasePortfolio'] = []
        self.start_date: str = ""
        self.end_date: str = ""
        self.initial_capital: float = 0.0
        self.slippage: float = 0.0

    def setup(self,
              portfolio_classes: List[type],
              start_date: str,
              end_date: str,
              initial_capital: float,
              slippage: float = 0.0):
        """
        Configures the backtest with the necessary parameters.
        """
        self.portfolio_classes = portfolio_classes
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.slippage = slippage
        self.logger.info("Backtest engine setup complete.")

    def run(self):
        """
        Initializes and runs the backtest for each portfolio.
        """
        if not self.portfolio_classes:
            self.logger.error("No portfolio classes provided to run backtests.")
            return

        for portfolio_class in self.portfolio_classes:
            try:
                # Instantiate the portfolio, which now has a db connection
                # The executor will be set by the BacktestRunner
                portfolio_instance = portfolio_class(db_connector=self.db_connector, executor=None)
                
                self.logger.info(f"--- Running backtest for portfolio: {portfolio_instance.portfolio_id} ---")

                runner = BacktestRunner(
                    portfolio=portfolio_instance,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    initial_capital=self.initial_capital,
                    slippage=self.slippage
                )
                runner.run()
                
                self.logger.info(f"--- Backtest for portfolio: {portfolio_instance.portfolio_id} finished ---")

            except Exception as e:
                self.logger.exception(f"Error running backtest for {portfolio_class.__name__}: {e}", exc_info=True)