# engines/backtest_engine.py

import logging
from datetime import datetime
from typing import Optional, Union, List
from portfolios.portfolio_BASE.strategy import BasePortfolio
from backtest.runner import BacktestRunner  # Assuming this is the correct import path for the BacktestRunner



class BacktestEngine:
    """
    Manages and runs backtests for multiple strategies.
    """
    def __init__(self, db_connector, backtest_executor):
        """
        Initializes the BacktestEngine.
        
        Args:
            db_connector: An instance of MQSDBConnector.
            backtest_executor: An executor designed for backtesting (simulates trades).
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_connector = db_connector
        self.backtest_executor = backtest_executor
        self.backtest_configs = []

    def setup(self, 
              portfolio_classes: List[type[BasePortfolio]], 
              start_date: Union[str, datetime], 
              end_date: Union[str, datetime], 
              initial_capital: float = 100_000.0):
        """
        Prepares portfolios for a backtest run.
        """
        for portfolio_cls in portfolio_classes:
            try:
                # In a backtest, the portfolio still needs to be instantiated.
                # It will be passed to a BacktestRunner later.
                portfolio_instance = portfolio_cls(
                    db_connector=self.db_connector,
                    executor=self.backtest_executor
                )
                config = {
                    "portfolio_instance": portfolio_instance,
                    "start_date": start_date,
                    "end_date": end_date,
                    "initial_capital": initial_capital
                }
                self.backtest_configs.append(config)
                self.logger.info(f"Successfully set up backtest for {portfolio_cls.__name__}.")
            except Exception as e:
                self.logger.exception(f"Failed to set up backtest for {portfolio_cls.__name__}: {e}")

    def run(self):
        """
        Executes the configured backtests.
        """
        if not self.backtest_configs:
            self.logger.warning("No backtests set up. Call setup() first.")
            return
        for config in self.backtest_configs:
            portfolio = config['portfolio_instance']
            self.logger.info(f"Running backtest for portfolio {portfolio.portfolio_id} from {config['start_date']} to {config['end_date']}.")
            try:
                # The BacktestRunner takes the instantiated portfolio and runs the simulation
                runner = BacktestRunner(
                    portfolio=portfolio,
                    start_date=config['start_date'],
                    end_date=config['end_date'],
                    initial_capital=config['initial_capital']
                )
                
                runner.run()
                self.logger.info(f"Backtest for portfolio {portfolio.portfolio_id} completed.")
            except Exception as e:
                self.logger.exception(f"Backtest failed for portfolio {portfolio.portfolio_id}: {e}")
        
        self.logger.info("All backtests completed.")