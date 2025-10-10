# src/backtest/backtest_engine.py

import inspect
import json
import logging
import os
import pandas as pd
from typing import List

from src.common.database.MQSDBConnector import MQSDBConnector
from src.portfolios.portfolio_BASE.strategy import BasePortfolio

from .runner import BacktestRunner


class BacktestEngine:
    """
    The BacktestEngine orchestrates backtesting runs.
    It is updated to load portfolio configurations dynamically.
    """

    def __init__(self, db_connector: 'MQSDBConnector', backtest_executor=None):
        self.db_connector = db_connector
        self.backtest_executor = backtest_executor
        self.logger = logging.getLogger(self.__class__.__name__)
        self.portfolio_classes: List[type[BasePortfolio]] = []
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
                # --- Dynamically load the config for the portfolio ---
                # Get the file path of the portfolio's strategy class
                class_file_path = inspect.getfile(portfolio_class)
                portfolio_dir = os.path.dirname(class_file_path)
                config_path = os.path.join(portfolio_dir, 'config.json')

                if not os.path.exists(config_path):
                    self.logger.error(f"Configuration file not found for {portfolio_class.__name__} at {config_path}")
                    continue

                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # --- Instantiate with the loaded config_dict ---
                portfolio_instance = portfolio_class(
                    db_connector=self.db_connector,
                    executor=None,  # The runner will set the executor later
                    config_dict=config_data,
                    backtest_start_date=pd.to_datetime(self.start_date)
                )
                
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