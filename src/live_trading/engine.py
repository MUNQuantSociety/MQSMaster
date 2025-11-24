# src/live_trading/engine.py

import inspect
import json
import logging
import os
import threading
import time
from typing import List
import time
from datetime import datetime, timezone

from src.portfolios.portfolio_BASE.strategy import BasePortfolio


class RunEngine:
    """
    Manages and runs multiple trading portfolios concurrently for live trading.
    Updated to dynamically load configurations for each portfolio.
    """
    def __init__(self, db_connector, executor, debug=False, max_consecutive_failures=5):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_connector = db_connector
        self.executor = executor
        self.debug = debug
        self.portfolios: List[BasePortfolio] = []
        self.running = True
        
        self.max_consecutive_failures = max_consecutive_failures
        self.failure_counts = {}

    def load_portfolios(self, portfolio_classes: List[type[BasePortfolio]]):
        """
        Initializes portfolio objects from the provided classes and loads them.
        """
        for portfolio_cls in portfolio_classes:
            try:
                # --- NEW: Dynamically load the config for the portfolio ---
                class_file_path = inspect.getfile(portfolio_cls)
                portfolio_dir = os.path.dirname(class_file_path)
                config_path = os.path.join(portfolio_dir, 'config.json')

                if not os.path.exists(config_path):
                    self.logger.error(f"Configuration file not found for {portfolio_cls.__name__} at {config_path}")
                    continue

                with open(config_path, 'r') as f:
                    config_data = json.load(f)

                # --- UPDATED: Instantiate with the loaded config_dict ---
                portfolio_instance = portfolio_cls(
                    db_connector=self.db_connector,
                    executor=self.executor,
                    debug=self.debug,
                    config_dict=config_data
                )
                self.portfolios.append(portfolio_instance)
                self.failure_counts[portfolio_instance.portfolio_id] = 0
                self.logger.info(f"Successfully loaded portfolio: {portfolio_cls.__name__}")

            except Exception as e:
                self.logger.exception(f"Failed to load portfolio {portfolio_cls.__name__}: {e}")

    def _run_portfolio(self, portfolio: BasePortfolio):
        """
        The target function for each portfolio's thread. Contains the polling loop
        and circuit breaker logic.
        """
        portfolio_id = portfolio.portfolio_id
        self.logger.info(f"Starting run loop for portfolio {portfolio_id} ({portfolio.__class__.__name__}).")
        
        while self.running:
            try:
                start_time = time.time()
                
                # --- FIX START ---
                # Capture the current UTC time to pass to the strategy
                now_utc = datetime.now(timezone.utc)
                
                data = portfolio.get_data(portfolio.data_feeds)
                
                # Pass 'now_utc' instead of 'None'
                portfolio.generate_signals_and_trade(data, current_time=now_utc)
                # --- FIX END ---

                if self.failure_counts[portfolio_id] > 0:
                    self.logger.info(f"Portfolio {portfolio_id} recovered after {self.failure_counts[portfolio_id]} failures.")
                    self.failure_counts[portfolio_id] = 0

                if portfolio.debug:
                    self.logger.info(f"Debug mode: stopping portfolio {portfolio_id} after one run.")
                    break

                elapsed_time = time.time() - start_time
                sleep_time = max(0, portfolio.poll_interval - elapsed_time)
                time.sleep(sleep_time)

            except Exception as e:
                self.failure_counts[portfolio_id] += 1
                self.logger.exception(
                    f"Exception in portfolio {portfolio_id} loop. Consecutive failure "
                    f"count: {self.failure_counts[portfolio_id]}/{self.max_consecutive_failures}. Error: {e}"
                )
                
                if self.failure_counts[portfolio_id] >= self.max_consecutive_failures:
                    self.logger.critical(
                        f"CIRCUIT BREAKER TRIPPED: Portfolio {portfolio_id} has failed "
                        f"{self.max_consecutive_failures} consecutive times. Stopping this portfolio thread."
                    )
                    break
                
                time.sleep(portfolio.poll_interval)

        self.logger.info(f"Stopped run loop for portfolio {portfolio_id}.")

    def run(self):
        """
        Starts the trading engine, running all loaded portfolios in separate threads.
        """
        if not self.portfolios:
            self.logger.warning("No portfolios loaded. Exiting.")
            return

        self.logger.info(f"Starting RunEngine with {len(self.portfolios)} portfolios.")
        threads = []
        for portfolio in self.portfolios:
            thread = threading.Thread(target=self._run_portfolio, args=(portfolio,))
            threads.append(thread)
            thread.start()

        try:
            while self.running:
                if not any(t.is_alive() for t in threads):
                    self.logger.warning("All portfolio threads have stopped. Shutting down RunEngine.")
                    self.running = False
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.warning("Keyboard interrupt received. Shutting down all portfolios.")
            self.running = False

        for thread in threads:
            thread.join()

        self.logger.info("All portfolio threads have been joined. RunEngine shutdown complete.")