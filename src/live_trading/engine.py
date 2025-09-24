# engines/run_engine.py

import time
import logging
import threading
from typing import List
from portfolios.portfolio_BASE.strategy import BasePortfolio

class RunEngine:
    """
    Manages and runs multiple trading portfolios concurrently for live trading.
    Includes a circuit breaker to stop portfolios that break repeatedly.
    """
    def __init__(self, db_connector, executor, debug=False, max_consecutive_failures=5):
        """
        Initializes the RunEngine.
        
        Args:
            db_connector: An instance of MQSDBConnector.
            executor: An instance of a trade executor (e.g., tradeExecutor).
            debug (bool): If True, portfolios will run once and exit.
            max_consecutive_failures (int): The number of consecutive errors after which 
                                            a portfolio's thread will be stopped.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_connector = db_connector
        self.executor = executor
        self.debug = debug
        self.portfolios: List[BasePortfolio] = []
        self.running = True
        
        # Circuit Breaker settings
        self.max_consecutive_failures = max_consecutive_failures
        self.failure_counts = {}

    def load_portfolios(self, portfolio_classes: List[type[BasePortfolio]]):
        """
        Initializes portfolio objects from the provided classes and loads them.
        """
        for portfolio_cls in portfolio_classes:
            try:
                portfolio_instance = portfolio_cls(
                    db_connector=self.db_connector,
                    executor=self.executor,
                    debug=self.debug
                )
                self.portfolios.append(portfolio_instance)
                # Initialize failure count for the circuit breaker
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
                
                data = portfolio.get_data(portfolio.data_feeds)
                portfolio.generate_signals_and_trade(data, current_time=None)

                # If the execution is successful, reset the failure counter.
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
                # --- Circuit Breaker Logic ---
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
                    # Break the loop to terminate the thread for this portfolio
                    break
                
                # Wait before retrying on failure
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
                # Check if any threads are still alive.
                # exit when all portfolios have stopped (e.g., all have tripped their breakers)
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