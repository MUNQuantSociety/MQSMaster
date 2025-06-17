# MQSMaster/engines/run_engine.py
import time
import logging
from datetime import datetime
from typing import Optional
from portfolios.portfolio_BASE.strategy import BasePortfolio

class RunEngine:
    """ Continuously polls live data and feeds it to the strategy."""
    def __init__(self, strategy: BasePortfolio):
        self.strategy = strategy
        self.logger = logging.getLogger(self.__class__.__name__)
        self.poll_interval = strategy.poll_interval
        self.data_feeds = strategy.data_feeds

    def start(self):
        self.logger.info("Starting live run loop.")
        while True:
            try:
                start_ts = time.time()
                data = self.strategy.get_data(self.data_feeds)
                if data:
                    # live mode: no current_time
                    self.strategy.generate_signals_and_trade(data)
                else:
                    self.logger.info("No new data.")
                if self.strategy.debug:
                    self.logger.info("Debug mode: exit after one iteration.")
                    break
                elapsed = time.time() - start_ts
                sleep = max(0, self.poll_interval - elapsed)
                time.sleep(sleep)
            except KeyboardInterrupt:
                self.logger.warning("Keyboard interrupt, stopping run loop.")
                break
            except Exception as e:
                self.logger.exception(f"Error in run loop: {e}")
        self.logger.info("Run loop exited.")
