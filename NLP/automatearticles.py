import os
import sys
import time
import logging
from logging import Logger
from datetime import datetime, time as dtime
import pytz

# Add NLP folder to path so we can import fetch_articles
BASE_DIR = os.path.dirname(__file__)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import fetch_articles  # reuse the existing fetch logic

# --- CONFIG ---
PRIMARY_LOG_FILE = "/var/log/fetch_articles_trading_hours.log"
TIMEZONE = pytz.timezone("America/New_York")
MARKET_OPEN = dtime(9, 30)   # 9:30 AM ET
MARKET_CLOSE = dtime(23, 59)  # 4:00 PM ET
FETCH_INTERVAL_SECONDS = 18  # every 30 minutes

def setup_logging() -> Logger:
    """
    Configure logging with a safe fallback:
    - Try to write to /var/log/...
    - If not permitted, fall back to ~/fetch_articles_trading_hours.log
    - Always add a console (stderr) handler for visibility
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

    file_handler = None
    target_log_file = PRIMARY_LOG_FILE
    try:
        log_dir = os.path.dirname(target_log_file) or "."
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(target_log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info("Logging to %s", target_log_file)
    except Exception as e:
        # Fall back to a user-writable location
        fallback_log = os.path.expanduser("~/fetch_articles_trading_hours.log")
        try:
            file_handler = logging.FileHandler(fallback_log)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
            # Add a brief console message about the fallback
            console = logging.StreamHandler()
            console.setFormatter(fmt)
            logger.addHandler(console)
            logger.warning(
                "Fell back to user log file at %s (reason: %s: %s)",
                fallback_log, type(e).__name__, e
            )
        except Exception as e2:
            # If even fallback fails, at least log to console
            console = logging.StreamHandler()
            console.setFormatter(fmt)
            logger.addHandler(console)
            logger.error(
                "Failed to open log files; logging to console only (reason: %s: %s)",
                type(e2).__name__, e2
            )

    # Always include a console handler for immediate feedback during dev
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

    return logger

def main():
    logger = setup_logging()
    logger.info("======= Starting Trading-Hours Article Fetcher =======")

    ticker = os.getenv("TICKER", "AAPL")
    # Defaults: start from Jan 1, 2025 unless overridden; end = 'today' in ET
    start_date = os.getenv("START_DATE", "2025-01-01")
    end_date = os.getenv("END_DATE", datetime.now(TIMEZONE).strftime("%Y-%m-%d"))

    try:
        while True:
            now_et = datetime.now(TIMEZONE).time()
            if MARKET_OPEN <= now_et <= MARKET_CLOSE:
                logger.info("Fetching articles for %s between %s and %s", ticker, start_date, end_date)
                try:
                    fetch_articles.update_ticker_csv(ticker, start_date, end_date)
                    logger.info("Fetch complete. Sleeping %ss before next cycle.", FETCH_INTERVAL_SECONDS)
                except Exception as fetch_err:
                    logger.exception("Fetch failed: %s", fetch_err)
                time.sleep(FETCH_INTERVAL_SECONDS)
            else:
                # Outside market hours: check every 5 minutes
                logger.info("Market is closed (now ET: %s). Sleeping 300s.", now_et.strftime("%H:%M:%S"))
                time.sleep(300)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Trading-hours article fetcher stopped by user.")
    finally:
        logger.info("======= Trading-Hours Article Fetcher Stopped =======")

if __name__ == "__main__":
    main()
