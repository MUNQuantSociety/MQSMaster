import os
import sys
import time
import logging
from datetime import datetime, time as dtime
import pytz

# Add NLP folder to path so we can import fetch_articles
BASE_DIR = os.path.dirname(__file__)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import fetch_articles # reuse the existing fetch logic

# --- CONFIG ---
LOG_FILE = "/var/log/fetch_articles_trading_hours.log"
TIMEZONE = pytz.timezone("America/New_York")
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
FETCH_INTERVAL_SECONDS = 1800 # every 30 minutes

def setup_logging():
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s"
    )

def main():
    setup_logging()
    logging.info("======= Starting Trading-Hours Article Fetcher =======")

    ticker = os.getenv("TICKER", "AAPL")
    start_date = os.getenv("START_DATE", "2023-01-01")
    end_date = os.getenv("END_DATE", datetime.now().strftime("%Y-%m-%d"))

    try:
        while True:
            now = datetime.now(TIMEZONE).time()
            if MARKET_OPEN <= now <= MARKET_CLOSE:
                logging.info(f"Fetching articles for {ticker} between {start_date} and {end_date}")
                fetch_articles.update_ticker_csv(ticker, start_date, end_date)
                logging.info(f"Sleeping {FETCH_INTERVAL_SECONDS}s before next cycle.")
                time.sleep(FETCH_INTERVAL_SECONDS)
            else:
                logging.info("Market is closed. Sleeping until next check.")
                time.sleep(300) # check every 5 minutes outside market hours
    except (KeyboardInterrupt, SystemExit):
        logging.info("Trading-hours article fetcher stopped by user.")
    finally:
        logging.info("======= Trading-Hours Article Fetcher Stopped =======")

if __name__ == "__main__":
    main()