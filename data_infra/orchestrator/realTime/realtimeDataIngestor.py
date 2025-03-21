import time
import logging
from datetime import datetime, time as dtime

# Your existing imports
from data_infra.marketData.fmpMarketData import FMPMarketData
from data_infra.database.MQSDBConnector import MQSDBConnector

# Import the helper methods
from data_infra.orchestrator.realTime.utils import load_tickers, collect_market_data

# Configure logging (adjust log file path as needed)
logging.basicConfig(
    filename='/var/log/market_data.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

MAX_WORKERS = 3

def main():
    """
    Main data-collection loop. 
    - Load tickers
    - Continuously fetch new data during market hours
    - Write to the database once 15 new entries have accumulated in memory
    """
    # Initialize FMP API client and DB connector
    fmp = FMPMarketData()
    db = MQSDBConnector()

    # Load all tickers
    tickers = load_tickers()
    logging.info(f"Loaded tickers: {tickers}")

    # Define market hours (change these to your local times as needed)
    market_open = dtime(9, 30)
    market_close = dtime(16, 0)

    # Keep track of the last timestamp seen for each ticker (to avoid duplicates)
    last_timestamps = {}

    # In-memory buffer for rows waiting to be inserted
    pending_entries = []
    batch_size = 15  # Only insert to DB once we hit this count

    while True:
        now = datetime.now()
        current_time = now.time()

        # Check if we're within market hours
        if market_open <= current_time <= market_close:
            # Fetch new data for all tickers
            new_rows = collect_market_data(fmp, tickers, last_timestamps, MAX_WORKERS)

            # Add new rows to our in-memory buffer
            pending_entries.extend(new_rows)

            # If we have enough pending entries, bulk-insert them into the DB
            if len(pending_entries) >= batch_size:
                _bulk_insert(db, pending_entries)
                pending_entries.clear()

            # Adjust the sleep time as desired to control frequency
            time.sleep(1)
        else:
            logging.info("Market closed. Exiting script.")
            break


def _bulk_insert(db, rows):
    """
    Performs a bulk insert into the database and logs the result.
    """
    try:
        result = db.inject_to_db("market_data", rows)
        if result["status"] == "success":
            logging.info(f"Inserted {len(rows)} rows into market_data.")
        else:
            logging.error(f"Error inserting data: {result['message']}")
    except Exception as e:
        logging.error(f"Error during bulk insert: {e}")


if __name__ == '__main__':
    main()
