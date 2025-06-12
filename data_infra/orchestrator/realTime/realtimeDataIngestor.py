import sys
import os
import time
import logging
import pandas as pd
import pytz
from datetime import datetime, time as dtime

# Ensure we can import data_infra.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from data_infra.marketData.fmpMarketData import FMPMarketData
from data_infra.database.MQSDBConnector import MQSDBConnector
from data_infra.orchestrator.realTime.utils import load_tickers

# --- Configuration ---
LOG_FILE = '/var/log/market_data_ingestor.log'
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
FETCH_INTERVAL_SECONDS = 60
EXCHANGE_TO_MONITOR = "NASDAQ"
DB_TABLE_NAME = "market_data"
TIMEZONE = pytz.timezone("America/New_York")

def setup_logging():
    """Configures the logging for the script."""
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s'
    )

# --- WARNING ---
# The function below is NOT robust to script crashes. It attempts to re-initialize
# state by reading the `volume` column, but that column stores the *interval* volume,
# not the *cumulative* volume from the API. For this to work correctly after a crash,
# the database schema would need a separate 'cumulative_volume' column to read from.
# As per your request, this change is deferred.
def initialize_volume_state(db: MQSDBConnector, tickers: set) -> dict:
    """
    Initializes the volume state from the DB for the current day to ensure
    continuity if the script restarts.
    """
    logging.info("Initializing volume state from database for today...")
    today_date = datetime.now(TIMEZONE).date()
    
    sql = f"""
    WITH LatestEntries AS (
        SELECT
            ticker,
            volume,
            RANK() OVER(PARTITION BY ticker ORDER BY timestamp DESC) as rnk
        FROM {DB_TABLE_NAME}
        WHERE date = '{today_date}' AND ticker = ANY(ARRAY{list(tickers)})
    )
    SELECT ticker, volume FROM LatestEntries WHERE rnk = 1;
    """
    
    result = db.read_db(sql=sql)
    
    if result['status'] == 'success' and result['data']:
        state = {row['ticker']: row['volume'] for row in result['data']}
        logging.info(f"Successfully initialized state for {len(state)} tickers from DB.")
        return state
        
    logging.warning("Could not initialize state from DB. Starting with fresh state.")
    return {}

def process_market_data(api_data: list, tickers_to_track: set, last_known_volumes: dict) -> list:
    """
    Transforms raw API data into a format ready for database injection using
    vectorized pandas operations.
    """
    if not api_data:
        logging.info("API returned no data to process.")
        return []

    # 1. Load data and filter for tracked tickers
    df = pd.DataFrame(api_data)
    df = df[df['symbol'].isin(tickers_to_track)].copy()

    if df.empty:
        logging.info("No relevant tickers found in the latest API batch.")
        return []
    
    logging.info(f"Processing {len(df)} records for tracked tickers.")

    # 2. Vectorized Interval Volume Calculation
    df_last_volumes = pd.DataFrame(last_known_volumes.items(), columns=['symbol', 'last_volume'])
    df = pd.merge(df, df_last_volumes, on='symbol', how='left')
    # Avoid inplace=True by reassigning the column.
    df['last_volume'] = df['last_volume'].fillna(0) 
    df['interval_volume'] = df['volume'] - df['last_volume']
    df.loc[df['interval_volume'] < 0, 'interval_volume'] = df['volume']

    # 3. Update the state for the next cycle (Optimized)
    # This vectorized approach is much faster than iterating.
    # It uses the cumulative volume from the API before it's overwritten.
    new_state = pd.Series(df.volume.values, index=df.symbol).to_dict()
    last_known_volumes.update(new_state)

    # 4. Format and align with DB schema
    df.rename(columns={
        'symbol': 'ticker',
        'price': 'close_price',
        'exchange': 'exchange'
    }, inplace=True)
    df['volume'] = df['interval_volume'] # Overwrite cumulative with interval volume

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(TIMEZONE).dt.round('min')
    df['date'] = df['timestamp'].dt.date
    
    df['open_price'] = None
    df['high_price'] = None
    df['low_price'] = None
    
    db_columns = [
        "ticker", "timestamp", "date", "exchange",
        "open_price", "high_price", "low_price", "close_price", "volume"
    ]
    
    df.dropna(subset=['timestamp', 'ticker'], inplace=True)
    return df[db_columns].to_dict('records')

def run_ingestion_cycle(fmp: FMPMarketData, db: MQSDBConnector, tickers_to_track: set, volume_state: dict):
    """

    Executes a single fetch-process-inject cycle.
    """
    logging.info("--- New Ingestion Cycle ---")
    api_data = fmp.get_realtime_data(EXCHANGE_TO_MONITOR)

    if not api_data:
        logging.warning("Failed to fetch data from API. Skipping cycle.")
        return

    rows_to_insert = process_market_data(api_data, tickers_to_track, volume_state)
    
    if not rows_to_insert:
        logging.info("No new data to insert after processing.")
        return

    result = db.bulk_inject_to_db(DB_TABLE_NAME, rows_to_insert)
    if result["status"] == "success":
        logging.info(f"Database injection result: {result['message']}")
    else:
        logging.error(f"Database injection failed: {result['message']}")

def main():
    """
    Main data-collection loop. Fetches bulk exchange data, processes it,
    and performs an efficient bulk insert into the database.
    """
    setup_logging()
    logging.info("======= Starting Real-Time Data Ingestor =======")
    
    db = MQSDBConnector()
    fmp = FMPMarketData()
    
    try:
        tickers_to_track = set(load_tickers())
        if not tickers_to_track:
            logging.error("No tickers loaded from portfolio configs. Exiting.")
            return
        
        volume_state = initialize_volume_state(db, tickers_to_track)

        while True:
            # Use timezone-aware datetime for robust market hours checking
            now_eastern = datetime.now(TIMEZONE)
            if not (MARKET_OPEN <= now_eastern.time() <= MARKET_CLOSE):
                logging.info("Market is closed. Stopping script.")
                break

            start_time = time.time()
            run_ingestion_cycle(fmp, db, tickers_to_track, volume_state)
            
            elapsed_time = time.time() - start_time
            sleep_time = max(0, FETCH_INTERVAL_SECONDS - elapsed_time)
            logging.info(f"Cycle finished in {elapsed_time:.2f}s. Sleeping for {sleep_time:.2f}s.")
            time.sleep(sleep_time)

    except (KeyboardInterrupt, SystemExit):
        logging.info("Script manually interrupted.")
    except Exception as e:
        logging.critical(f"A critical error occurred in the main loop: {e}", exc_info=True)
    finally:
        db.close_all_connections()
        logging.info("======= Real-Time Data Ingestor Stopped =======")

if __name__ == '__main__':
    main()