import sys
import os
import glob
import csv
import psycopg2
from psycopg2.extras import execute_values
import concurrent.futures
import logging

# Ensure that the parent directory is in the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from common.database.MQSDBConnector import MQSDBConnector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# SQL statement for bulk insert into market_data
INSERT_SQL = """
    INSERT INTO market_data (
        ticker, 
        timestamp, 
        date, 
        exchange,
        open_price,
        high_price, 
        low_price, 
        close_price, 
        volume
    )
    VALUES %s
"""

def ticker_exists(db, ticker):
    """
    Checks if any row for the given ticker exists in market_data.
    """
    query = "SELECT 1 FROM market_data WHERE ticker = %s LIMIT 1"
    result = db.execute_query(query, (ticker,), fetch=True)
    if result.get('status') == 'success' and result.get('data'):
        return True
    return False

def process_file(csv_file, db):
    """
    Processes a single CSV file:
    - Extracts the ticker from the file name.
    - Checks if the ticker already exists in the DB.
    - If not, reads the CSV file and performs batch inserts,
      printing a % complete progress update for each batch.
    """
    try:
        filename = os.path.basename(csv_file)
        # Assuming filename format "2y_mkt_data_{ticker}.csv"
        if not filename.startswith("2y_mkt_data_") or not filename.endswith(".csv"):
            logging.warning(f"Skipping file with unexpected name format: {csv_file}")
            return

        ticker = filename[len("2y_mkt_data_"):-len(".csv")]
        logging.info(f"Processing file: {csv_file} for ticker: {ticker}")

        # Check if this ticker already exists in the DB
        if ticker_exists(db, ticker):
            logging.info(f"Ticker {ticker} already exists in the database. Skipping file: {csv_file}")
            return

        # Count total rows (excluding header) to calculate progress.
        with open(csv_file, 'r', newline='') as f:
            total_rows = sum(1 for _ in f) - 1
        if total_rows <= 0:
            logging.info(f"No valid rows in {csv_file}. Skipping insert.")
            return

        batch_size = 5000
        processed_rows = 0
        conn = db.get_connection()
        try:
            with open(csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                batch = []
                for row in reader:
                    try:
                        open_price = float(row['open'])
                        high_price = float(row['high'])
                        low_price = float(row['low'])
                        close_price = float(row['close'])
                        volume = int(float(row['volume']))
                    except (ValueError, KeyError) as e:
                        logging.warning(f"Skipping row due to parse error: {row} error: {e}")
                        continue

                    batch.append((
                        row['ticker'],
                        row['datetime'],  # e.g. '2023-03-07 15:59:00'
                        row['date'],      # e.g. '2023-03-07'
                        'nasdaq',
                        open_price,
                        high_price,
                        low_price,
                        close_price,
                        volume
                    ))
                    processed_rows += 1

                    # If batch is full, insert and log progress.
                    if len(batch) >= batch_size:
                        try:
                            with conn.cursor() as cursor:
                                execute_values(cursor, INSERT_SQL, batch)
                            conn.commit()
                        except psycopg2.Error as e:
                            try:
                                conn.rollback()
                            except psycopg2.InterfaceError:
                                logging.error("Connection already closed during rollback.")
                            logging.error(f"Error during bulk insert for {csv_file}: {e}")
                            return
                        batch = []
                        progress = (processed_rows / total_rows) * 100
                        logging.info(f"{ticker}: {progress:.2f}% complete")
                # Insert any remaining rows.
                if batch:
                    try:
                        with conn.cursor() as cursor:
                            execute_values(cursor, INSERT_SQL, batch)
                        conn.commit()
                    except psycopg2.Error as e:
                        try:
                            conn.rollback()
                        except psycopg2.InterfaceError:
                            logging.error("Connection already closed during rollback.")
                        logging.error(f"Error during final bulk insert for {csv_file}: {e}")
                        return
                    progress = (processed_rows / total_rows) * 100
                    logging.info(f"{ticker}: {progress:.2f}% complete")
            logging.info(f"Finished processing file: {csv_file} for ticker: {ticker}")
        finally:
            db.release_connection(conn)
    except Exception as general_exception:
        logging.error(f"Unexpected error processing {csv_file}: {general_exception}")

def load_csv_files_to_db(directory_path, max_workers=5):
    """
    Loads all CSV files matching the pattern '2y_mkt_data*.csv' from directory_path.
    Files are processed in parallel using multiple threads.
    """
    db = MQSDBConnector()  # Instantiate your DB connector
    pattern = os.path.join(directory_path, "2y_mkt_data*.csv")
    csv_files = glob.glob(pattern)
    logging.info(f"Found {len(csv_files)} CSV files: {csv_files}")

    # Use a ThreadPoolExecutor to process files concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, csv_file, db) for csv_file in csv_files]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logging.error(f"File processing generated an exception: {exc}")

    db.close_all_connections()
    logging.info("All CSV files processed successfully!")

if __name__ == "__main__":
    # Provide the folder that contains your CSV files.
    csv_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/backfill_cache/"))
    load_csv_files_to_db(csv_folder_path, max_workers=5)
    script_path = os.path.abspath(__file__)
    logging.info(f"Script path: {script_path}")
    logging.info(f"CSV folder path: {csv_folder_path}")
