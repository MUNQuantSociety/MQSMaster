"""
concurrent_backfill.py
----------------------
A multi-threaded approach for backfilling multiple tickers in parallel
using the existing 'backfill_data' function from 'backfill.py'.
Each ticker is processed in its own thread to reduce total runtime.

Results:
  - Data is injected directly into the 'market_data' table in the database using MQSDBConnector.
"""

import sys
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from psycopg2.extras import execute_values
import json

# Ensure we can import backfill.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from data_infra.orchestrator.backfill.backfill import backfill_data
from data_infra.database.MQSDBConnector import MQSDBConnector

# Number of threads to use. NEEDS TO BE LESS THAN MQSDBCONNECTOR MAX CONN VALUE!
MAX_WORKERS = 8


def parse_date_arg(date_str):
    """Parses date string in DDMMYY format and returns a datetime.date object."""
    try:
        return datetime.strptime(date_str, "%d%m%y").date()
    except ValueError:
        print(f"❌ Invalid date format: {date_str}. Expected format: DDMMYY (e.g., 040325 for March 4, 2025).")
        sys.exit(1)


def backfill_single_ticker(ticker, start_date, end_date, interval, exchange, db_connector):
    """
    Calls backfill_data(...) for a single ticker and injects data into the DB.
    This function now receives a shared MQSDBConnector instance.
    """
    conn = None
    try:
        # Fetch the data in-memory (output_filename=None => returns DataFrame)
        df = backfill_data(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            exchange=exchange,
            output_filename=None
        )

        # Check if data was returned and not empty
        if df is None or df.empty:
            print(f"[{ticker}] No data returned from backfill.")
            return

        # Get a DB connection from the shared pool
        conn = db_connector.get_connection()
        if not conn:
            print(f"[{ticker}] Could not get DB connection from pool.")
            return

        insert_data = []
        for _, row in df.iterrows():
            try:
                insert_data.append((
                    row['ticker'],
                    row['datetime'],  # timestamp
                    row['date'],
                    exchange.lower() if exchange else 'nasdaq',
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(float(row['volume'])),
                ))
            except Exception as parse_ex:
                print(f"[{ticker}] Skipping row due to parsing error: {parse_ex}")
                continue

        # Bulk insert
        insert_sql = """
            INSERT INTO market_data (
                ticker, timestamp, date, exchange,
                open_price, high_price, low_price, close_price, volume
            )
            VALUES %s
        """

        if insert_data:
            with conn.cursor() as cursor:
                execute_values(cursor, insert_sql, insert_data)
            conn.commit()
            print(f"[{ticker}] Inserted {len(insert_data)} rows into DB.")
        else:
            print(f"[{ticker}] No valid rows to insert.")

    except Exception as e:
        print(f"[{ticker}] Error during backfill or insert: {e}")
    finally:
        # Release the connection back to the pool
        if conn:
            db_connector.release_connection(conn)


def concurrent_backfill(tickers, start_date, end_date, interval, exchange=None):
    """
    Spawns multiple threads, each calling 'backfill_data' for a single ticker.
    Injects each ticker's results directly into the DB using a shared connector.
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    print(f"[ConcurrentBackfill] Starting concurrency for {len(tickers)} tickers.")
    print(f"  Date range: {start_date} to {end_date}, interval={interval} min, exchange={exchange}")
    print(f"  Using up to {MAX_WORKERS} threads.")
    
    # Create ONE shared database connector instance
    db_connector = MQSDBConnector()
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Pass the single db_connector instance to each worker
            futures = [
                executor.submit(backfill_single_ticker, ticker, start_date, end_date, interval, exchange, db_connector)
                for ticker in tickers
            ]
            for fut in futures:
                try:
                    fut.result()
                except Exception as ex:
                    print(f"[ConcurrentBackfill:ERROR] A worker failed with: {ex}")
    finally:
        # Ensure all pool connections are closed at the end
        print("[ConcurrentBackfill] All threads completed. Closing connection pool.")
        db_connector.close_all_connections()


if __name__ == "__main__":
    # 1. Load tickers from tickers.json
    script_dir = os.path.dirname(__file__)
    # Corrected typo from .jsonx to .json for robustness
    ticker_file_path = os.path.join(script_dir, '..', 'tickers.json')

    try:
        with open(ticker_file_path, 'r') as f:
            MY_TICKERS = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ticker file not found at {ticker_file_path}. Please create it.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {ticker_file_path}. Please check the file format.")
        sys.exit(1)

    # 2. Parse command-line arguments
    start_date_arg = None
    end_date_arg = None

    for arg in sys.argv[1:]:
        if arg.startswith("startdate="):
            start_date_arg = arg.split("=")[1]
        elif arg.startswith("enddate="):
            end_date_arg = arg.split("=")[1]

    if not start_date_arg or not end_date_arg:
        print("❌ Missing required arguments: startdate and enddate.")
        print("Usage: python3 concurrentbackfill.py startdate=DDMMYY enddate=DDMMYY")
        sys.exit(1)

    # Parse date strings
    start_date = parse_date_arg(start_date_arg)
    end_date = parse_date_arg(end_date_arg)

    # 3. Call concurrent backfill
    concurrent_backfill(
        tickers=MY_TICKERS,
        start_date=start_date,
        end_date=end_date,
        interval=1,
        exchange="NASDAQ",
    )

    print("✅ Concurrent backfill completed and data inserted into the database.")