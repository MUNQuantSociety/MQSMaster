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
# Ensure we can import backfill.py from the orchestrator dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from data_infra.orchestrator.backfill.backfill import backfill_data
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from data_infra.database.MQSDBConnector import MQSDBConnector
from psycopg2.extras import execute_values


# Number of threads to use. Adjust based on CPU/network constraints.
MAX_WORKERS = 3

def parse_date_arg(date_str):
    """Parses date string in DDMMYY format and returns a datetime.date object."""
    try:
        return datetime.strptime(date_str, "%d%m%y").date()
    except ValueError:
        print(f"❌ Invalid date format: {date_str}. Expected format: DDMMYY (e.g., 040325 for March 4, 2025).")
        sys.exit(1)

def backfill_single_ticker(ticker, start_date, end_date, interval, exchange, _):
    """
    Calls backfill_data(...) for a single ticker. 
    Instead of writing to CSV, injects directly into DB.
    """
    try:
        # Fetch the data
        df = backfill_data(
            tickers=[ticker],   # pass a list of length 1 
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            exchange=exchange,
            output_filename=None  # Make sure backfill_data handles None as "don't write"
        )

        if df is None or df.empty:
            print(f"[{ticker}] No data returned from backfill.")
            return

        # Create DB connection
        db = MQSDBConnector()
        conn = db.get_connection()

        # Prepare the data for insertion
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
            except Exception as e:
                print(f"[{ticker}] Skipping row due to parsing error: {e}")
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
        if 'conn' in locals():
            db.release_connection(conn)

def concurrent_backfill(
    tickers, 
    start_date, 
    end_date, 
    interval, 
    exchange=None,
    output_prefix="2y_mkt_data"  # kept for compatibility but unused
):
    """
    Spawns multiple threads, each calling 'backfill_data' for a single ticker.
    Injects each ticker's results directly into the DB.
    """
    # Convert input dates if they are strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    print(f"[ConcurrentBackfill] Starting concurrency for {len(tickers)} tickers.")
    print(f"  Date range: {start_date} to {end_date}, interval={interval} min, exchange={exchange}")
    print(f"  Using up to {MAX_WORKERS} threads.")

    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for ticker in tickers:
            fut = executor.submit(
                backfill_single_ticker,
                ticker,
                start_date,
                end_date,
                interval,
                exchange,
                None  # output_filename is no longer used
            )
            futures.append(fut)

        for fut in futures:
            try:
                fut.result()
            except Exception as ex:
                print(f"[ConcurrentBackfill:ERROR] A worker failed with: {ex}")

    print("[ConcurrentBackfill] All threads completed.")



if __name__ == "__main__":
    # 1. Define tickers
    MY_TICKERS = ['TXG', 'MMM', 'ETNB', 'ATEN', 'AAON', 'AIR', 'ABT', 'ABBV', 'ANF', 'ABM']

    # 2. Parse command-line arguments
    start_date_arg = None
    end_date_arg = None

    for arg in sys.argv[1:]:
        if arg.startswith("startdate="):
            start_date_arg = arg.split("=")[1]
        elif arg.startswith("enddate="):
            end_date_arg = arg.split("=")[1]

    # Validate date arguments
    if not start_date_arg or not end_date_arg:
        print("❌ Missing required arguments: startdate and enddate.")
        print("Usage: python3 concurrentbackfill.py startdate=040325 enddate=300325")
        sys.exit(1)

    # Parse and convert date strings
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