"""
concurrent_backfill.py
----------------------
A multi-threaded approach for backfilling multiple tickers in parallel
using the existing 'backfill_data' function from 'backfill.py'.
Each ticker is processed in its own thread to reduce total runtime.

Results:
  - Data is injected directly into the 'market_data' table in the database using MQSDBConnector.
"""

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from psycopg2.extras import execute_values

# Ensure we can import backfill.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.common.database.MQSDBConnector import MQSDBConnector
from src.orchestrator.backfill.backfill import backfill_data

logger = logging.getLogger(__name__)
# Number of threads to use. NEEDS TO BE LESS THAN MQSDBCONNECTOR MAX CONN VALUE!
MAX_WORKERS = 8


def parse_date_arg(date_str):
    """Parses date string in DDMMYY format and returns a datetime.date object."""
    try:
        return datetime.strptime(date_str, "%d%m%y").date()
    except ValueError:
        print(
            f"❌ Invalid date format: {date_str}. Expected format: DDMMYY (e.g., 040325 for March 4, 2025)."
        )
        sys.exit(1)


def parse_main_args(argv: list[str]) -> tuple[str, str]:
    start_date_arg = None
    end_date_arg = None

    for arg in argv:
        if arg.startswith("startdate="):
            start_date_arg = arg.split("=", 1)[1]
        elif arg.startswith("enddate="):
            end_date_arg = arg.split("=", 1)[1]

    if not start_date_arg or not end_date_arg:
        print("❌ Missing required arguments: startdate and enddate.")
        print("Usage: python3 concurrent_backfill.py startdate=DDMMYY enddate=DDMMYY")
        sys.exit(1)

    return start_date_arg, end_date_arg


def backfill_single_ticker(
    ticker,
    start_date,
    end_date,
    interval,
    exchange,
    db_connector,
    dry_run=False,
    on_conflict="fail",
    exchange_map=None,
):
    """
    Calls backfill_data(...) for a single ticker and injects data into the DB.
    This function now receives a shared MQSDBConnector instance.
    """
    conn = None
    effective_exchange = (exchange_map or {}).get(ticker.upper(), exchange or "nasdaq")

    try:
        # Fetch the data in-memory (output_filename=None => returns DataFrame)
        df = backfill_data(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            exchange=effective_exchange,
            output_filename=None,
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
                insert_data.append(
                    (
                        row["ticker"],
                        row["datetime"],  # timestamp
                        row["date"],
                        effective_exchange.lower() if effective_exchange else "nasdaq",
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        int(float(row["volume"])),
                    )
                )
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
        if on_conflict == "ignore":
            insert_sql += " ON CONFLICT (ticker, timestamp) DO NOTHING"

        if insert_data and not dry_run:
            with conn.cursor() as cursor:
                # Bound waits so a stuck server-side lock surfaces as an
                # error instead of wedging the worker (and the pool) forever.
                # The error path in this function rolls back, so the
                # connection is safe to reuse.
                cursor.execute("SET LOCAL lock_timeout = '30s'")
                cursor.execute("SET LOCAL statement_timeout = '120s'")
                execute_values(cursor, insert_sql, insert_data)
            conn.commit()
            print(f"[{ticker}] Inserted {len(insert_data)} rows into DB.")
        elif dry_run:
            logger.info(f"[{ticker}:DRY_RUN] Rows prepared: {len(insert_data)}.")
        else:
            print(f"[{ticker}] No valid rows to insert.")

    except Exception as e:
        print(f"[{ticker}] Error during backfill or insert: {e}")
        # Roll back any aborted transaction so the pooled connection
        # is not poisoned for the next worker that picks it up.
        if conn is not None:
            try:
                conn.rollback()
            except Exception as rb_ex:
                print(f"[{ticker}] Rollback failed: {rb_ex}")
                # Mark connection bad so the pool drops it instead of reusing.
                try:
                    conn.close()
                except Exception:
                    pass
    finally:
        # Release the connection back to the pool
        if conn:
            db_connector.release_connection(conn)


def concurrent_backfill(
    tickers,
    start_date,
    end_date,
    interval,
    exchange=None,
    exchange_map=None,
    dry_run=False,
    on_conflict="fail",
    threads=MAX_WORKERS,
):
    """
    Spawns multiple threads, each calling 'backfill_data' for a single ticker.
    Injects each ticker's results directly into the DB using a shared connector.
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    display_exchange = exchange
    if exchange_map:
        unique_exchanges = sorted(set(exchange_map.values()))
        display_exchange = (
            unique_exchanges[0] if len(unique_exchanges) == 1 else "mixed"
        )

    print(f"[ConcurrentBackfill] Starting concurrency for {len(tickers)} tickers.")
    print(
        f"  Date range: {start_date} to {end_date}, interval={interval} min, exchange={display_exchange}"
    )
    print(f"  Using up to {threads} threads.")

    # Create ONE shared database connector instance
    db_connector = MQSDBConnector()
    try:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Pass the single db_connector instance to each worker
            futures = [
                executor.submit(
                    backfill_single_ticker,
                    ticker,
                    start_date,
                    end_date,
                    interval,
                    exchange,
                    db_connector,
                    dry_run,
                    on_conflict,
                    exchange_map,
                )
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
