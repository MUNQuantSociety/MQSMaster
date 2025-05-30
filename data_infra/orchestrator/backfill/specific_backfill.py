"""
specific_backfill.py
---------------
Backfills historical data for specified tickers and injects it directly into the database.
"""

import sys
import os
from datetime import datetime
from psycopg2.extras import execute_values
import json

# Ensure we can import backfill.py from the orchestrator dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from data_infra.database.MQSDBConnector import MQSDBConnector
from data_infra.orchestrator.backfill.backfill import backfill_data

def parse_date_arg(date_str):
    """Parses date string in DDMMYY format and returns a datetime.date object."""
    try:
        return datetime.strptime(date_str, "%d%m%y").date()
    except ValueError:
        print(f"❌ Invalid date format: {date_str}. Expected format: DDMMYY (e.g., 040325 for March 4, 2025).")
        sys.exit(1)

def backfill_db(tickers, start_date, end_date, interval, exchange):
    for ticker in tickers:
        try:
            df = backfill_data(
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                exchange=exchange,
                output_filename=None  # Do not write to CSV
            )

            if df is None or df.empty:
                print(f"[{ticker}] No data returned from backfill.")
                continue

            db = MQSDBConnector()
            conn = db.get_connection()

            insert_data = []
            for _, row in df.iterrows():
                try:
                    insert_data.append((
                        row['ticker'],
                        row['datetime'],
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

if __name__ == "__main__":
    # 1. Load tickers from tickers.json
    script_dir = os.path.dirname(__file__)
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

    # Validate date arguments
    if not start_date_arg or not end_date_arg:
        print("❌ Missing required arguments: startdate and enddate.")
        print("Usage: python3 specific_backfill.py startdate=040325 enddate=300325")
        sys.exit(1)

    # Parse and convert date strings
    start_date = parse_date_arg(start_date_arg)
    end_date = parse_date_arg(end_date_arg)

    # 3. Perform backfill and inject into DB
    backfill_db(
        tickers=MY_TICKERS,
        start_date=start_date,
        end_date=end_date,
        interval=1,
        exchange="NASDAQ"
    )

    print("✅ Specific backfill completed and data inserted into the database.")
