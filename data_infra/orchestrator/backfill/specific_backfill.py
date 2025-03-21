"""
specific_backfill.py
---------------
Backfills historical data for specified tickers and injects it directly into the database.
"""

import sys
import os
from datetime import datetime, timedelta
from psycopg2.extras import execute_values
from data_infra.database.MQSDBConnector import MQSDBConnector

# Ensure we can import backfill.py from the orchestrator dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from data_infra.orchestrator.backfill import backfill_data

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
    # 1. Define tickers
    my_tickers = ['TXG', 'MMM', 'ETNB']

    # 2. Define date range (e.g., last 5 trading days)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=6)

    # 3. Perform backfill and inject into DB
    backfill_db(
        tickers=my_tickers,
        start_date=start_date,
        end_date=end_date,
        interval=1,
        exchange="NASDAQ"
    )

    print("✅ Specific backfill completed and data inserted into the database.")
