"""
specific_backfill.py
---------------
Backfills historical data for specified tickers and injects it directly into the database.
"""

import sys
import os
from datetime import datetime
import logging
from psycopg2.extras import execute_values
import json

logger = logging.getLogger(__name__)
# Ensure we can import backfill.py from the orchestrator dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.common.database.MQSDBConnector import MQSDBConnector
from src.orchestrator.backfill.backfill import backfill_data

def parse_date_arg(date_str):
    """Parses date string in DDMMYY format and returns a datetime.date object."""
    try:
        return datetime.strptime(date_str, "%d%m%y").date()
    except ValueError:
        print(f"❌ Invalid date format: {date_str}. Expected format: DDMMYY (e.g., 040325 for March 4, 2025).")
        sys.exit(1)

def backfill_db(tickers, start_date, end_date, interval, exchange, dry_run, on_conflict='fail', output=None):
    stats = {ticker: {"inserted": 0, "skipped": 0, "total": 0} for ticker in tickers}
    db = MQSDBConnector()

    for ticker in tickers:
        try:
            insert_data = []
            df = backfill_data(
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                exchange=exchange,
                output_filename=output  # Pass output filename if needed
            )

            if df is None or df.empty:
                print(f"[{ticker}] No data returned from backfill.")
                continue
            
            stats[ticker]["total"] = len(df)
            conn = db.get_connection()

            if conn is None:
                logger.error("No DB connection available for %s", ticker)
                continue

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
                    stats[ticker]["inserted"] += 1
                    
                except Exception as e:
                    stats[ticker]["skipped"] += 1
                    print(f"[{ticker}] Skipping row due to parsing error: {e}")
                    continue

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
                    execute_values(cursor, insert_sql, insert_data)
                conn.commit()
                logger.info(f"[{ticker}] Rows inserted: {len(insert_data)}, Rows skipped: {stats[ticker]['skipped']}, Total rows: {stats[ticker]['total']}")
                fill_gaps_for_ticker(db, ticker, start_date, end_date, interval=interval, exchange=exchange, dry_run=dry_run, on_conflict=on_conflict)
                print("✅ Specific backfill completed and data inserted into the database.")
            elif dry_run:
                fill_gaps_for_ticker(db, ticker, start_date, end_date, interval=interval, exchange=exchange, dry_run=dry_run, on_conflict=on_conflict)
                logger.info(f"\n\t[{ticker}:DRY_RUN]\n\tRows inserted: {len(insert_data)}\n\tRows skipped:  {stats[ticker]['skipped']}\n\tTotal rows:    {stats[ticker]['total']}\n")
            else:
                print(f"[{ticker}] No valid rows to insert.")
        except Exception as e:
            print(f"[{ticker}] Error during backfill or insert: {e}")
        finally:
            if 'conn' in locals():
                db.release_connection(conn)
    return {"prepared": len(insert_data), "inserted": len(insert_data) - stats[ticker]["skipped"], "skipped": stats[ticker]["skipped"]}

from src.orchestrator.backfill.backfill import BATCH_DAYS
def get_existing_dates(db, ticker, start_date, end_date):
    sql = """
      SELECT DISTINCT date
      FROM market_data
      WHERE ticker = %s
        AND date BETWEEN %s AND %s
    """
    res = db.execute_query(sql, (ticker, start_date, end_date), fetch=True)
    if res.get("status") != "success":
        raise RuntimeError(f"DB error fetching dates for {ticker}: {res}")

    have = set()
    for row in res.get("data", []):
        # if row is a dict, grab row["date"], otherwise assume tuple/list
        if isinstance(row, dict):
            have.add(row.get("date"))
        else:
            have.add(row[0])
    return have

def chunk_consecutive(dates, max_chunk_size):
    """
    dates: sorted list of date objects
    yields lists of up to max_chunk_size consecutive dates
    """
    from itertools import groupby
    for _, group in groupby(enumerate(dates),
                            key=lambda ix: ix[0] - ix[1].toordinal()):
        block = [dt for _, dt in group]
        # further split into BATCH_DAYS sized pieces
        for i in range(0, len(block), max_chunk_size):
            yield block[i : i + max_chunk_size]

def fill_gaps_for_ticker(db, ticker, start_date, end_date, interval=1, exchange="NASDAQ", dry_run=False, on_conflict="fail"):
    import pandas as pd
    # 1) list all real trading days
    all_days = pd.bdate_range(start_date, end_date).date

    # 2) find which days already have data
    have = get_existing_dates(db, ticker, start_date, end_date)

    # 3) missing days = set difference
    missing = sorted(d for d in all_days if d not in have)
    if not missing:
        logger.info(f"[{ticker}] no days to fill.")
        return

    logger.warning(f"[{ticker}] missing {len(missing)} days: {missing}")

    # 4) batch consecutive days into sub-lists of up to BATCH_DAYS
    for batch in chunk_consecutive(missing, BATCH_DAYS):
        from_day = batch[0].strftime("%Y-%m-%d")
        to_day   = batch[-1].strftime("%Y-%m-%d")
        print(f"  Backfilling {ticker} from {from_day} to {to_day}…")
        df = backfill_data(
            tickers=[ticker],
            start_date=from_day,
            end_date=to_day,
            interval=interval,
            exchange=exchange,
            output_filename=None
        )
        if df is None or df.empty:
            logger.warning(f"    → no data returned for {from_day} to /{to_day}")
            continue

        # 5) insert returned rows into DB
        conn = db.get_connection()
        try:
            insert_tuples = [
                (
                    row.ticker,
                    row.datetime,
                    row.date,
                    exchange.lower(),
                    float(row.open),
                    float(row.high),
                    float(row.low),
                    float(row.close),
                    int(row.volume),
                )
                for row in df.itertuples(index=False)
            ]
            
            sql = """
              INSERT INTO market_data (
                ticker, timestamp, date, exchange,
                open_price, high_price, low_price, close_price, volume
              ) VALUES %s
            """
            on_conflict = on_conflict.lower()
            if on_conflict == 'ignore':
                sql += " ON CONFLICT (ticker, timestamp) DO NOTHING"

            if dry_run:
                logger.info(f"[{ticker}:DRY_RUN] Rows inserted: {len(insert_tuples)}")
                continue
            else:
                logger.info(f"[{ticker}] Inserting {len(insert_tuples)} rows…")
                from psycopg2.extras import execute_values
                with conn.cursor() as cur:
                    execute_values(cur, sql, insert_tuples)
                conn.commit()
                print(f"    → inserted {len(insert_tuples)} rows")
        finally:
            db.release_connection(conn)


if __name__ == "__main__":
    # 1. Load tickers from tickers.json
    script_dir = os.path.dirname(__file__)
    ticker_file_path = os.path.join(script_dir, 'tickers.json')

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
    result_summary = backfill_db(
        tickers=MY_TICKERS,
        start_date=start_date,
        end_date=end_date,
        interval=1,
        exchange="NASDAQ",
        dry_run=False  # Set to True to skip actual DB insertion for testing
    )

    # Print summary of results
    print("\nBackfill summary:")
    print("----------------")
    print(f"Prepared: {result_summary['prepared']}")
    print(f"Inserted: {result_summary['inserted']}")
    print(f"Skipped:  {result_summary['skipped']}")
    print("✅ Specific backfill completed and data inserted into the database.")
