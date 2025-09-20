#!/usr/bin/env python3
"""
fill_missing_backfill.py

For each ticker in tickers.json, looks at your market_data table to find
business days with no bars (or fewer than expected bars), then calls
backfill_data() to fetch & insert exactly those missing days.
"""

import sys, os
from datetime import datetime
import pandas as pd
from itertools import groupby
from operator import itemgetter
import json

# make your modules importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            "../../..")))

from common.database.MQSDBConnector import MQSDBConnector
from data_infra.orchestrator.backfill.backfill import backfill_data

# how many days per API batch
from data_infra.orchestrator.backfill.backfill import BATCH_DAYS

def parse_date_arg(arg):
    return datetime.strptime(arg, "%Y-%m-%d").date()

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
    for _, group in groupby(enumerate(dates),
                            key=lambda ix: ix[0] - ix[1].toordinal()):
        block = [dt for _, dt in group]
        # further split into BATCH_DAYS sized pieces
        for i in range(0, len(block), max_chunk_size):
            yield block[i : i + max_chunk_size]

def fill_gaps_for_ticker(db, ticker, start_date, end_date,
                         interval=1, exchange="NASDAQ"):
    # 1) list all real trading days
    all_days = pd.bdate_range(start_date, end_date).date

    # 2) find which days already have data
    have = get_existing_dates(db, ticker, start_date, end_date)

    # 3) missing days = set difference
    missing = sorted(d for d in all_days if d not in have)
    if not missing:
        print(f"[{ticker}] no days to fill.")
        return

    print(f"[{ticker}] missing {len(missing)} days: {missing}")

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
            print(f"    → no data returned for {from_day}–{to_day}")
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
            from psycopg2.extras import execute_values
            with conn.cursor() as cur:
                execute_values(cur, sql, insert_tuples)
            conn.commit()
            print(f"    → inserted {len(insert_tuples)} rows")
        finally:
            db.release_connection(conn)

def main():
    script_dir = os.path.dirname(__file__)

    # take dates as YYYY-MM-DD
    if len(sys.argv) not in (3, 4):
        print("Usage: python fill_missing_backfill.py START END [TICKER]")
        print(" e.g. python fill_missing_backfill.py 2025-03-01 2025-06-27 AAPL")
        sys.exit(1)

    start = parse_date_arg(sys.argv[1])
    end   = parse_date_arg(sys.argv[2])

    # Load full tickers list
    with open(os.path.join(script_dir, "..", "tickers.json")) as f:
        tickers = json.load(f)

    # If a single ticker was passed, override the list
    if len(sys.argv) == 4:
        tickers = [sys.argv[3].upper()]
    db = MQSDBConnector()
    try:
        for tk in tickers:
            fill_gaps_for_ticker(db, tk, start, end,
                                 interval=1,
                                 exchange="NASDAQ")
    finally:
        db.close_all_connections()

if __name__ == "__main__":
    main()
