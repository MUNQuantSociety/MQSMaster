"""
specific_backfill.py
---------------
Backfills historical data for specified tickers and injects it directly into the database, with gap filling measures.
"""

import sys
from datetime import datetime
import logging
from psycopg2.extras import execute_values
from psycopg2.errors import UniqueViolation
from src.common.database.MQSDBConnector import MQSDBConnector
from src.orchestrator.backfill.backfill import backfill_data, BATCH_DAYS

logger = logging.getLogger(__name__)


def parse_date_arg(date_str):
    """Parses date string in DDMMYY format and returns a datetime.date object."""
    try:
        return datetime.strptime(date_str, "%d%m%y").date()
    except ValueError:
        print(f"❌ Invalid date format: {date_str}. Expected format: DDMMYY (e.g., 040325 for March 4, 2025).")
        sys.exit(1)

def backfill_db(tickers, start_date, end_date, interval, exchange, dry_run, on_conflict='fail', output=None):
    per_ticker = {ticker: {"prepared": 0, "inserted": 0, "skipped": 0, "total": 0} for ticker in tickers}
    agg_prepared = 0
    agg_inserted = 0
    agg_skipped = 0

    db = MQSDBConnector()

    for ticker in tickers:
        conn = None
        try:
            insert_data = []
            df = backfill_data(
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                exchange=exchange,
                output_filename=output
            )

            if df is None or df.empty:
                logger.info("[%s] No data returned from backfill.", ticker)
                continue

            per_ticker[ticker]["total"] = len(df)
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
                    per_ticker[ticker]["prepared"] += 1
                except Exception as e:
                    per_ticker[ticker]["skipped"] += 1
                    logger.warning("[%s] Skipping row due to parsing error: %s", ticker, e)

            # Build insert SQL
            insert_sql = """
                INSERT INTO market_data (
                    ticker, timestamp, date, exchange,
                    open_price, high_price, low_price, close_price, volume
                )
                VALUES %s
            """
            if on_conflict == "ignore":
                insert_sql += " ON CONFLICT (ticker, timestamp) DO NOTHING"

            prepared = len(insert_data)
            if prepared and not dry_run:
                try:
                    with conn.cursor() as cursor:
                        execute_values(cursor, insert_sql, insert_data)
                    conn.commit()
                    # We approximate inserted as prepared here; conflict-ignored rows are not counted separately.
                    per_ticker[ticker]["inserted"] = prepared
                    logger.info("[%s] prepared=%d inserted=%d skipped=%d total=%d",
                                ticker, prepared, per_ticker[ticker]["inserted"], per_ticker[ticker]["skipped"], per_ticker[ticker]["total"])
                    # fill gaps in the database for any missing or skipped dates
                    fill_gaps_for_ticker(db, ticker, start_date, end_date, interval, exchange, dry_run, on_conflict)
                except UniqueViolation as uv:
                    # Clean handling for duplicate key errors when on_conflict='fail'
                    conn.rollback()
                    logger.error("[%s] Duplicate key violation (likely existing rows). Consider --on-conflict ignore.\n\n[WARNING]: %s", ticker, uv)
            elif dry_run:
                logger.info("[%s:DRY_RUN] prepared=%d skipped=%d total=%d",
                            ticker, prepared, per_ticker[ticker]["skipped"], per_ticker[ticker]["total"])
            else:
                logger.info("[%s] No valid rows to insert.", ticker)
        except Exception as e:
            logger.exception("[%s] Error during backfill or insert: %s", ticker, e)
        finally:
            if conn:
                db.release_connection(conn)

        # Aggregate totals after each ticker
        agg_prepared += per_ticker[ticker]["prepared"]
        agg_inserted += per_ticker[ticker]["inserted"]
        agg_skipped  += per_ticker[ticker]["skipped"]

    return {
        "prepared": agg_prepared,
        "inserted": agg_inserted,
        "skipped": agg_skipped,
        "per_ticker": per_ticker,
    }

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
                logger.info(f"[{ticker}:DRY_RUN] Rows prepared: {len(insert_tuples)}")

            else:
                logger.info(f"[{ticker}] Inserting {len(insert_tuples)} rows…")
                from psycopg2.extras import execute_values
                with conn.cursor() as cur:
                    execute_values(cur, sql, insert_tuples)
                conn.commit()
                print(f"    → inserted {len(insert_tuples)} rows")
        finally:
            db.release_connection(conn)
    logger.info(f"[{ticker}] gap fill complete.")