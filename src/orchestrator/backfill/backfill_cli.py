#!/usr/bin/env python3
"""Unified Backfill CLI
=======================
A single entry-point to run different backfill modes against the MQS data stack.

Subcommands:
  specific       Backfill a contiguous date range for tickers (inserts directly)
  fill-missing   Detect and patch missing business days per ticker
  concurrent     Parallel backfill across tickers (threaded)
  inject-csv     Load historical CSV dumps (pattern driven) into DB

Examples:
  python -m src.orchestrator.backfill.backfill_cli specific \
      --start 2025-01-01 --end 2025-02-01 --tickers AAPL MSFT --interval 1

  python -m src.orchestrator.backfill.backfill_cli fill-missing \
      --start 2025-01-01 --end 2025-03-31 --tickers AAPL

  python -m src.orchestrator.backfill.backfill_cli concurrent \
      --start 2025-01-01 --end 2025-02-01 --tickers AAPL MSFT GOOGL --interval 1 --threads 6

  python -m src.orchestrator.backfill.backfill_cli inject-csv \
      --csv-dir data/backfill_cache --force

Future Extensions:
  - Add rate limiting flags
  - Add ON CONFLICT toggle
  - Add dry-run mode
  - Add progress metrics aggregation
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Optional, Tuple

# Ensure repository root import path (adjust relative to this file)
CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# Lazy imports inside handlers to avoid loading everything on simple --help

DATE_FMT = "%Y-%m-%d"
ALLOWED_INTERVALS = {1,5,15,30,60}

logger = logging.getLogger("backfill_cli")


def _parse_date(val: str):
    try:
        return datetime.strptime(val, DATE_FMT).date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date '{val}'. Expected format YYYY-MM-DD")


def _ensure_tickers(args) -> List[str]:
    if args.tickers:
        return [t.upper() for t in args.tickers]
    # If user did not pass tickers, try reading tickers.json beside this module's parent
    fallback_path = os.path.join(CURRENT_DIR, "..", "tickers.json")
    if os.path.exists(fallback_path):
        import json
        with open(fallback_path, 'r') as f:
            return [t.upper() for t in json.load(f)]
    raise SystemExit("No tickers specified and tickers.json not found.")


def _validate_interval(interval: int):
    if interval not in ALLOWED_INTERVALS:
        raise SystemExit(f"Interval {interval} not in allowed set {sorted(ALLOWED_INTERVALS)}")


def _row_builder(df, exchange: str) -> Tuple[List[Tuple], int]:
    rows: List[Tuple] = []
    skipped = 0
    for row in df.itertuples(index=False):
        try:
            rows.append(
                (
                    getattr(row, 'ticker'),
                    getattr(row, 'datetime'),
                    getattr(row, 'date'),
                    exchange,
                    float(getattr(row, 'open')),
                    float(getattr(row, 'high')),
                    float(getattr(row, 'low')),
                    float(getattr(row, 'close')),
                    int(float(getattr(row, 'volume'))),
                )
            )
        except Exception:
            skipped += 1
    return rows, skipped

# ---------------------- Subcommand Handlers ---------------------- #

def cmd_specific(args):
    _validate_interval(args.interval)
    tickers = _ensure_tickers(args)
    from src.orchestrator.backfill.backfill import backfill_data
    from common.database.MQSDBConnector import MQSDBConnector
    start = args.start
    end = args.end
    exchange = (args.exchange or 'nasdaq').lower()
    conflict = args.on_conflict
    dry = args.dry_run

    db = MQSDBConnector()
    total_inserted = 0
    total_skipped = 0
    try:
        for ticker in tickers:
            t0 = time.time()
            df = backfill_data(
                tickers=[ticker],
                start_date=start,
                end_date=end,
                interval=args.interval,
                exchange=exchange,
                output_filename=None
            )
            if df is None or df.empty:
                logger.info(f"[{ticker}] No data returned.")
                continue

            rows, skipped = _row_builder(df, exchange)
            total_skipped += skipped
            if not rows:
                logger.warning(f"[{ticker}] All rows invalid or skipped.")
                continue

            if dry:
                logger.info(f"[DRY] {ticker}: would insert {len(rows)} rows (skipped {skipped}).")
                continue

            conn = db.get_connection()
            if conn is None:
                logger.error(f"[{ticker}] DB connection unavailable.")
                continue
            try:
                sql = ("""
                    INSERT INTO market_data (
                      ticker, timestamp, date, exchange,
                      open_price, high_price, low_price, close_price, volume
                    ) VALUES %s
                """.strip())
                if conflict == 'ignore':
                    sql = sql.rstrip() + " ON CONFLICT (ticker, timestamp) DO NOTHING"
                from psycopg2.extras import execute_values as _exec
                with conn.cursor() as cur:
                    _exec(cur, sql, rows)
                conn.commit()
                total_inserted += len(rows)
                elapsed = time.time() - t0
                logger.info(f"[{ticker}] Inserted {len(rows)} rows (skipped {skipped}) in {elapsed:0.2f}s")
            finally:
                db.release_connection(conn)
    finally:
        if not dry:
            logger.info(f"TOTAL inserted={total_inserted} skipped={total_skipped}")
        else:
            logger.info(f"DRY-RUN summary: would insert={total_inserted} skipped={total_skipped}")
        db.close_all_connections()


def cmd_fill_missing(args):
    _validate_interval(args.interval)
    tickers = _ensure_tickers(args)
    from src.orchestrator.backfill.fill_missing_backfill import fill_gaps_for_ticker
    from common.database.MQSDBConnector import MQSDBConnector
    db = MQSDBConnector()
    try:
        for tk in tickers:
            fill_gaps_for_ticker(db, tk, args.start, args.end, interval=args.interval, exchange=args.exchange)
    finally:
        db.close_all_connections()


def cmd_concurrent(args):
    _validate_interval(args.interval)
    tickers = _ensure_tickers(args)
    from src.orchestrator.backfill.concurrent_backfill import concurrent_backfill
    # Pass threads if the implementation supports it (fallback gracefully)
    try:
        concurrent_backfill(
            tickers=tickers,
            start_date=args.start,
            end_date=args.end,
            interval=args.interval,
            exchange=args.exchange,
        )
    except TypeError:
        # If the function signature doesn't accept threads currently
        logger.warning("concurrent_backfill does not accept thread override in current version.")


def cmd_inject_csv(args):
    from src.orchestrator.backfill.injectBackfill import load_csv_files_to_db
    directory = args.csv_dir
    if not os.path.isdir(directory):
        raise SystemExit(f"CSV directory does not exist: {directory}")
    load_csv_files_to_db(directory_path=directory, max_workers=args.threads)


# ---------------------- Parser Construction ---------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="backfill-cli",
        description="Unified interface for MQS market data backfilling operations"
    )
    sub = p.add_subparsers(dest="command", required=True)

    # common date args factory
    def add_date_args(sp):
        sp.add_argument("--start", required=True, type=_parse_date, help="Start date YYYY-MM-DD")
        sp.add_argument("--end", required=True, type=_parse_date, help="End date YYYY-MM-DD")
        sp.add_argument("--tickers", nargs="*", help="Optional explicit tickers (default: read tickers.json)")
        sp.add_argument("--exchange", default="NASDAQ", help="Exchange code (default: NASDAQ)")
        sp.add_argument("--interval", type=int, default=1, help="Bar interval minutes (default: 1)")
        sp.add_argument("--dry-run", action="store_true", help="Fetch & parse but do not insert (where applicable)")

    # specific
    sp_specific = sub.add_parser("specific", help="Backfill continuous date range for tickers and insert into DB")
    add_date_args(sp_specific)
    sp_specific.add_argument("--on-conflict", choices=["ignore","fail"], default="fail", help="Conflict handling (requires unique index if 'ignore')")
    sp_specific.set_defaults(func=cmd_specific)

    # fill-missing
    sp_fill = sub.add_parser("fill-missing", help="Fill only missing business days for tickers")
    add_date_args(sp_fill)
    sp_fill.set_defaults(func=cmd_fill_missing)

    # concurrent
    sp_conc = sub.add_parser("concurrent", help="Concurrent multi-ticker backfill")
    add_date_args(sp_conc)
    sp_conc.add_argument("--threads", type=int, default=6, help="Max worker threads (cap to DB pool size)")
    sp_conc.set_defaults(func=cmd_concurrent)

    # inject-csv
    sp_csv = sub.add_parser("inject-csv", help="Inject previously downloaded CSV dumps into DB")
    sp_csv.add_argument("--csv-dir", required=True, help="Directory containing CSV dumps")
    sp_csv.add_argument("--threads", type=int, default=5, help="Worker threads for CSV ingestion")
    sp_csv.set_defaults(func=cmd_inject_csv)

    return p


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Logging verbosity")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s %(levelname)s: %(message)s')
    logger.debug("Parsed arguments: %s", args)
    args.func(args)


if __name__ == "__main__":
    main()
