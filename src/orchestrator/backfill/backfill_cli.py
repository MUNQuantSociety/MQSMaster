import argparse
import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger("backfill_cli")

# Ensure repository root import path (adjust relative to this file)
CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
DATE_FMT = "%d%m%y"
ALLOWED_INTERVALS = {1,5,15,30,60}

def _parse_date(date_str: str):
    try:
        return datetime.strptime(date_str, DATE_FMT).date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date '{date_str}'. Expected format DDMMYY (e.g., 040325 for March 4, 2025).")

def _ensure_tickers(args) -> List[str]:
    if args.tickers:
        return [t.upper() for t in args.tickers]
    # If user did not pass tickers, try reading tickers.json
    fallback_path = os.path.join(CURRENT_DIR, "tickers.json")
    if os.path.exists(fallback_path):
        import json
        with open(fallback_path, 'r') as f:
            tickers = json.load(f)
            cont = input ("No tickers specified. Continue with first 5 tickers? [y/n]: ")
            if cont.lower() != 'y':
                raise SystemExit("Aborted by user.")
            logger.info("Loaded first 5 tickers from %s", fallback_path)
        return tickers[:5]
    raise SystemExit(f"No tickers specified and tickers.json not found. {fallback_path}")

def _validate_interval(interval: int):
    if interval not in ALLOWED_INTERVALS:
        raise SystemExit(f"Interval {interval} not in allowed set {sorted(ALLOWED_INTERVALS)}")

# ---------------------- Subcommand Handlers ---------------------- #
# Lazy imports inside handlers to avoid loading everything on simple --help
def cmd_specific(args):
    _validate_interval(args.interval)
    tickers = _ensure_tickers(args)
    start = args.start
    end = args.end
    if start > end:
        raise SystemExit("Start date must not be after end date")
    else:
        stats_total = {"inserted": 0, "skipped": 0, "tickers": 0}
    from src.orchestrator.backfill.specific_backfill import backfill_db

    exchange = (args.exchange or 'nasdaq').lower()
    dry_run = args.dry_run
    on_conflict = args.on_conflict.lower()
    output = args.output_filename

    wall_start = datetime.now()

    try:
        for ticker in tickers:
            t_start = time.time()
            per = backfill_db(
                tickers=[ticker],
                start_date=start,
                end_date=end,
                interval=args.interval,
                exchange=exchange,
                dry_run=dry_run,
                on_conflict=on_conflict,
                output=output
            )
            stats_total["inserted"] += per["inserted"]
            stats_total["skipped"]  += per.get("skipped", 0)
            stats_total["tickers"]  += 1

            elapsed = time.time() - t_start
            logger.info(f"[{ticker}] Inserted in {elapsed:0.2f}s")
            print('-----------------------------\n')
    finally:
        total_elapsed = datetime.now() - wall_start
        elapsed_str = str(total_elapsed).split('.')[0]
        logger.info(
            "Summary: tickers=%d inserted=%d skipped=%d elapsed=%s",
            stats_total["tickers"], stats_total["inserted"], stats_total["skipped"], elapsed_str
        )


def cmd_concurrent(args):
    _validate_interval(args.interval)
    tickers = _ensure_tickers(args)
    from src.orchestrator.backfill.concurrent_backfill import concurrent_backfill
    try:
        concurrent_backfill(
            tickers=tickers,
            start_date=args.start,
            end_date=args.end,
            interval=args.interval,
            exchange=args.exchange.lower(),
            dry_run=args.dry_run,
            on_conflict=args.on_conflict.lower(),
            threads=args.threads
        )
    except Exception as e:
        logger.error("Error: concurrent_backfill failed: %s", e)
        raise 


def cmd_inject_csv(args):
    from src.orchestrator.backfill.injectBackfill import load_csv_files_to_db
    directory = args.csv_dir
    threads = args.threads

    if not os.path.isdir(directory):
        raise SystemExit(f"Directory not found: {directory}")
    logger.info(f"Injecting CSV files from directory: {directory} using {threads} threads")

    try:
        load_csv_files_to_db(directory_path=directory, max_workers=threads)
    except Exception as e:
        logger.error("Error injecting CSV files: %s", e)
        raise SystemExit(1)


# ---------------------- Parser Construction ---------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="backfill-cli",
        description="Unified interface for MQS market data backfilling operations"
    )
    sub = p.add_subparsers(dest="command", required=True)

    # common args
    def add_date_args(sp):
        sp.add_argument("--start", required=True, type=_parse_date, help="Start date DDMMYY (e.g., 040325 for March 4, 2025)")
        sp.add_argument("--end", required=True, type=_parse_date, help="End date DDMMYY (e.g., 040325 for March 4, 2025)")
        sp.add_argument("--tickers", nargs="+", help="Optional explicit tickers (default: read tickers.json)")
        sp.add_argument("--exchange", default="NASDAQ", help="Exchange code (default: NASDAQ)")
        sp.add_argument("--interval", type=int, default=1, help="Bar interval minutes (default: 1)")
        sp.add_argument("--dry-run", action="store_true", help="Fetch & parse but do not insert (where applicable)")
        sp.add_argument("--output-filename", type=str, default=None, help="Output CSV filename (default: None)")
        sp.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Logging level (default: INFO)")
        sp.add_argument("--on-conflict", choices=["ignore","fail"], default="fail", help="Conflict handling (requires unique index if 'ignore')")
        sp.set_defaults(func=lambda args: None)  # default no-op

    # specific
    sp_specific = sub.add_parser("specific", help="Backfill continuous date range for tickers and insert into DB")
    add_date_args(sp_specific)
    sp_specific.set_defaults(func=cmd_specific)

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
    args = parser.parse_args(argv)
    log_level = getattr(args, 'log_level', 'INFO')
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format='%(asctime)s %(levelname)s: %(message)s')
    logger.debug("Parsed arguments: %s", args)
    if hasattr(args, 'func') and args.func:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
