import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger("backfill_cli")

# Ensure repository root import path (adjust relative to this file)
CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
DATE_FMT = "%d%m%y"
ALLOWED_INTERVALS = {1, 5, 15, 30, 60}


def _parse_date(date_str: str):
    try:
        return datetime.strptime(date_str, DATE_FMT).date()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{date_str}'. Expected format DDMMYY (e.g., 040325 for March 4, 2025)."
        )


def _ensure_tickers(args) -> List[str]:
    if args.tickers:
        return [t.upper() for t in args.tickers]
    fallback_path = _resolve_ticker_file("tickers.json")
    tickers = _load_tickers_file(fallback_path, "default tickers")
    if not tickers:
        raise SystemExit(f"tickers.json is empty: {fallback_path}")
    if not (args.yes or not sys.stdin.isatty()):
        cont = input(
            f"No tickers specified. Continue with all {len(tickers)} tickers from tickers.json? [y/n]: "
        )
        if cont.lower() != "y":
            raise SystemExit("Aborted by user.")
    logger.info("Loaded %d tickers from %s", len(tickers), fallback_path)
    return tickers


def _validate_interval(interval: int):
    if interval not in ALLOWED_INTERVALS:
        raise SystemExit(
            f"Interval {interval} not in allowed set {sorted(ALLOWED_INTERVALS)}"
        )


def _resolve_ticker_file(filename: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = [
        os.path.join(script_dir, "update", "extra_tickers"),  # update layout
        os.path.abspath(os.path.join(script_dir, "..")),  # expected location
        script_dir,  # fallback: same folder
        os.getcwd(),  # fallback: run directory
    ]

    for base in candidate_dirs:
        path = os.path.join(base, filename)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Could not locate {filename}. Checked: "
        + ", ".join(os.path.join(d, filename) for d in candidate_dirs)
    )


def _load_tickers_file(path: str, label: str) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"[EXCHANGE]: Missing {label} file: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(
            f"[EXCHANGE]: Invalid JSON in {label} file: {path} (line {e.lineno}, col {e.colno})"
        )
        sys.exit(1)
    except OSError as e:
        logging.error(f"[EXCHANGE]: Could not read {label} file: {path} ({e})")
        sys.exit(1)

    if not isinstance(data, list):
        logging.error(f"[EXCHANGE]: Expected a JSON array in {label} file: {path}")
        sys.exit(1)

    return [str(t).strip().upper() for t in data if str(t).strip()]


def _load_all_ticker_lists() -> tuple[
    list[str], list[str], list[str], list[str], list[str]
]:
    ticker_file_path = _resolve_ticker_file("tickers.json")
    crypto_ticker_file_path = _resolve_ticker_file("crypto_tickers.json")
    sp500_ticker_file_path = _resolve_ticker_file("sp500_tickers.json")
    commodity_ticker_file_path = _resolve_ticker_file("commodity_tickers.json")
    nasdaq_ticker_file_path = _resolve_ticker_file("nasdaq_tickers.json")

    my_tickers = _load_tickers_file(ticker_file_path, "default tickers")
    crypto_tickers = _load_tickers_file(crypto_ticker_file_path, "crypto tickers")
    sp500_tickers = _load_tickers_file(sp500_ticker_file_path, "sp500 tickers")
    commodity_tickers = _load_tickers_file(
        commodity_ticker_file_path, "commodity tickers"
    )
    nasdaq_tickers = _load_tickers_file(nasdaq_ticker_file_path, "nasdaq tickers")
    return my_tickers, crypto_tickers, sp500_tickers, commodity_tickers, nasdaq_tickers


def _build_exchange_sets() -> dict[str, set[str]]:
    _my_tickers, crypto_tickers, sp500_tickers, commodity_tickers, nasdaq_tickers = (
        _load_all_ticker_lists()
    )
    return {
        "crypto": set(crypto_tickers),
        "commodity": set(commodity_tickers),
        "sp500": set(sp500_tickers),
        "nasdaq": set(nasdaq_tickers),
    }


def _identify_exchange_for_ticker(
    ticker: str, exchange_sets: dict[str, set[str]]
) -> str:
    symbol = ticker.upper()
    if symbol in exchange_sets["crypto"]:
        return "crypto"
    if symbol in exchange_sets["commodity"]:
        return "commodity"
    if symbol in exchange_sets["sp500"]:
        return "sp500"
    if symbol in exchange_sets["nasdaq"]:
        return "nasdaq"
    return "nasdaq"


def _build_exchange_map(selected_tickers: list[str]) -> dict[str, str]:
    exchange_sets = _build_exchange_sets()
    return {
        ticker.upper(): _identify_exchange_for_ticker(ticker, exchange_sets)
        for ticker in selected_tickers
    }


# ---------------------- Subcommand Handlers ---------------------- #
# Lazy imports inside handlers to avoid loading everything on simple --help
def cmd_specific(args):
    _validate_interval(args.interval)
    tickers = _ensure_tickers(args)
    exchange_map = _build_exchange_map(tickers)
    start = args.start
    end = args.end
    if start > end:
        raise SystemExit("Start date must not be after end date")
    else:
        stats_total = {"inserted": 0, "skipped": 0, "tickers": 0}
    from src.orchestrator.backfill.specific_backfill import backfill_db

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
                exchange=exchange_map.get(ticker.upper(), "nasdaq"),
                dry_run=dry_run,
                on_conflict=on_conflict,
                output=output,
            )
            stats_total["inserted"] += per["inserted"]
            stats_total["skipped"] += per.get("skipped", 0)
            stats_total["tickers"] += 1

            elapsed = time.time() - t_start
            logger.info(f"[{ticker}] Inserted in {elapsed:0.2f}s")
            print("-----------------------------\n")
    finally:
        total_elapsed = datetime.now() - wall_start
        elapsed_str = str(total_elapsed).split(".")[0]
        logger.info(
            "Summary: tickers=%d inserted=%d skipped=%d elapsed=%s",
            stats_total["tickers"],
            stats_total["inserted"],
            stats_total["skipped"],
            elapsed_str,
        )


def cmd_concurrent(args):
    _validate_interval(args.interval)
    tickers = _ensure_tickers(args)
    exchange_map = _build_exchange_map(tickers)
    from src.orchestrator.backfill.concurrent_backfill import concurrent_backfill

    try:
        concurrent_backfill(
            tickers=tickers,
            start_date=args.start,
            end_date=args.end,
            interval=args.interval,
            exchange="nasdaq",
            exchange_map=exchange_map,
            dry_run=args.dry_run,
            on_conflict=args.on_conflict.lower(),
            threads=args.threads,
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
    logger.info(
        f"Injecting CSV files from directory: {directory} using {threads} threads"
    )

    try:
        load_csv_files_to_db(directory_path=directory, max_workers=threads)
    except Exception as e:
        logger.error("Error injecting CSV files: %s", e)
        raise SystemExit(1)


# ---------------------- Parser Construction ---------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="backfill-cli",
        description="Unified interface for MQS market data backfilling operations",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # common args
    def add_date_args(sp):
        sp.add_argument(
            "--start",
            required=True,
            type=_parse_date,
            help="Start date DDMMYY (e.g., 040325 for March 4, 2025)",
        )
        sp.add_argument(
            "--end",
            required=True,
            type=_parse_date,
            help="End date DDMMYY (e.g., 040325 for March 4, 2025)",
        )
        sp.add_argument(
            "--tickers",
            nargs="+",
            help="Optional explicit tickers (default: read tickers.json)",
        )
        sp.add_argument(
            "--yes",
            action="store_true",
            help="Auto-confirm fallback to first 5 tickers when --tickers is omitted",
        )
        sp.add_argument(
            "--interval", type=int, default=1, help="Bar interval minutes (default: 1)"
        )
        sp.add_argument(
            "--dry-run",
            action="store_true",
            help="Fetch & parse but do not insert (where applicable)",
        )
        sp.add_argument(
            "--output-filename",
            type=str,
            default=None,
            help="Output CSV filename (default: None)",
        )
        sp.add_argument(
            "--log-level",
            type=str,
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging level (default: INFO)",
        )
        sp.add_argument(
            "--on-conflict",
            choices=["ignore", "fail"],
            default="fail",
            help="Conflict handling (requires unique index if 'ignore')",
        )
        sp.set_defaults(func=lambda args: None)  # default no-op

    # specific
    sp_specific = sub.add_parser(
        "specific", help="Backfill continuous date range for tickers and insert into DB"
    )
    add_date_args(sp_specific)
    sp_specific.set_defaults(func=cmd_specific)

    # concurrent
    sp_conc = sub.add_parser("concurrent", help="Concurrent multi-ticker backfill")
    add_date_args(sp_conc)
    sp_conc.add_argument(
        "--threads",
        type=int,
        default=6,
        help="Max worker threads (cap to DB pool size)",
    )
    sp_conc.set_defaults(func=cmd_concurrent)

    # inject-csv
    sp_csv = sub.add_parser(
        "inject-csv", help="Inject previously downloaded CSV dumps into DB"
    )
    sp_csv.add_argument(
        "--csv-dir", required=True, help="Directory containing CSV dumps"
    )
    sp_csv.add_argument(
        "--threads", type=int, default=5, help="Worker threads for CSV ingestion"
    )
    sp_csv.set_defaults(func=cmd_inject_csv)

    return p


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    log_level = getattr(args, "log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    logger.debug("Parsed arguments: %s", args)
    if hasattr(args, "func") and args.func:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
