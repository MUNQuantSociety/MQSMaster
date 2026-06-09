import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Setup path for imports
SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from orchestrator.backfill.concurrent_backfill import concurrent_backfill
    from orchestrator.marketData.fmpMarketData import FMPMarketData
except Exception as e:
    logging.error(f"Failed to import required modules: {e}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
EXTRA_TICKERS_DIR = SCRIPT_DIR / "extra_tickers"
TICKERS_FILE = EXTRA_TICKERS_DIR / "nasdaq_tickers.json"


def load_existing_tickers(tickers_path: Path) -> list[str]:
    """Load existing tickers from JSON file."""
    try:
        if tickers_path.exists():
            with open(tickers_path, "r") as f:
                tickers = json.load(f)
                logger.info(
                    f"Loaded {len(tickers)} existing tickers from {tickers_path.name}"
                )
                return tickers
        else:
            logger.warning(f"Tickers file not found at {tickers_path}, starting fresh")
            return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {tickers_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading tickers: {e}")
        return []


def save_tickers(tickers: list[str], tickers_path: Path) -> None:
    """Save tickers to JSON file + append a dated entry to <name>_history.json.

    The history sidecar is a minimum-viable PIT trail (D2 Tier 1, 2026-05-20).
    Each entry records (date, set of tickers); a follow-up PR can promote this
    into a proper index_membership DB table. The sidecar is append-only; old
    entries are never removed.
    """
    try:
        tickers_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tickers_path, "w") as f:
            json.dump(tickers, f, indent=2)
        logger.info(f"Saved {len(tickers)} tickers to {tickers_path.name}")

        # D2 Tier 1 history sidecar.
        from datetime import datetime as _dt
        history_path = tickers_path.with_name(tickers_path.stem + "_history.json")
        history: list = []
        if history_path.exists():
            try:
                with open(history_path, "r") as hf:
                    history = json.load(hf) or []
                if not isinstance(history, list):
                    history = []
            except (OSError, json.JSONDecodeError):
                history = []
        history.append({
            "date": _dt.utcnow().strftime("%Y-%m-%d"),
            "tickers": sorted(tickers),
            "n": len(tickers),
        })
        try:
            with open(history_path, "w") as hf:
                json.dump(history, hf, indent=2)
        except OSError as e:
            logger.warning(f"Could not write history sidecar {history_path.name}: {e}")
    except Exception as e:
        logger.error(f"Error saving tickers: {e}")
        raise


def fetch_and_merge_tickers(
    fmp: FMPMarketData, existing_tickers: list[str]
) -> list[str]:
    """Fetch new tickers and merge with existing ones."""
    try:
        # Fetch S&P 500 tickers
        sp500 = fmp.get_sp500_tickers()
        logger.info(f"Fetched {len(sp500)} S&P 500 tickers from FMP")

        # Fetch commodity tickers
        commodity_tickers = fmp.get_commodity_tickers()
        logger.info(f"Fetched {len(commodity_tickers)} commodity tickers from FMP")

        crypto = fmp.get_crypto_tickers()
        logger.info(f"Fetched {len(crypto)} crypto tickers from FMP")

        new_tickers = sp500 + commodity_tickers + crypto
        logger.info(f"Total new tickers fetched: {len(new_tickers)}")
        # Merge and deduplicate using set operations (much faster than pandas)
        combined = list(set(existing_tickers) | set(new_tickers))
        combined.sort()  # Keep consistent ordering

        logger.info(
            f"Total unique tickers after merge: {len(combined)} "
            f"(+{len(combined) - len(existing_tickers)} new)"
        )

        return combined

    except Exception as e:
        logger.error(f"Error fetching tickers: {e}")
        raise


def parse_cli_arguments() -> argparse.Namespace:
    """Parse command-line arguments for backfill configuration."""
    parser = argparse.ArgumentParser(
        description="Refresh tickers and optionally backfill market data"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date in DDMMYY format (default: 30 days ago)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date in DDMMYY format (default: today)",
    )
    parser.add_argument(
        "--interval", type=int, default=1, help="Bar interval in minutes (default: 1)"
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="Number of worker threads (default: 4, must be < MQSDBConnector pool maxconn=6)"
    )
    parser.add_argument(
        "--exchange", type=str, default="NYSE", help="Exchange code (default: NYSE)"
    )
    parser.add_argument(
        "--on-conflict",
        choices=["ignore", "fail"],
        default="ignore",
        help="How to handle duplicate data (default: ignore)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data but don't insert into database",
    )
    parser.add_argument(
        "--skip-backfill",
        action="store_true",
        help="Only update tickers, skip backfill process",
    )

    return parser.parse_args()


def parse_date_or_exit(value: str, arg_name: str):
    """Parse a DDMMYY date argument or exit with a clear message."""
    try:
        return datetime.strptime(value, "%d%m%y").date()
    except ValueError:
        logger.error(f"Invalid {arg_name} date '{value}'. Expected format: DDMMYY")
        raise SystemExit(2)


def main():
    """Main execution function."""
    # Parse arguments early
    args = parse_cli_arguments()

    # Get tickers file path
    tickers_path = TICKERS_FILE

    # Load existing tickers
    existing_tickers = load_existing_tickers(tickers_path)

    # Fetch and merge tickers
    fmp = FMPMarketData()
    combined_tickers = fetch_and_merge_tickers(fmp, existing_tickers)

    # Save updated tickers
    save_tickers(combined_tickers, tickers_path)

    # Display sample of tickers
    logger.info(f"Sample tickers: {combined_tickers[:10]}")
    logger.info("✓ Updated tickers.json successfully")

    # Skip backfill if requested
    if args.skip_backfill:
        logger.info("Skipping backfill (--skip-backfill flag set)")
        return

    # Parse dates for backfill
    if args.end:
        end_date = parse_date_or_exit(args.end, "--end")
    else:
        end_date = datetime.now().date()

    if args.start:
        start_date = parse_date_or_exit(args.start, "--start")
    else:
        start_date = end_date - timedelta(days=30)

    if start_date > end_date:
        logger.error(
            f"Invalid date range: start date {start_date} is after end date {end_date}"
        )
        raise SystemExit(2)

    # Display backfill configuration
    print(f"\n{'=' * 60}")
    print("Concurrent Backfill Configuration:")
    print(f"  Tickers: {len(combined_tickers)}")
    print(f"  Date Range: {start_date} to {end_date}")
    print(f"  Interval: {args.interval} min")
    print(f"  Threads: {args.threads}")
    print(f"  Exchange: {args.exchange}")
    print(f"  On Conflict: {args.on_conflict}")
    print(f"  Dry Run: {args.dry_run}")
    print(f"{'=' * 60}\n")

    # Run concurrent backfill with COMBINED tickers (not just new ones!)
    try:
        concurrent_backfill(
            tickers=combined_tickers,  # FIXED: was using df_tickers instead
            start_date=start_date,
            end_date=end_date,
            interval=args.interval,
            exchange=args.exchange.lower(),
            dry_run=args.dry_run,
            on_conflict=args.on_conflict,
            threads=args.threads,
        )
        logger.info("✓ Backfill completed successfully")
    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        raise


if __name__ == "__main__":
    main()
