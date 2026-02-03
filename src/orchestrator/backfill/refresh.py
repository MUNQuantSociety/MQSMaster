import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Setup path for imports
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from orchestrator.marketData.fmpMarketData import FMPMarketData
    from orchestrator.backfill.concurrent_backfill import concurrent_backfill
except Exception as e:
    logging.error(f"Failed to import required modules: {e}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    """Save tickers to JSON file."""
    try:
        with open(tickers_path, "w") as f:
            json.dump(tickers, f, indent=2)
        logger.info(f"Saved {len(tickers)} tickers to {tickers_path.name}")
    except Exception as e:
        logger.error(f"Error saving tickers: {e}")
        raise


def fetch_and_merge_tickers(
    fmp: FMPMarketData, existing_tickers: list[str]
) -> list[str]:
    """Fetch new tickers and merge with existing ones."""
    try:
        # Fetch S&P 500 tickers
        sp500 = fmp.get_SP500_tickers()
        logger.info(f"Fetched {len(sp500)} S&P 500 tickers from FMP")

        # TODO: Uncomment when crypto support is needed
        # crypto = fmp.get_crypto_tickers()
        # logger.info(f"Fetched {len(crypto)} crypto tickers from FMP")
        # new_tickers = sp500 + crypto

        new_tickers = sp500

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
        "--threads", type=int, default=8, help="Number of worker threads (default: 8)"
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


def main():
    """Main execution function."""
    # Parse arguments early
    args = parse_cli_arguments()

    # Get tickers file path
    tickers_path = Path(__file__).parent / "tickers.json"

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
        end_date = datetime.strptime(args.end, "%d%m%y").date()
    else:
        end_date = datetime.now().date()

    if args.start:
        start_date = datetime.strptime(args.start, "%d%m%y").date()
    else:
        start_date = end_date - timedelta(days=30)

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
