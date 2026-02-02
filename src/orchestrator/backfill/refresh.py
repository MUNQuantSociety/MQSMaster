import logging
import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from orchestrator.marketData.fmpMarketData import FMPMarketData
except Exception as e:
    logging.error(f"Failed to import APIAuth after adding src to sys.path. {e}")
    raise


def main():
    # fetch current SP500 tickers from FMP
    fmp = FMPMarketData()
    sp500 = fmp.get_SP500_tickers()
    print(f"Fetched {len(sp500)} S&P 500 tickers.")
<<<<<<< Updated upstream
    # get path of tickers.json
    tickers_path = Path(__file__).parent / "tickers.json"

=======
    #get path of tickers.json
    tickers_path = Path(__file__).parent / "tickers.json"
    
>>>>>>> Stashed changes
    # fetch crypto tickers from FMP
    # crypto = fmp.get_crypto_tickers()
    # print(f"Fetched {len(crypto)} crypto tickers.")

    # combine tickers
    all_tickers = sp500  # + crypto
    print(f"Combined total tickers: {len(all_tickers)}")
    df_tickers = pd.DataFrame(all_tickers)

    # get existing tickers from tickers.json
    df_existing = pd.read_json(tickers_path)
    print(f"Existing tickers in tickers.json: {len(df_existing)}")
    # merge and deduplicate
    df_combined = (
        pd.concat([df_existing, df_tickers]).drop_duplicates().reset_index(drop=True)
    )
    print(f"Total unique tickers after merge: {len(df_combined)}")
<<<<<<< Updated upstream
    df_combined.to_json(tickers_path, orient="records", indent=2)
=======
    df_combined.to_json(
        tickers_path, orient="records", indent=2
    )
>>>>>>> Stashed changes
    print(df_combined.head())
    print("✓ Updated tickers.json successfully.")
    import argparse
    from datetime import datetime, timedelta

    from orchestrator.backfill.concurrent_backfill import concurrent_backfill

    # Parse optional CLI arguments for backfill parameters
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--start", type=str, default=None, help="Start date DDMMYY")
    parser.add_argument("--end", type=str, default=None, help="End date DDMMYY")
    parser.add_argument(
        "--interval", type=int, default=1, help="Bar interval in minutes"
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="Number of worker threads"
    )
    parser.add_argument("--exchange", type=str, default="NYSE", help="Exchange code")
    parser.add_argument("--on-conflict", choices=["ignore", "fail"], default="ignore")
    parser.add_argument("--dry-run", action="store_true", help="Fetch but don't insert")

    args, unknown = parser.parse_known_args()

    # Parse dates or use defaults (last 30 days)
    if args.end:
        end_date = datetime.strptime(args.end, "%d%m%y").date()
    else:
        end_date = datetime.now().date()

    if args.start:
        start_date = datetime.strptime(args.start, "%d%m%y").date()
    else:
        start_date = end_date - timedelta(days=30)

    # Extract tickers from dataframe
    tickers = (
        df_tickers[0].tolist()
        if 0 in df_tickers.columns
        else df_tickers.iloc[:, 0].tolist()
    )

    print(f"\n{'=' * 60}")
    print("Concurrent Backfill Configuration:")
    print(f"  Tickers: {len(tickers)}")
    print(f"  Date Range: {start_date} to {end_date}")
    print(f"  Interval: {args.interval} min")
    print(f"  Threads: {args.threads}")
    print(f"  Exchange: {args.exchange}")
    print(f"  Dry Run: {args.dry_run}")
    print(f"{'=' * 60}\n")

    # Run concurrent backfill
    concurrent_backfill(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        interval=args.interval,
        exchange=args.exchange.lower(),
        dry_run=args.dry_run,
        on_conflict=args.on_conflict,
        threads=args.threads,
    )

    print("✓ Updated tickers successfully.")


if __name__ == "__main__":
    main()
