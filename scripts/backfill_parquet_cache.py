"""
Backfills historical daily OHLCV directly into local parquet cache using FMP

HOW TO USE:
    - TICKERS{} -> Add all tickers you want to backfill.
    - REPLACE_ENTIRELY{} -> Add any tickers that were filled with different time spacing or something
        that you want to delete and overwrite.
    - DAILY_TICKERS{} -> some tickers only have daily values in FMP (e.g., VIX before a certain date, GLD). 
        Optimizer will NOT work for a daily-only ticker if not included in this dict.

Usage:
    python -m scripts.backfill_parquet_cache
    python -m scripts.backfill_parquet_cache --tickers TSLA AMZN --overwrite TSLA --interval 30 --chunk-size 30
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from src.backtest.data.backfill_cache import cache as _cache
from src.orchestrator.marketData.fmpMarketData import FMPMarketData

CACHE_DIR = Path("src/backtest/data/backfill_cache")

FROM_DATE = "2010-01-01"
TO_DATE   = "2025-12-31"

TICKERS = [
    "TSLA", "AMZN", "MSFT", "NVDA",
    "JPM",  "XOM",  "UNH",  "CAT",  "WMT",
    "TLT",  "GLD", "SPY", "^VIX"
]

# Delete and fully rebuild — switching to intraday means all existing daily parquets
# are incompatible and must be replaced
REPLACE_ENTIRELY = {"TSLA", "AMZN", "MSFT", "NVDA", "JPM", "XOM",
                    "UNH", "CAT", "WMT", "TLT", "GLD", "SPY", "^VIX"}


INTERVAL    = 30          # bar size in minutes
CHUNK_DAYS  = 30          # days per FMP API call (intraday endpoint truncates large ranges)


def _parse_rows(records: list, ticker: str) -> list:
    rows = []
    for r in records:
        date_str = r.get("date")
        if not date_str:
            continue
        try:
            ts = pd.Timestamp(date_str).tz_localize(
                "America/New_York", ambiguous="NaT", nonexistent="NaT"
            )
        except Exception:
            continue
        if pd.isna(ts):
            continue
        rows.append({
            "ticker":      ticker,
            "timestamp":   ts,
            "open_price":  r.get("open"),
            "high_price":  r.get("high"),
            "low_price":   r.get("low"),
            "close_price": r.get("close"),
            "volume":      r.get("volume") or 0,
        })
    return rows


# VIX has no intraday history on FMP before 2023 — fetch daily instead so
# regime detection works across the full 2015-2025 training window.
DAILY_TICKERS = {"^VIX", "GLD"}


def _fetch_daily(fmp: FMPMarketData, ticker: str) -> list:
    """Fetch daily OHLCV from FMP, stamp each row at 15:59 ET."""
    print(f"\n[{ticker}] Fetching daily bars {FROM_DATE} → {TO_DATE} ...")
    records = fmp.get_historical_data(ticker, FROM_DATE, TO_DATE)
    if not records:
        return []
    rows = []
    for r in records:
        date_str = r.get("date")
        if not date_str:
            continue
        try:
            ts = pd.Timestamp(f"{date_str} 15:59:00").tz_localize(
                "America/New_York", ambiguous="NaT", nonexistent="NaT"
            )
        except Exception:
            continue
        if pd.isna(ts):
            continue
        rows.append({
            "ticker":      ticker,
            "timestamp":   ts,
            "open_price":  r.get("open"),
            "high_price":  r.get("high"),
            "low_price":   r.get("low"),
            "close_price": r.get("close"),
            "volume":      r.get("volume") or 0,
        })
    return rows


def _fetch_intraday(fmp: FMPMarketData, ticker: str) -> list:
    """Fetch intraday OHLCV from FMP in 30-day chunks."""
    start = pd.Timestamp(FROM_DATE)
    end   = pd.Timestamp(TO_DATE)
    chunk = pd.Timedelta(days=CHUNK_DAYS)
    n_chunks = max(1, int((end - start).days / CHUNK_DAYS) + 1)
    print(f"\n[{ticker}] Fetching {INTERVAL}-min bars {FROM_DATE} → {TO_DATE} "
          f"({n_chunks} chunks) ...")
    all_rows = []
    current  = start
    while current <= end:
        chunk_end = min(current + chunk, end)
        records   = fmp.get_intraday_data(
            ticker,
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
            INTERVAL,
        )
        count = len(records) if records else 0
        if records:
            all_rows.extend(_parse_rows(records, ticker))
        print(f"  {current.date()} → {chunk_end.date()}: {count} bars")
        current = chunk_end + pd.Timedelta(days=1)
    return all_rows


def fetch_and_cache(fmp: FMPMarketData, ticker: str) -> int:
    """Fetch full history from FMP and merge into parquet."""
    if ticker in DAILY_TICKERS:
        all_rows = _fetch_daily(fmp, ticker)
    else:
        all_rows = _fetch_intraday(fmp, ticker)

    if not all_rows:
        print(f"[{ticker}] No valid rows after processing.")
        return 0

    new_df = pd.DataFrame(all_rows)

    if ticker in REPLACE_ENTIRELY:
        safe = ticker.replace("^", "_").replace("/", "_")
        parquet_path = CACHE_DIR / f"{safe}.parquet"
        if parquet_path.exists():
            parquet_path.unlink()
            print(f"[{ticker}] Deleted existing parquet.")
        cached = pd.DataFrame()
    else:
        cached = _cache.load(ticker)

    merged = _cache.merge_and_save(ticker, cached, new_df)
    print(f"[{ticker}] {len(new_df)} rows fetched → parquet now has {len(merged)} rows")
    return len(new_df)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill historical OHLCV into local parquet cache from FMP."
    )
    parser.add_argument(
        "--tickers", nargs="+", default=TICKERS,
        help=f"Tickers to backfill (space-separated). Default: {TICKERS}",
    )
    parser.add_argument(
        "--overwrite", nargs="+", default=sorted(REPLACE_ENTIRELY),
        help="Tickers to delete and fully rebuild instead of merging. "
             f"Default: {sorted(REPLACE_ENTIRELY)}",
    )
    parser.add_argument(
        "--interval", type=int, default=INTERVAL,
        help=f"Intraday bar size in minutes (1, 5, 15, 30, 60). Default: {INTERVAL}",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_DAYS,
        help=f"Days per FMP intraday API call. Default: {CHUNK_DAYS}",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    global TICKERS, REPLACE_ENTIRELY, INTERVAL, CHUNK_DAYS
    TICKERS = args.tickers
    REPLACE_ENTIRELY = set(args.overwrite)
    INTERVAL = args.interval
    CHUNK_DAYS = args.chunk_size

    fmp = FMPMarketData()
    total_rows = 0
    failed = []

    run_start = time.perf_counter()
    for ticker in TICKERS:
        try:
            count = fetch_and_cache(fmp, ticker)
            total_rows += count
            if count == 0:
                failed.append(ticker)
        except Exception as e:
            print(f"[{ticker}] ERROR: {e}")
            failed.append(ticker)
    run_seconds = time.perf_counter() - run_start

    stats = fmp.get_transfer_stats()
    print(f"\n{'='*50}")
    print(f"Done. {total_rows} total rows written across {len(TICKERS)} tickers.")
    print(f"Total time: {run_seconds:.1f}s over {stats['request_count']} FMP requests")
    print(f"Downloaded: {stats['total_bytes'] / (1024 * 1024):.2f} MB "
          f"(avg {stats['avg_mb_per_sec']:.3f} MB/s while requests were in flight)")
    if failed:
        print(f"Failed or empty: {failed}")
        print("For ^VIX, try changing the ticker to '%5EVIX' if FMP returned no data.")


if __name__ == "__main__":
    main()
