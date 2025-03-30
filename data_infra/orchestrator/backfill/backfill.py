"""
backfill.py
-----------
Handles large-scale intraday data backfilling from FMP.
Uses caching (temporary CSV storage) to avoid RAM overflow.
"""

import sys
import os
import time
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to sys.path to import FMPMarketData
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from data_infra.marketData.fmpMarketData import FMPMarketData

# Batch size: number of days per API call (e.g., 2 means requesting 2 days at once)
BATCH_DAYS = 2

# Directory for storing temporary CSVs
TEMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/backfill_cache"))
os.makedirs(TEMP_DIR, exist_ok=True)  # Ensure the directory exists


def convert_to_date(date_value):
    """Converts a string or datetime to a date object."""
    if isinstance(date_value, str):
        return datetime.strptime(date_value, "%Y-%m-%d").date()
    return date_value


def generate_output_filename(tickers, start_date, end_date, interval, exchange, output_filename):
    """Generates a dynamic output filename if none is provided."""
    if not output_filename or output_filename == "backfilled_data.csv":
        ticker_str = tickers[0] if len(tickers) == 1 else "multiple_tickers"
        date_range_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        interval_str = f"{interval}min"
        exchange_str = exchange.lower() if exchange else "nasdaq"
        # Format: backfill_{ticker}_{date_range}_{interval}_{exchange}.csv
        output_filename = f"backfill_{ticker_str}_{date_range_str}_{interval_str}_{exchange_str}.csv"
    
    return os.path.join(TEMP_DIR, output_filename)


def prepare_data(df_chunk, ticker):
    """Prepares data by renaming columns, adding date column, and ensuring correct order."""
    if "date" in df_chunk.columns:
        df_chunk.rename(columns={"date": "datetime"}, inplace=True)
        df_chunk["datetime"] = pd.to_datetime(df_chunk["datetime"])
    
    df_chunk["ticker"] = ticker
    df_chunk["date"] = df_chunk["datetime"].dt.date

    # Ensure column order: Ticker → Date → Datetime → OHLCV
    column_order = ["ticker", "date", "datetime", "open", "high", "low", "close", "volume"]
    return df_chunk[column_order]


def fetch_and_process_data(fmp, ticker, from_date, to_date, interval, output_path):
    """Fetches intraday data for a given ticker and date range, and writes to CSV if applicable."""
    attempt, success = 0, False
    while attempt < 2 and not success:
        attempt += 1
        try:
            data_chunk = fmp.get_intraday_data(
                tickers=ticker,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
            )

            if data_chunk and isinstance(data_chunk, list):
                df_chunk = pd.DataFrame(data_chunk)

                if not df_chunk.empty:
                    df_chunk = prepare_data(df_chunk, ticker)

                    # Write to CSV if output_path is valid
                    if output_path:
                        df_chunk.to_csv(
                            output_path, mode="a", index=False, header=not os.path.exists(output_path)
                        )
                    else:
                        print(f"[Backfill] Skipping file write for {ticker} as output_filename is None.")
                else:
                    print(f"[Backfill] No data for {ticker} from {from_date} to {to_date}. Possible holiday or no trades.")
            success = True

        except Exception as ex:
            print(f"[Backfill:ERROR] Attempt {attempt} error fetching {ticker} from {from_date} to {to_date}: {ex}")
            time.sleep(1)  # Small wait before retry


def backfill_data(
    tickers,
    start_date,
    end_date,
    interval,
    exchange=None,
    output_filename="backfilled_data.csv",
):
    """
    Pulls intraday data from FMP for each ticker from start_date to end_date,
    and writes data incrementally to a CSV to prevent memory issues.

    :param tickers: list of ticker symbols
    :param start_date: 'YYYY-MM-DD' or date object
    :param end_date: 'YYYY-MM-DD' or date object
    :param interval: integer (1, 5, 15, 30, 60) indicating the bar size
    :param exchange: optional string, e.g., 'NYSE', 'NASDAQ'
    :param output_filename: Name of the final output CSV file
    """

    # Convert start_date and end_date to date objects if needed
    start_date = convert_to_date(start_date)
    end_date = convert_to_date(end_date)

    print(f"Starting backfill for {tickers} from {start_date} to {end_date}, interval={interval}min, exchange={exchange}")

    # Prepare an instance of FMPMarketData
    fmp = FMPMarketData()

    # Generate a list of business days in the range (excludes weekends & holidays)
    all_dates = pd.bdate_range(start=start_date, end=end_date).date
    if len(all_dates) == 0:
        print("[Backfill] No valid trading days in the specified range.")
        return

    # Group business days into batches of BATCH_DAYS
    grouped_dates = [all_dates[i : i + BATCH_DAYS] for i in range(0, len(all_dates), BATCH_DAYS)]

    # Generate output filename dynamically if not provided
    output_path = generate_output_filename(tickers, start_date, end_date, interval, exchange, output_filename)

    # Ensure the file is empty before appending data if output_path is provided
    if output_path and os.path.exists(output_path):
        os.remove(output_path)

    # Process data for each ticker
    for ticker in tickers:
        print(f"[Backfill] Processing {ticker} ...")
        for date_group in grouped_dates:
            from_date = date_group[0].strftime("%Y-%m-%d")
            to_date = date_group[-1].strftime("%Y-%m-%d")

            print(f"  Fetching {ticker} from {from_date} to {to_date}")
            fetch_and_process_data(fmp, ticker, from_date, to_date, interval, output_path)

    # Completion message based on whether file writing occurred
    if output_path:
        print(f"[Backfill] Completed. Data saved to {output_path}")
    else:
        print(f"[Backfill] Completed. No CSV file generated.")
