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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from data_infra.marketData.fmpMarketData import FMPMarketData

# Batch size: number of days per API call (e.g., 2 means requesting 2 days at once)
BATCH_DAYS = 2

# Directory for storing temporary CSVs
TEMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/backfill_cache"))
os.makedirs(TEMP_DIR, exist_ok=True)  # Ensure the directory exists

def backfill_data(
    tickers, 
    start_date, 
    end_date, 
    interval, 
    exchange=None,
    output_filename="backfilled_data.csv"
):
    """
    Pulls intraday data from FMP for each ticker from start_date to end_date, 
    writes data incrementally to a CSV to prevent memory issues.

    :param tickers: list of ticker symbols
    :param start_date: 'YYYY-MM-DD' or date object
    :param end_date: 'YYYY-MM-DD' or date object
    :param interval: integer (1,5,15,30,60) indicating the bar size
    :param exchange: optional string, e.g. 'NYSE', 'NASDAQ'; not used in logic yet
    :param output_filename: Name of the final output CSV file
    """

    # Convert start/end to date objects if strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    print(f"Starting backfill for {tickers} from {start_date} to {end_date}, interval={interval}min, exchange={exchange}")

    # Prepare an instance of FMPMarketData (with rate limiter)
    fmp = FMPMarketData()

    # Generate a list of business days in the range (excludes weekends & holidays)
    all_dates = pd.bdate_range(start=start_date, end=end_date).date
    if len(all_dates) == 0:
        print("[Backfill] No valid trading days in range.")
        return

    # Group business days into batches of BATCH_DAYS
    grouped_dates = [all_dates[i : i + BATCH_DAYS] for i in range(0, len(all_dates), BATCH_DAYS)]

    # Full path to output CSV
    output_path = os.path.join(TEMP_DIR, output_filename)

    # Ensure the file is empty before appending data
    if os.path.exists(output_path):
        os.remove(output_path)

    for ticker in tickers:
        print(f"[Backfill] Processing {ticker} ...")
        for date_group in grouped_dates:
            from_date = date_group[0].strftime("%Y-%m-%d")
            to_date = date_group[-1].strftime("%Y-%m-%d")

            print(f"  Fetching {ticker} from {from_date} to {to_date}")

            # Attempt up to 2 tries per batch
            attempt = 0
            success = False
            while attempt < 2 and not success:
                attempt += 1
                try:
                    data_chunk = fmp.get_intraday_data(
                        tickers=ticker, 
                        from_date=from_date, 
                        to_date=to_date, 
                        interval=interval
                    )

                    if data_chunk and isinstance(data_chunk, list):
                        df_chunk = pd.DataFrame(data_chunk)
                        
                        if not df_chunk.empty:
                            # Ensure proper column names
                            df_chunk["ticker"] = ticker

                            # Standardize datetime format
                            if "date" in df_chunk.columns:
                                df_chunk.rename(columns={"date": "datetime"}, inplace=True)
                                df_chunk["datetime"] = pd.to_datetime(df_chunk["datetime"])
                            
                            # Create a separate "date" column
                            df_chunk["date"] = df_chunk["datetime"].dt.date

                            # Ensure column order: Ticker → Date → Datetime → OHLCV
                            column_order = ["ticker", "date", "datetime", "open", "high", "low", "close", "volume"]
                            df_chunk = df_chunk[column_order]

                            # Append to CSV instead of keeping data in RAM
                            df_chunk.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path))

                    else:
                        print(f"[Backfill] No data returned for {ticker} from {from_date} to {to_date} (possible holiday or no trades).")
                    success = True

                except Exception as ex:
                    print(f"[Backfill:ERROR] Attempt {attempt} error fetching {ticker} from {from_date} to {to_date}: {ex}")
                    time.sleep(1)  # Small wait before retry

    print(f"[Backfill] Completed. Data saved to {output_path}")
