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
from ...orchestrator.marketData.fmpMarketData import FMPMarketData

# Batch size: number of days per API call (e.g., 2 means requesting 2 days at once)
BATCH_DAYS = 3

# Directory for storing temporary CSVs
TEMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/backfill_cache"))

os.makedirs(TEMP_DIR, exist_ok=True)  # Ensure the directory exists


def convert_to_date(date_value):
    """Converts a string or datetime to a date object."""
    if isinstance(date_value, str):
        return datetime.strptime(date_value, "%Y-%m-%d").date()
    return date_value


def generate_output_filename(tickers, start_date, end_date, interval, exchange, output_filename):
    """
    Generates a dynamic output filename if:
    - `output_filename` is None, OR
    - `output_filename` is the default 'backfilled_data.csv'.

    Format: backfill_{tickerOrMultiple}_{YYYYMMDD_YYYYMMDD}_{interval}min_{exchange}.csv
    """
    if not output_filename or output_filename == "backfilled_data.csv":
        ticker_str = tickers[0] if len(tickers) == 1 else "multiple_tickers"
        date_range_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        interval_str = f"{interval}min"
        exchange_str = exchange.lower() if exchange else "nasdaq"
        output_filename = f"backfill_{ticker_str}_{date_range_str}_{interval_str}_{exchange_str}.csv"
    
    return os.path.join(TEMP_DIR, output_filename)


def prepare_data(df_chunk, ticker):
    """
    Prepares data by renaming columns, adding date column, and ensuring correct order.
    The function returns a DataFrame with columns:
       ['ticker', 'date', 'datetime', 'open', 'high', 'low', 'close', 'volume'].
    """
    if "date" in df_chunk.columns:
        # rename 'date' → 'datetime'
        df_chunk.rename(columns={"date": "datetime"}, inplace=True)
        df_chunk["datetime"] = pd.to_datetime(df_chunk["datetime"])
    
    df_chunk["ticker"] = ticker
    df_chunk["date"] = df_chunk["datetime"].dt.date

    column_order = ["ticker", "date", "datetime", "open", "high", "low", "close", "volume"]
    df_chunk = df_chunk[column_order]
    return df_chunk


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
    writes data incrementally to a CSV to prevent memory issues, and
    returns the combined data as a DataFrame if output_filename=None.

    :param tickers: list of ticker symbols
    :param start_date: 'YYYY-MM-DD' or date object
    :param end_date: 'YYYY-MM-DD' or date object
    :param interval: integer (1, 5, 15, 30, 60) indicating bar size
    :param exchange: optional string, e.g., 'NYSE', 'NASDAQ'
    :param output_filename: Name of the final output CSV file (None => skip writing)
    :return: Combined DataFrame of all fetched data if output_filename=None, else None
    """

    # 1) Convert start_date and end_date to date objects if needed
    start_date = convert_to_date(start_date)
    end_date = convert_to_date(end_date)

    print(f"Starting backfill for {tickers} from {start_date} to {end_date}, "
          f"interval={interval}min, exchange={exchange}")

    # 2) Prepare an instance of FMPMarketData
    fmp = FMPMarketData()

    # 3) Generate a list of business days (excludes weekends & holidays)
    all_dates = pd.bdate_range(start=start_date, end=end_date).date
    if len(all_dates) == 0:
        print("[Backfill] No valid trading days in the specified range.")
        return pd.DataFrame() if output_filename is None else None

    # 4) Group business days into batches
    grouped_dates = [all_dates[i : i + BATCH_DAYS] for i in range(0, len(all_dates), BATCH_DAYS)]

    # 5) Generate the output path if output_filename is provided
    #    (or auto-generate one if it's the default)
    output_path = None
    if output_filename is not None:
        output_path = generate_output_filename(tickers, start_date, end_date, interval, exchange, output_filename)
        # Ensure the file is empty before appending data
        if os.path.exists(output_path):
            os.remove(output_path)

    # 6) We'll store all data in-memory so we can return it if needed
    #    or so we can insert it to DB
    all_dfs = []

    # 7) Loop over tickers and date batches
    for ticker in tickers:
        print(f"[Backfill] Processing {ticker}...")
        for date_group in grouped_dates:
            from_date_str = date_group[0].strftime("%Y-%m-%d")
            to_date_str = date_group[-1].strftime("%Y-%m-%d")

            print(f"  Fetching {ticker} from {from_date_str} to {to_date_str}...")
            # Attempt up to 2 tries per batch
            attempt = 0
            success = False
            while attempt < 2 and not success:
                attempt += 1
                try:
                    data_chunk = fmp.get_intraday_data(
                        tickers=ticker,
                        from_date=from_date_str,
                        to_date=to_date_str,
                        interval=interval,
                    )

                    # If data is valid
                    if data_chunk and isinstance(data_chunk, list):
                        df_chunk = pd.DataFrame(data_chunk)
                        if not df_chunk.empty:
                            df_chunk = prepare_data(df_chunk, ticker)
                            # Always keep a local copy in-memory
                            all_dfs.append(df_chunk)

                            # If output_path is set, write to CSV
                            if output_path:
                                df_chunk.to_csv(
                                    output_path,
                                    mode="a",
                                    index=False,
                                    header=(not os.path.exists(output_path)),
                                )
                        else:
                            print(f"  [Backfill] No data returned for {ticker} "
                                  f"from {from_date_str} to {to_date_str}.")
                    success = True

                except Exception as ex:
                    print(f"[Backfill:ERROR] Attempt {attempt} error fetching {ticker} "
                          f"from {from_date_str} to {to_date_str}: {ex}")
                    time.sleep(1)  # small wait before retry

    # 8) Print completion message
    if output_path:
        print(f"[Backfill] Completed. Data saved to {output_path}")
    else:
        print(f"[Backfill] Completed. No CSV file generated (output_filename=None).")

    # 9) If output_filename=None, return the combined in-memory DataFrame
    if output_filename is None:
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    # Otherwise, no need to return anything
    return None
