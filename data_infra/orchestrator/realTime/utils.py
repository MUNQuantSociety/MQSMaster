import os
import json
import logging
import concurrent.futures
from datetime import datetime

def load_tickers(portfolios_base_path="../../portfolios"):
    """
    Load ticker symbols from config.json files located in portfolio directories.
    
    Args:
        portfolios_base_path (str): Relative path where the portfolios are located.
    
    Returns:
        list: A list of unique ticker symbols.
    """
    portfolios_dir = _get_portfolios_dir(portfolios_base_path)
    if not os.path.isdir(portfolios_dir):
        logging.error(f"Directory '{portfolios_dir}' does not exist.")
        return []

    tickers = set()

    portfolio_folders = _get_portfolio_folders(portfolios_dir)
    for folder_path in portfolio_folders:
        config_path = os.path.join(folder_path, "config.json")
        tickers_in_config = _read_config_file(config_path)
        tickers.update(tickers_in_config)

    return list(tickers)


def _get_portfolios_dir(portfolios_base_path):
    """
    Returns the absolute path for the portfolios directory.
    """
    return os.path.abspath(os.path.join(os.getcwd(), portfolios_base_path))


def _get_portfolio_folders(portfolios_dir):
    """
    Finds folders in portfolios_dir that start with 'portfolio_'.
    """
    folders = []
    for folder in os.listdir(portfolios_dir):
        folder_path = os.path.join(portfolios_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("portfolio_"):
            folders.append(folder_path)
    return folders


def _read_config_file(config_path):
    """
    Reads the config.json file if it exists and returns the 'TICKERS' list.
    """
    if not os.path.isfile(config_path):
        return []

    try:
        with open(config_path, "r") as file:
            config_data = json.load(file)
            if "TICKERS" in config_data:
                return config_data["TICKERS"]
            else:
                logging.warning(f"No 'TICKERS' key found in {config_path}")
                return []
    except (json.JSONDecodeError, OSError) as e:
        logging.error(f"Error reading {config_path}: {e}")
        return []


def collect_market_data(fmp, tickers, last_timestamps, max_workers=5):
    """
    Concurrently fetch real-time data for all tickers. 
    Only returns rows that are newer than the last known timestamp for each ticker.

    Args:
        fmp (FMPMarketData): An instance of the market data client.
        tickers (list): A list of ticker symbols to fetch data for.
        last_timestamps (dict): Dictionary tracking the last known timestamps for each ticker.
        max_workers (int): Maximum number of worker threads.

    Returns:
        list: A list of newly fetched data rows (dictionaries) from all tickers.
    """
    new_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(fetch_new_data, ticker, fmp, last_timestamps): ticker
            for ticker in tickers
        }

        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                if result is not None:
                    new_rows.append(result)
            except Exception as ex:
                logging.error(f"Error in thread for {ticker}: {ex}")

    return new_rows


def fetch_new_data(ticker, fmp, last_timestamps):
    """
    Fetch new real-time data for a given ticker, returning a data row if the timestamp is newer than
    the last recorded timestamp in last_timestamps. Otherwise, return None.

    Args:
        ticker (str): The ticker symbol to fetch data for.
        fmp (FMPMarketData): An instance of the market data client.
        last_timestamps (dict): Dictionary tracking the last seen timestamp for each ticker.

    Returns:
        dict or None: A dictionary representing new market data for insertion, or None if no new data.
    """
    try:
        data = fmp.get_realtime_quote(ticker)
        if not data:
            logging.warning(f"No data received for ticker: {ticker}")
            return None
        
        # Convert the 'date' string from FMP into a datetime object
        dt = datetime.strptime(data["date"], "%Y-%m-%d %H:%M:%S")

        if not _is_new_timestamp(ticker, dt, last_timestamps):
            logging.info(f"Duplicate or older timestamp for {ticker}. Skipping entry.")
            return None

        # Update last timestamp
        last_timestamps[ticker] = dt

        row = _build_market_data_row(ticker, data, dt)
        logging.info(f"{ticker}: {data['close']} (new timestamp: {dt})")
        return row

    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None


def _is_new_timestamp(ticker, dt, last_timestamps):
    """
    Checks if the given timestamp is newer than what we have stored for the ticker.
    """
    return (ticker not in last_timestamps) or (dt > last_timestamps[ticker])


def _build_market_data_row(ticker, data, dt):
    """
    Builds the dictionary representing a row to be inserted into the market_data table.
    """
    return {
        "ticker": ticker,
        "timestamp": dt,             # full timestamp
        "date": dt.date(),           # date portion
        "exchange": "NASDAQ",        # default exchange (adjust if needed)
        "open_price": data["open"],
        "high_price": data["high"],
        "low_price": data["low"],
        "close_price": data["close"],
        "volume": data["volume"]
    }
