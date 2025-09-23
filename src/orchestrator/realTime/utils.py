import os
import sys
import json

def load_tickers():
    """
    Load ticker symbols from config.json files located in portfolio directories.
    
    Args:
        portfolios_base_path (str): Relative path where the portfolios are located.
    
    Returns:
        list: A list of unique ticker symbols.
    """
    # 1. Load tickers from tickers.json
    script_dir = os.path.dirname(__file__)
    # Corrected typo from .jsonx to .json for robustness
    ticker_file_path = os.path.join(script_dir, '..', 'tickers.json')

    try:
        with open(ticker_file_path, 'r') as f:
            MY_TICKERS = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ticker file not found at {ticker_file_path}. Please create it.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {ticker_file_path}. Please check the file format.")
        sys.exit(1)

    return list(MY_TICKERS)
