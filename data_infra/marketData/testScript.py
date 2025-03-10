"""
Financial Modeling Prep API Data Fetcher

This script provides a step-by-step guide on how to pull historical and intraday stock data using the Financial Modeling Prep API.
Read the official documentation for more thorough steps.

Copy this into a jupyter notebook if you have the functionality, it'll be easier!

### Instructions:
0. Create a venv.

1. Install dependencies:
   ```sh
   pip install --no-cache-dir --only-binary :all: -r requirements.txt
   ```
   Ensure you are in the root director while running this.

2. Run the script:
   ```
   python data_infra/marketData/testScript.py
   ```

This script fetches:
- Historical stock data over a specified date range.
- Intraday stock data with user-defined intervals.
"""

import pandas as pd
import requests

def get_historical_data(tickers, from_date, to_date, api_key):
    """
    Fetch historical stock data from Financial Modeling Prep API.

    Args:
        tickers (list or str): A single ticker as a string or multiple tickers as a list (e.g., "AAPL" or ["AAPL", "MSFT"]).
        from_date (str): Start date in 'YYYY-MM-DD' format.
        to_date (str): End date in 'YYYY-MM-DD' format.
        api_key (str): Your API key from Financial Modeling Prep.

    Returns:
        pd.DataFrame: A DataFrame containing historical stock data.
    """
    if isinstance(tickers, list):
        tickers = ",".join(tickers)
    
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{tickers}?from={from_date}&to={to_date}&apikey={api_key}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        historical_data = []
        
        if 'historical' in data:
            return pd.DataFrame(data['historical'])
        elif 'historicalStockList' in data:
            for stock in data['historicalStockList']:
                for record in stock['historical']:
                    record['ticker'] = stock['symbol']
                    historical_data.append(record)
            return pd.DataFrame(historical_data)
        else:
            print("No historical data found.")
            return pd.DataFrame()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return pd.DataFrame()


def get_intraday_data(tickers, from_date, to_date, interval, api_key):
    """
    Fetch intraday historical stock data from Financial Modeling Prep API.

    Args:
        tickers (list or str): Ticker(s) as a string or list (e.g., "AAPL" or ["AAPL", "MSFT"]).
        from_date (str): Start date in 'YYYY-MM-DD' format.
        to_date (str): End date in 'YYYY-MM-DD' format.
        interval (int): Interval in minutes (1, 5, 15, 30, 60).
        api_key (str): Your API key.

    Returns:
        pd.DataFrame: A DataFrame containing intraday stock data.
    """
    if isinstance(tickers, list):
        tickers = ",".join(tickers)
    
    interval_map = {1: "1min", 5: "5min", 15: "15min", 30: "30min", 60: "1hour"}
    interval_str = interval_map.get(interval, "5min")
    
    url = f"https://financialmodelingprep.com/api/v3/historical-chart/{interval_str}/{tickers}?from={from_date}&to={to_date}&apikey={api_key}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")
        return pd.DataFrame()


if __name__ == "__main__":
    # User Inputs
    selected_tickers = ["AAPL", "MSFT"]
    from_date = "2025-03-06"
    to_date = "2025-03-07"
    interval = 5  # Interval in minutes (1, 5, 15, 30, 60)
    api_key = "YOUR_API_KEY"  # Find this in data_infra/authentication/apiAuth.py
    
    # Fetch historical data
    print("Fetching historical data...")
    historical_df = get_historical_data(selected_tickers, from_date, to_date, api_key)
    if not historical_df.empty:
        print(historical_df.head())
    else:
        print("No historical data available.")
    
    # Fetch intraday data
    print("Fetching intraday data...")
    intraday_df = get_intraday_data(selected_tickers, from_date, to_date, interval, api_key)
    if not intraday_df.empty:
        print(intraday_df.describe())
        print("------")
        print(intraday_df.head(5))
    else:
        print("No intraday data available.")
    
    print("Data fetching complete.")
