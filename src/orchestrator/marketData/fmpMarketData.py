import threading
import requests
import time
import pandas as pd
from datetime import datetime
from common.auth.apiAuth import APIAuth
import logging

class FMPMarketData:
    """
    Thread-safe FMP market data client.
    Allows multiple threads/processes to call get_intraday_data / get_historical_data
    simultaneously, without exceeding 2999 requests per minute.
    Also handles internet outages, request retries, and timeouts.
    """

    # A class-level lock, used if you want to share a single rate-limiter across
    # multiple FMPMarketData instances. If you only ever create one instance,
    # an instance-level lock is sufficient. We'll do instance-level below.
    # _global_lock = threading.Lock()

    def __init__(self):
        self.api_auth = APIAuth()
        self.fmp_api_key = self.api_auth.get_fmp_api_key()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Rate Limiting Config
        self.request_timestamps = []
        self.MAX_REQUESTS_PER_MIN = 3000
        self.LOCK_WINDOW_SECONDS = 60

        # API Request Config
        self.MAX_RETRIES = 6
        self.TIMEOUT_SECONDS = 10  # Prevents script from freezing on a request

        # A lock to protect rate-limiter data (request_timestamps), etc.
        self._lock = threading.Lock()

    def _check_rate_limit(self):
        """
        Enforces API rate limits in a thread-safe manner.
        If the limit is exceeded, it waits.
        """
        with self._lock:
            current_time = time.time()

            # Remove timestamps older than the rate limit window (60 seconds)
            self.request_timestamps = [
                t for t in self.request_timestamps
                if (current_time - t) < self.LOCK_WINDOW_SECONDS
            ]

            # If we are at or above the limit, wait
            if len(self.request_timestamps) >= self.MAX_REQUESTS_PER_MIN:
                wait_time = self.LOCK_WINDOW_SECONDS - (current_time - self.request_timestamps[0])
                if wait_time > 0:
                    print(f"[RateLimiter] Hit API limit ({self.MAX_REQUESTS_PER_MIN} calls/min). Sleeping for {wait_time:.2f} s...")
                    time.sleep(wait_time)

            # Record timestamp for this request
            self.request_timestamps.append(time.time())

    def _wait_for_internet(self):
        """
        Keeps retrying until the internet is restored (thread-safe approach).
        All threads that lose connection will call this.
        """
        # We could add a global or shared lock here so only
        # one thread checks connectivity, but it's simpler
        # to allow each thread to do it if they're stuck.
        while True:
            try:
                requests.get("https://www.google.com", timeout=5)  # Check internet access
                print("[Internet] Connection restored ✅")
                return  # Exit loop when internet is back
            except requests.exceptions.ConnectionError:
                print("[Internet] No connection. Retrying in 10 seconds...")
                time.sleep(10)  # Wait before retrying

    def _make_request(self, url, params):
        """
        Handles API requests with retries, error handling, 
        internet failure detection, and thread-safe rate limiting.
        """
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # Enforce rate limit before making the request
                self._check_rate_limit()

                response = requests.get(url, params=params, timeout=self.TIMEOUT_SECONDS)

                if response.status_code == 200:
                    return response.json()

                print(f"[FMP API] Warning: Received {response.status_code} - {response.text} (Attempt {attempt}/{self.MAX_RETRIES})")

                # If it's a 429 (Too Many Requests), wait longer (exponential backoff)
                if response.status_code == 429:
                    print("[FMP API] Too many requests. Waiting before retrying...")
                    time.sleep(10 * attempt)

            except requests.exceptions.Timeout:
                print(f"[FMP API] Timeout on {url} (Attempt {attempt}/{self.MAX_RETRIES}). Retrying...")
                time.sleep(5 * attempt)

            except requests.exceptions.ConnectionError:
                print("[FMP API] Lost internet connection. Waiting for reconnection...")
                self._wait_for_internet()  # Block until we get the net back

            except requests.exceptions.RequestException as ex:
                # Catches all other requests-related errors
                print(f"[FMP API] Request failed: {ex} (Attempt {attempt}/{self.MAX_RETRIES})")
                time.sleep(5 * attempt)

        print(f"[FMP API] Failed to fetch data from {url} after {self.MAX_RETRIES} attempts.")
        return None  # Fail gracefully

    def get_historical_data(self, tickers, from_date, to_date):
        """
        Fetches daily historical data for given tickers.
        
        :param tickers: List or string of stock symbols
        :param from_date: Start date (YYYY-MM-DD)
        :param to_date: End date (YYYY-MM-DD)
        :return: List of historical records or None
        """
        if isinstance(tickers, list):
            tickers = ",".join(tickers)

        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{tickers}"
        params = {"from": from_date, "to": to_date, "apikey": self.fmp_api_key}

        data = self._make_request(url, params)
        if not data:
            return None

        # Parse single or multiple ticker response
        historical_data = []
        if isinstance(data, dict) and 'historical' in data:
            return data['historical']  # single ticker
        elif isinstance(data, dict) and 'historicalStockList' in data:
            for stock in data['historicalStockList']:
                t_symbol = stock['symbol']
                for record in stock['historical']:
                    record['ticker'] = t_symbol
                    historical_data.append(record)
            return historical_data

        print("[FMP API] No historical data found.")
        return None

    def get_intraday_data(self, tickers, from_date, to_date, interval):
        """
        Fetches intraday historical data.

        :param tickers: List or string of stock symbols
        :param from_date: Start date (YYYY-MM-DD)
        :param to_date: End date (YYYY-MM-DD)
        :param interval: Time interval in minutes (1, 5, 15, 30, 60)
        :return: List of intraday records or None
        """
        if isinstance(tickers, list):
            tickers = ",".join(tickers)

        interval_str = "1hour" if interval == 60 else f"{interval}min"

        url = f"https://financialmodelingprep.com/api/v3/historical-chart/{interval_str}/{tickers}"
        params = {"from": from_date, "to": to_date, "apikey": self.fmp_api_key}

        data = self._make_request(url, params)
        if not data:
            return None

        # Should be a list of dicts if successful
        if isinstance(data, list) and len(data) > 0:
            return data

        print("[FMP API] No intraday data found.")
        return None
        
    def get_realtime_data(self, exchange="NASDAQ"):
        """
        Fetch real-time stock data for an entire exchange using a single batch request.

        Args:
            exchange (str): The stock exchange to query (e.g., "NASDAQ", "NYSE").

        Returns:
            list: A list of dictionaries, where each dict contains quote data for a ticker.
                  Returns None if the API call fails.
        """
        url = "https://financialmodelingprep.com/stable/batch-exchange-quote?"
        params = {"exchange": exchange, "short": "false", "apikey": self.fmp_api_key}
        
        print(f"Fetching batch data for {exchange}...")
        data = self._make_request(url, params)
        
        if isinstance(data, list):
            return data

        print(f"[FMP API] Failed to fetch or parse batch data for {exchange}.")
        return None
    
    def get_current_price(self, ticker):
        """
        Fetch real-time stock price for a single ticker using FMP API.
        Returns float: Current price of the ticker, or 0.0 if not found or on error.
        """
        url = f"https://financialmodelingprep.com/stable/quote?symbol={ticker}"
        params = {"apikey": self.fmp_api_key}

        try:
            data = self._make_request(url, params)
            if isinstance(data, list) and data and 'price' in data[0] and data[0]['price'] is not None:
                return float(data[0]['price'])

            self.logger.warning(f"No valid price found for ticker {ticker} in API response. Response: {data}")
            return 0.0

        except Exception as e:
            self.logger.error(f"Price fetch failed for {ticker}: {e}")
            return 0.0