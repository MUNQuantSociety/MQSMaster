import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import requests

SRC_ROOT = Path(__file__).resolve().parents[2]
BACKFILL_DIR = SRC_ROOT / "orchestrator" / "backfill"
SP500_TICKERS_PATH = BACKFILL_DIR / "update" / "extra_tickers" / "sp500_tickers.json"
COMMODITY_TICKERS_PATH = BACKFILL_DIR / "update" / "extra_tickers" / "commodity_tickers.json"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from common.auth.apiAuth import APIAuth
    from common.fmp_rate_limiter import FMPRateLimiter
    from orchestrator.backfill.update.crypto_tickers import (
        format_for_fmp,
        get_top_crypto_from_coingecko,
        load_reference_list,
        save_crypto_details,
    )
except Exception as e:
    logging.error(f"Failed to import APIAuth after adding src to sys.path. {e}")
    raise


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
        if not isinstance(self.fmp_api_key, str) or not self.fmp_api_key:
            raise ValueError(
                "FMP API key is missing or invalid (empty). Ensure FMP_API_KEY is set in environment / .env"
            )
        self.logger = logging.getLogger(self.__class__.__name__)

        # Rate Limiting: shared process-singleton limiter at 1000 req/min for
        # the realtime ingestor. NLP gets the rest of the 3000 account budget
        # via its own FMPRateLimiter.for_nlp() instance. See
        # src/common/fmp_rate_limiter.py.
        self._rate_limiter = FMPRateLimiter.for_realtime()

        # API Request Config
        self.MAX_RETRIES = 6
        self.TIMEOUT_SECONDS = 10  # Prevents script from freezing on a request

        # Crypto ticker retrieval defaults
        self.CRYPTO_TOP_LIMIT = 500

        # Connectivity recovery defaults
        self.INTERNET_RETRY_INTERVAL_SECONDS = 10
        self.INTERNET_MAX_WAIT_SECONDS = 120

    def _check_rate_limit(self):
        """Block until a slot is available, then claim it."""
        self._rate_limiter.acquire()

    def _wait_for_internet(self, max_wait_seconds=None):
        """
        Keeps retrying until the internet is restored (thread-safe approach),
        with an optional max wait to avoid blocking forever.
        All threads that lose connection will call this.
        """
        if max_wait_seconds is None:
            max_wait_seconds = self.INTERNET_MAX_WAIT_SECONDS
        start_time = time.time()

        # We could add a global or shared lock here so only
        # one thread checks connectivity, but it's simpler
        # to allow each thread to do it if they're stuck.
        while True:
            try:
                requests.get(
                    "https://www.google.com", timeout=5
                )  # Check internet access
                self.logger.info("[Internet] Connection restored")
                return True  # Exit loop when internet is back
            except requests.exceptions.ConnectionError:
                elapsed = time.time() - start_time
                if elapsed >= max_wait_seconds:
                    self.logger.warning(
                        "[Internet] Still offline after %ss. Proceeding with retry limits.",
                        max_wait_seconds,
                    )
                    return False
                self.logger.warning(
                    "[Internet] No connection. Retrying in %s seconds...",
                    self.INTERNET_RETRY_INTERVAL_SECONDS,
                )
                time.sleep(self.INTERNET_RETRY_INTERVAL_SECONDS)

    def _make_request(self, url, params):
        """
        Handles API requests with retries, error handling,
        internet failure detection, and thread-safe rate limiting.
        """
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # Enforce rate limit before making the request
                self._check_rate_limit()

                response = requests.get(
                    url, params=params, timeout=self.TIMEOUT_SECONDS
                )

                if response.status_code == 200:
                    with self._lock:
                        self.request_timestamps.append(time.time())
                    return response.json()

                self.logger.warning(
                    "[FMP API] Received %s - %s (Attempt %s/%s)",
                    response.status_code,
                    response.text,
                    attempt,
                    self.MAX_RETRIES,
                )

                # If it's a 429 (Too Many Requests), wait longer (exponential backoff)
                if response.status_code == 429:
                    self.logger.warning(
                        "[FMP API] Too many requests. Waiting before retrying..."
                    )
                    time.sleep(10 * attempt)

            except requests.exceptions.Timeout:
                self.logger.warning(
                    "[FMP API] Timeout on %s (Attempt %s/%s). Retrying...",
                    url,
                    attempt,
                    self.MAX_RETRIES,
                )
                time.sleep(5 * attempt)

            except requests.exceptions.ConnectionError:
                self.logger.warning(
                    "[FMP API] Lost internet connection. Waiting for reconnection..."
                )
                reconnected = self._wait_for_internet()
                if not reconnected:
                    self.logger.warning(
                        "[FMP API] Reconnection timeout reached for this attempt."
                    )

            except requests.exceptions.RequestException as ex:
                # Catches all other requests-related errors
                self.logger.warning(
                    "[FMP API] Request failed: %s (Attempt %s/%s)",
                    ex,
                    attempt,
                    self.MAX_RETRIES,
                )
                time.sleep(5 * attempt)

        self.logger.error(
            "[FMP API] Failed to fetch data from %s after %s attempts.",
            url,
            self.MAX_RETRIES,
        )
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

        url = (
            f"https://financialmodelingprep.com/api/v3/historical-price-full/{tickers}"
        )
        params = {"from": from_date, "to": to_date, "apikey": self.fmp_api_key}

        data = self._make_request(url, params)
        if not data:
            return None

        # Parse single or multiple ticker response
        historical_data = []
        if isinstance(data, dict) and "historical" in data:
            return data["historical"]  # single ticker
        elif isinstance(data, dict) and "historicalStockList" in data:
            for stock in data["historicalStockList"]:
                t_symbol = stock["symbol"]
                for record in stock["historical"]:
                    record["ticker"] = t_symbol
                    historical_data.append(record)
            return historical_data

        self.logger.info("[FMP API] No historical data found.")
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

        self.logger.info("[FMP API] No intraday data found.")
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
        url = "https://financialmodelingprep.com/stable/batch-exchange-quote"
        params = {"exchange": exchange, "short": "false", "apikey": self.fmp_api_key}

        self.logger.info("Fetching batch data for %s...", exchange)
        data = self._make_request(url, params)

        if isinstance(data, list):
            return data

        self.logger.error(
            "[FMP API] Failed to fetch or parse batch data for %s.", exchange
        )
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
            if (
                isinstance(data, list)
                and data
                and "price" in data[0]
                and data[0]["price"] is not None
            ):
                return float(data[0]["price"])

            self.logger.warning(
                f"No valid price found for ticker {ticker} in API response. Response: {data}"
            )
            return 0.0

        except Exception as e:
            self.logger.error(f"Price fetch failed for {ticker}: {e}")
            return 0.0

    def _write_json_atomic(self, target_path: Path, tickers: List[str], label: str) -> bool:
        temp_path: Optional[Path] = None
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=target_path.parent,
                prefix=f".{target_path.name}.",
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as temp_file:
                json.dump(tickers, temp_file, indent=2)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_path = Path(temp_file.name)

            os.replace(temp_path, target_path)
            return True
        except (OSError, IOError) as error:
            self.logger.error(
                "[FMP API] Failed to write %s (%s tickers) to %s: %s",
                label,
                len(tickers),
                target_path,
                error,
                exc_info=True,
            )
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            return False

    def get_sp500_tickers(self) -> list[str]:
        """
        Fetches the list of S&P 500 tickers from FMP API.
        Returns a list of ticker symbols.
        File write errors are logged and do not change the returned ticker list.
        """
        url = "https://financialmodelingprep.com/stable/sp500-constituent"
        params = {"apikey": self.fmp_api_key}

        data = self._make_request(url, params)
        if not data:
            self.logger.error("[FMP API] Failed to fetch S&P 500 tickers.")
            return []

        tickers = [
            item.get("symbol")
            for item in data
            if isinstance(item, dict)
            and isinstance(item.get("symbol"), str)
            and item.get("symbol").strip()
        ]
        if not tickers:
            self.logger.error("[FMP API] No valid S&P 500 ticker symbols found.")
            return []
        self._write_json_atomic(SP500_TICKERS_PATH, tickers, "S&P 500 tickers")
        return tickers

    def get_crypto_tickers(self) -> list[str]:
        """
        Fetches top crypto tickers using CoinGecko and formats them for FMP.
        Returns a list of ticker symbols.
        """

        try:
            coingecko_data = get_top_crypto_from_coingecko(limit=self.CRYPTO_TOP_LIMIT)
            reference_rows = load_reference_list()
            fmp_tickers, _matched_count = format_for_fmp(
                coingecko_data, reference_rows
            )

        except Exception as e:
            self.logger.error("Error fetching crypto data for formatting: %s", e, exc_info=True)
            return []

        symbols = []
        seen = set()
        for ticker in fmp_tickers:
            symbol = ticker.get("symbol")
            if isinstance(symbol, str) and symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)

        if not symbols:
            self.logger.error("[FMP API] Error parsing crypto ticker symbols.")
            return []

        save_crypto_details(fmp_tickers, log_summary=False)
        return symbols

    def get_commodity_tickers(self) -> list[str]:
        """
        Fetches the list of commodity tickers from FMP API.
        Returns a list of ticker symbols.
        File write errors are logged and do not change the returned ticker list.
        """
        url = "https://financialmodelingprep.com/stable/batch-commodity-quotes"
        params = {"apikey": self.fmp_api_key}

        data = self._make_request(url, params)
        if not data:
            self.logger.error("[FMP API] Failed to fetch commodity tickers.")
            return []

        tickers = [
            item.get("symbol")
            for item in data
            if isinstance(item, dict)
            and isinstance(item.get("symbol"), str)
            and item.get("symbol").strip()
        ]
        if not tickers:
            self.logger.error("[FMP API] No valid commodity ticker symbols found.")
            return []
        self._write_json_atomic(COMMODITY_TICKERS_PATH, tickers, "commodity tickers")
        return tickers


if __name__ == "__main__":
    fmp = FMPMarketData()
    sp500_tickers = fmp.get_sp500_tickers()
    print(f"First 10 S&P 500 Tickers:\n {sp500_tickers[:10]}")

    crypto_tickers = fmp.get_crypto_tickers()
    print(f"First 10 Crypto Tickers:\n {crypto_tickers[:10]}")

    commodity_tickers = fmp.get_commodity_tickers()
    print(f"Commodity Tickers:\n {commodity_tickers[:10]}")
