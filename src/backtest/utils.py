from collections import defaultdict
from datetime import datetime
from typing import List

import pandas as pd

from src.backtest.data.backfill_cache import cache as _cache
from src.portfolios.portfolio_BASE.strategy import BasePortfolio


def _fetch_from_db(portfolio, tickers: List[str], start, end) -> pd.DataFrame:
    """
    Fetches raw intraday market data for the specified ticker(s) within the
    specified date range. Returns one row per bar (no daily aggregation).
    """
    logger = portfolio.logger
    placeholders = ", ".join(["%s"] * len(tickers))
    # NOTE: replaced an ARRAY_AGG-based per-day aggregator with a much cheaper
    # DISTINCT ON-style pull. For 519-ticker x multi-year intraday backtests the
    # original SQL blew PG timeouts; this version uses two index-friendly scans
    # (close = last bar of day, max/min/sum via GROUP BY) and joins them in
    # Python. close_price comes from the 16:00 bar via DISTINCT ON; open_price
    # from the 09:30 bar via a symmetric DISTINCT ON ASC.
    sql_close = f"""
        SELECT DISTINCT ON (ticker, DATE(timestamp AT TIME ZONE 'America/New_York'))
            ticker,
            timestamp,
            close_price
          FROM market_data
         WHERE ticker IN ({placeholders})
           AND timestamp BETWEEN %s AND %s
           AND (timestamp AT TIME ZONE 'America/New_York')::time
               BETWEEN '09:30' AND '16:00'
         ORDER BY
            ticker,
            DATE(timestamp AT TIME ZONE 'America/New_York'),
            timestamp DESC
    """
    sql_open = f"""
        SELECT DISTINCT ON (ticker, DATE(timestamp AT TIME ZONE 'America/New_York'))
            ticker,
            DATE(timestamp AT TIME ZONE 'America/New_York') AS trade_date,
            open_price
          FROM market_data
         WHERE ticker IN ({placeholders})
           AND timestamp BETWEEN %s AND %s
           AND (timestamp AT TIME ZONE 'America/New_York')::time
               BETWEEN '09:30' AND '16:00'
         ORDER BY
            ticker,
            DATE(timestamp AT TIME ZONE 'America/New_York'),
            timestamp ASC
    """
    sql_hlv = f"""
        SELECT
            ticker,
            DATE(timestamp AT TIME ZONE 'America/New_York') AS trade_date,
            MAX(high_price) AS high_price,
            MIN(low_price)  AS low_price,
            SUM(volume)     AS volume
          FROM market_data
         WHERE ticker IN ({placeholders})
           AND timestamp BETWEEN %s AND %s
           AND (timestamp AT TIME ZONE 'America/New_York')::time
               BETWEEN '09:30' AND '16:00'
         GROUP BY ticker, DATE(timestamp AT TIME ZONE 'America/New_York')
    """
    params = tickers + [start, end]
    logger.debug(
        "DB query for %d tickers from %s to %s", len(tickers), start, end
    )

    def _run(sql_text, label):
        try:
            res = portfolio.db.execute_query(sql_text, params, fetch=True)
        except Exception as e:
            logger.exception("DB query %s failed: %s", label, e, exc_info=True)
            return None
        if res.get("status") != "success":
            logger.error(
                "DB query %s failed: %s", label, res.get("message", "<no message>")
            )
            return None
        return res.get("data") or []

    rows_close = _run(sql_close, "close")
    rows_open = _run(sql_open, "open")
    rows_hlv = _run(sql_hlv, "hlv")
    if rows_close is None or rows_open is None or rows_hlv is None:
        return pd.DataFrame()
    if not rows_close:
        logger.warning("DB query returned no rows for tickers %s.", tickers)
        return pd.DataFrame()

    df_close = pd.DataFrame(rows_close)
    df_open = pd.DataFrame(rows_open)
    df_hlv = pd.DataFrame(rows_hlv)

    df_close["timestamp"] = pd.to_datetime(
        df_close["timestamp"], utc=True, errors="coerce"
    ).dt.tz_convert("America/New_York")
    df_close["trade_date"] = df_close["timestamp"].dt.date

    for frame in (df_open, df_hlv):
        if "trade_date" in frame.columns:
            frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date

    df = df_close.merge(
        df_open[["ticker", "trade_date", "open_price"]],
        on=["ticker", "trade_date"],
        how="left",
    ).merge(
        df_hlv[["ticker", "trade_date", "high_price", "low_price", "volume"]],
        on=["ticker", "trade_date"],
        how="left",
    )
    df = df.drop(columns=["trade_date"])

    for col in ["open_price", "high_price", "low_price", "close_price", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df.dropna(subset=["timestamp", "ticker", "close_price"], inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with invalid values after DB fetch.", dropped)

    return df


def fetch_historical_data(
    portfolio: BasePortfolio,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Returns daily OHLCV data for all portfolio tickers in [start_date, end_date].

    When db is queried for new tickers, new .parquet files are created in the cache
    which store tickers' data for the specified date range. Cached data is 
    returned when possible. If a ticker is already cached but a date range is 
    requested that is not included in the cache, db is queried for the missing
    ticker data and is added to the .parquet file.

    Files cached in src/backtest/data/backfill_cache/tickername.parquet. 
    """
    logger = portfolio.logger                       # Initialize debug logger
    tickers = getattr(portfolio, "tickers", [])     # Fetch desired tickers

    if not tickers:
        # If portfolio config has no tickers, return empty df
        logger.warning("No tickers specified in the portfolio; returning empty DataFrame.")
        return pd.DataFrame()

    # Initialize start and end dates
    start = pd.to_datetime(start_date, utc=True)
    # Ensure end date includes the entire end day, up until close
    end = pd.to_datetime(end_date, utc=True) + pd.Timedelta(
        hours=23, minutes=59, seconds=59
    )
    # Load any available ticker data from local cache
    caches = {t: _cache.load(t) for t in tickers}


    # Group tickers in dict by missing date range needed for batching db queries. Speeds
    # up query time. E.g., If one portfolio backtested with 3 tickers for a given date 
    # range, then another portfolio is backtested with a wider date range and 5 tickers,
    # 2 already having been cached from the previous backtest, then the missing
    # date range for both extant ticker caches is fetched in one db query.
    #
    #   {[start, end] : (ticker1, ticker2, ..., tickerk)}
    #
    # @see backfill_cache/cache.py for missing_ranges calculation
    range_to_tickers = defaultdict(list)
    for ticker in tickers:
        for gap in _cache.missing_ranges(caches[ticker], start, end):
            range_to_tickers[gap].append(ticker)

    # --- Step 2: Fetch each unique missing range from DB (batched by range) ---
    # Fetch each unique missing range from db. Batch identical ranges into one query.
    # 
    if range_to_tickers:
        unique_tickers_needing_fetch = {t for ts in range_to_tickers.values() for t in ts}
        logger.info(
            "Cache miss: fetching %d date range(s) from DB covering %d ticker(s).",
            len(range_to_tickers),
            len(unique_tickers_needing_fetch),
        )
        # Chunk by BOTH ticker and date window. A single 30-ticker x 6yr
        # intraday SQL (with GROUP BY day aggregation) overwhelmed PG: ~17M raw
        # rows scanned per query, timing out repeatedly. Smaller slabs let each
        # query finish in seconds.
        TICKER_CHUNK = 10
        DATE_SLAB_DAYS = 365
        slab_td = pd.Timedelta(days=DATE_SLAB_DAYS)
        for (gap_start, gap_end), gap_tickers in range_to_tickers.items():
            slab_start = gap_start
            slab_idx = 0
            while slab_start < gap_end:
                slab_end = min(slab_start + slab_td, gap_end)
                slab_idx += 1
                for chunk_start in range(0, len(gap_tickers), TICKER_CHUNK):
                    chunk = gap_tickers[chunk_start : chunk_start + TICKER_CHUNK]
                    logger.info(
                        "Fetching DB slab%d %s..%s tickers %d-%d/%d",
                        slab_idx,
                        slab_start.date(),
                        slab_end.date(),
                        chunk_start + 1,
                        chunk_start + len(chunk),
                        len(gap_tickers),
                    )
                    fetched = _fetch_from_db(portfolio, chunk, slab_start, slab_end)
                    if fetched.empty:
                        continue
                    for ticker in chunk:
                        ticker_rows = fetched[fetched["ticker"] == ticker]
                        caches[ticker] = _cache.merge_and_save(
                            ticker, caches[ticker], ticker_rows
                        )
                slab_start = slab_end
    else:
        logger.info("Fetching all data from local cache")


    # Splice each ticker's cache to the specified range
    parts = []
    for ticker in tickers:
        df = caches[ticker]
        if not df.empty:
            parts.append(df[(df["timestamp"] >= start) & (df["timestamp"] <= end)])

    if not parts:
        # Failed to splice any data
        logger.error("No data available after cache splice.")
        return pd.DataFrame()

    # Concatenate processed (spliced) ticker data, sort by timestamp, reset index, return
    result = pd.concat(parts, ignore_index=True)
    result.sort_values("timestamp", inplace=True)
    result.reset_index(drop=True, inplace=True)
    logger.info("Returning %d rows of historical data (%d tickers).", len(result), len(tickers))
    return result
