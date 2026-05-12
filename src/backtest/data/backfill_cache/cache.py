"""
Per-ticker parquet cache for backtest market data.

Each ticker gets its own .parquet file in this directory. The cache is
date-range-aware: only date sub-ranges not already stored are fetched from
the DB, so re-running a backtest with the same (or overlapping) dates
skips the query entirely and loads from disk instead.
"""
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent


def _path(ticker: str) -> Path:
    """
    Generate path for ticker. Used for both fetching and creating
    new parquet files.

    This is safe -> prevents dir errors. Files will all be uniformly
    named and saved in the same location.
    """
    safe = ticker.replace("^", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}.parquet"


def load(ticker: str) -> pd.DataFrame:
    """
    Load cached data for a single ticker. 
    Returns empty DataFrame on miss.
    """
    path = _path(ticker)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        logger.debug(
            "[%s] Cache hit: %d rows (%s -> %s)",
            ticker, len(df), df["timestamp"].min(), df["timestamp"].max(),
        )
        return df
    except Exception as e:
        logger.warning("[%s] Cache read failed, will re-fetch from DB: %s", ticker, e)
        return pd.DataFrame()


def save(ticker: str, df: pd.DataFrame) -> None:
    """
    Saves a ticker's DataFrame to its parquet file, sorted by timestamp.
    """
    if df.empty:
        # Check that df isnt empty before attempting to sort or save.
        return
    try:
        # Sort values, save to parquet
        df.sort_values("timestamp").reset_index(drop=True).to_parquet(_path(ticker), index=False)
        logger.debug("[%s] Cache saved: %d rows", ticker, len(df))
    except Exception as e:
        # Error occured while sorting or saving
        logger.warning("[%s] Cache write failed: %s", ticker, e)


def missing_ranges(
    cached: pd.DataFrame,   # cached data passed as element of a list (can be an empty df)
    start: pd.Timestamp,
    end: pd.Timestamp
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Return the sub-ranges within [start, end] not covered by the cache to be used 
    in db query. Return up to 2 subranges, depending on whether data missing
    before and/or after. 

    Return entire [start,end] range if no cached data.
    Return empty list gaps [] if data already exists.
    """
    if cached.empty:
        return [(start, end)]

    cached_min = cached["timestamp"].min()
    cached_max = cached["timestamp"].max()

    gaps = []
    if start < cached_min:
        gaps.append((start, cached_min))
    if end > cached_max:
        gaps.append((cached_max, end))
    return gaps


def merge_and_save(ticker: str, cached: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new rows into cached data, deduplicate on timestamp, sort, save, return.
    """
    if new.empty:
        return cached
    combined = pd.concat([cached, new], ignore_index=True) if not cached.empty else new.copy()
    combined.drop_duplicates(subset=["timestamp"], inplace=True)
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    save(ticker, combined)
    return combined
