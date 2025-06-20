# data_infra/tradingOps/backtest/utils.py

import pandas as pd
import logging
from typing import List, Optional, Union
from datetime import datetime

from portfolios.portfolio_BASE.strategy import BasePortfolio


def fetch_historical_data(
    portfolio: BasePortfolio,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Fetches historical market data for the portfolio's tickers within the date range.

    Args:
        portfolio: The portfolio instance containing tickers and DB connection.
        start_date: The start datetime for the data query.
        end_date: The end datetime for the data query.

    Returns:
        A pandas DataFrame containing cleaned market data, or an empty DataFrame on failure.
    """
    logger = portfolio.logger
    tickers = getattr(portfolio, "tickers", [])

    if not tickers:
        logger.warning("No tickers specified in the portfolio; returning empty DataFrame.")
        return pd.DataFrame()

    # Prepare SQL
    placeholders = ", ".join(["%s"] * len(tickers))
    sql = f"""
        SELECT *
          FROM market_data
         WHERE ticker IN ({placeholders})
           AND timestamp BETWEEN %s AND %s
         ORDER BY timestamp ASC
    """
    params = tickers + [start_date, end_date]

    logger.debug(
        f"Executing historical data query for {len(tickers)} tickers "
        f"from {start_date} to {end_date}."
    )

    try:
        result = portfolio.db.execute_query(sql, params, fetch=True)
    except Exception as e:
        logger.exception(f"Database query exception: {e}", exc_info=True)
        return pd.DataFrame()

    if result.get("status") != "success":
        msg = result.get("message", "<no message>")
        logger.error(f"Database query failed: {msg}")
        return pd.DataFrame()

    raw_data = result.get("data", [])
    if not raw_data:
        logger.warning("Historical data query returned no rows.")
        return pd.DataFrame()

    df = pd.DataFrame(raw_data)

    # Parse and clean
    # 1) Timestamp → datetime (UTC)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    # 2) Numeric columns
    numeric_cols = ["open_price", "high_price", "low_price", "close_price", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before_drop = len(df)
    df.dropna(subset=["timestamp", "ticker", "close_price"], inplace=True)
    dropped = before_drop - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows due to missing timestamp, ticker, or close_price.")

    if df.empty:
        logger.error("No valid data remains after cleaning; returning empty DataFrame.")
        return pd.DataFrame()

    # Sort and reset index
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Fetched and cleaned {len(df)} rows of historical data.")
    return df
