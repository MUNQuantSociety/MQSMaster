from datetime import datetime

import pandas as pd

from src.portfolios.portfolio_BASE.strategy import BasePortfolio


def fetch_historical_data(
    portfolio: BasePortfolio, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """
    Fetches historical market data for the portfolio's tickers within the date range.
    """
    logger = portfolio.logger
    tickers = getattr(portfolio, "tickers", [])

    if not tickers:
        logger.warning(
            "No tickers specified in the portfolio; returning empty DataFrame."
        )
        return pd.DataFrame()

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

    # --- Start of Final, Corrected Cleaning Logic ---

    # Step 1: Convert timestamp column to datetime objects, standardizing to UTC.
    # This robustly handles mixed timezone formats and is the fix for the original NaT issue.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Step 2: Now that the column has a datetime type, convert from UTC to 'America/New_York'.
    # This fixes the "Can only use .dt accessor" error.
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")

    # Step 3: Convert all numeric columns.
    numeric_cols = ["open_price", "high_price", "low_price", "close_price", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- End of Corrected Logic ---

    # Step 4: Drop any rows that failed coercion in the steps above.
    before_drop = len(df)
    df.dropna(subset=["timestamp", "ticker", "close_price"], inplace=True)
    dropped = before_drop - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows due to missing or invalid values.")

    if df.empty:
        logger.error("No valid data remains after cleaning; returning empty DataFrame.")
        return pd.DataFrame()

    # Final sort and index reset
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Fetched and cleaned {len(df)} rows of historical data.")
    return df
