# data_infra/tradingOps/backtest/utils.py

import pandas as pd
import logging
from typing import List, Optional, Union
from datetime import datetime

# Assuming portfolio structure includes:
# portfolio.logger, portfolio.tickers, portfolio.db.execute_query
from portfolios.portfolio_BASE.strategy import BasePortfolio


def fetch_historical_data(portfolio: BasePortfolio,
                            start_date: datetime,
                            end_date: datetime) -> pd.DataFrame:
    """
    Fetches historical market data for the portfolio's tickers within the date range.

    Args:
        portfolio: The portfolio instance containing tickers and DB connection.
        start_date: The start datetime for the data query.
        end_date: The end datetime for the data query.

    Returns:
        A pandas DataFrame containing the market data, or an empty DataFrame on failure.
    """
    logger = portfolio.logger # Use portfolio's logger
    tickers = portfolio.tickers

    if not tickers:
        logger.warning("No tickers specified in the portfolio; returning empty DataFrame.")
        return pd.DataFrame()

    # Prepare SQL query
    # Ensure date formats match database expectations if necessary (e.g., YYYY-MM-DD HH:MM:SS)
    # Using placeholders is crucial for security against SQL injection
    placeholders = ', '.join(['%s'] * len(tickers))
    sql = f"""
        SELECT *
        FROM market_data
        WHERE ticker IN ({placeholders})
          AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp ASC
    """
    # Parameters must match the order of %s placeholders
    params = tickers + [start_date, end_date]

    logger.debug(f"Executing SQL query for historical data with {len(tickers)} tickers.")

    try:
        result = portfolio.db.execute_query(sql, params, fetch=True)

        if result.get('status') == 'success':
            data = result.get('data', [])
            if data:
                logger.info(f"Successfully fetched {len(data)} rows of historical data.")
                return pd.DataFrame(data)
            else:
                logger.warning("Historical data query executed successfully but returned no rows.")
                return pd.DataFrame()
        else:
            error_msg = result.get('message', 'Unknown database error')
            logger.error(f"Database query failed: {error_msg}")
            return pd.DataFrame()

    except Exception as e:
        logger.exception(f"An error occurred during database query execution: {e}", exc_info=True)
        return pd.DataFrame()