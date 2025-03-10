"""
specific_backfill.py
---------------
Example usage of the backfill_data function.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from datetime import datetime, timedelta
from data_infra.orchestrator.backfill import backfill_data

if __name__ == "__main__":
    # 1. Define tickers
    my_tickers = ['TXG', 'MMM', 'ETNB']
    # 2. Define date range (eg: last 5 trading days)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=6)

    # 3. Call backfill
    backfill_data(
        tickers=my_tickers,
        start_date=start_date,
        end_date=end_date,
        interval=1,
        exchange="NASDAQ",  # or "NYSE", or None
        output_filename="2y_mkt_data.csv"
    )