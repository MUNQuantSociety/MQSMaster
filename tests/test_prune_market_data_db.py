"""Integration test for the market_data retention pruner's actual delete.

Runs against a real MQSDBConnector (the db_connection fixture, CI's
throwaway postgres:16-alpine service container). Seeds rows both older and
newer than the cutoff, scoped to a distinctive test ticker so this is safe
to run even against a shared/persistent database, then asserts the batched
delete removes only the old ones.
"""

from datetime import date, timedelta

import pytest

from src.orchestrator.retention.prune_market_data import _prune_once

pytestmark = [pytest.mark.db]

TEST_TICKER = "ZZPRUNE1"  # market_data.ticker is VARCHAR(10) -- must fit

_INSERT_SQL = """
    INSERT INTO market_data (ticker, timestamp, date, exchange, open_price, high_price, low_price, close_price, volume)
    VALUES (%s, %s, %s, 'TEST', 1.0, 1.0, 1.0, 1.0, 100)
"""


def test_prune_once_deletes_only_rows_older_than_cutoff(db_connection):
    today = date.today()
    retention_days = 545
    old_date = today - timedelta(days=retention_days + 30)
    recent_date = today - timedelta(days=10)

    try:
        insert_old = db_connection.execute_query(_INSERT_SQL, (TEST_TICKER, old_date, old_date))
        insert_recent = db_connection.execute_query(_INSERT_SQL, (TEST_TICKER, recent_date, recent_date))
        assert insert_old["status"] == "success", insert_old.get("message")
        assert insert_recent["status"] == "success", insert_recent.get("message")

        deleted = _prune_once(db_connection, today, retention_days=retention_days, batch_size=1000)
        assert deleted >= 1

        remaining = db_connection.execute_query(
            "SELECT date FROM market_data WHERE ticker = %s",
            (TEST_TICKER,),
            fetch=True,
        )
        remaining_dates = {row["date"] for row in remaining["data"]}

        assert old_date not in remaining_dates
        assert recent_date in remaining_dates
    finally:
        db_connection.execute_query(
            "DELETE FROM market_data WHERE ticker = %s", (TEST_TICKER,)
        )
