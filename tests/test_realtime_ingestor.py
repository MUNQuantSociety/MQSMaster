import logging

import pytest

from src.orchestrator.realTime import realtimeDataIngestor as rti

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.workflow_live,
]

# 2026-09-15 14:00:00 UTC == 10:00 America/New_York
TS = 1789480800


def _quote(symbol, exchange="NASDAQ", price=10.0, volume=1000, timestamp=TS, **extra):
    row = {
        "symbol": symbol,
        "exchange": exchange,
        "price": price,
        "volume": volume,
        "timestamp": timestamp,
        "open": 9.0,
        "dayHigh": 11.0,
        "dayLow": 8.0,
    }
    row.update(extra)
    return row


class TestProcessMarketData:
    def test_filters_to_tracked_tickers(self):
        rows = rti.process_market_data(
            [_quote("AAPL"), _quote("ZZZZ")], {"AAPL"}, {}, feed_label="NASDAQ"
        )
        assert [r["ticker"] for r in rows] == ["AAPL"]

    def test_uses_row_exchange_over_feed_label(self):
        rows = rti.process_market_data(
            [_quote("BRK-A", exchange="AMEX")], {"BRK-A"}, {}, feed_label="NYSE"
        )
        assert rows[0]["exchange"] == "AMEX"

    def test_null_exchange_falls_back_to_feed_label(self):
        rows = rti.process_market_data(
            [_quote("XYZ", exchange=None)], {"XYZ"}, {}, feed_label="NYSE"
        )
        assert rows[0]["exchange"] == "NYSE"

    def test_null_exchange_without_feed_label_is_dropped(self):
        rows = rti.process_market_data(
            [_quote("XYZ", exchange=None), _quote("AAPL")], {"XYZ", "AAPL"}, {}
        )
        assert [r["ticker"] for r in rows] == ["AAPL"]

    def test_ohl_stay_null_even_when_fmp_returns_day_aggregates(self):
        rows = rti.process_market_data([_quote("AAPL")], {"AAPL"}, {}, feed_label="NASDAQ")
        assert rows[0]["open_price"] is None
        assert rows[0]["high_price"] is None
        assert rows[0]["low_price"] is None
        assert rows[0]["close_price"] == 10.0

    def test_interval_volume_is_delta_from_last_cumulative(self):
        state = {"AAPL": 400}
        rows = rti.process_market_data(
            [_quote("AAPL", volume=1000)], {"AAPL"}, state, feed_label="NASDAQ"
        )
        assert rows[0]["volume"] == 600
        assert state["AAPL"] == 1000

    def test_interval_volume_resets_to_cumulative_when_delta_negative(self):
        state = {"AAPL": 5000}
        rows = rti.process_market_data(
            [_quote("AAPL", volume=1000)], {"AAPL"}, state, feed_label="NASDAQ"
        )
        assert rows[0]["volume"] == 1000

    def test_null_price_rows_are_dropped(self):
        rows = rti.process_market_data(
            [_quote("AAPL", price=None), _quote("MSFT")], {"AAPL", "MSFT"}, {}, feed_label="NASDAQ"
        )
        assert [r["ticker"] for r in rows] == ["MSFT"]

    def test_null_volume_is_recorded_as_zero(self):
        state = {}
        rows = rti.process_market_data(
            [_quote("AAPL", volume=None)], {"AAPL"}, state, feed_label="NASDAQ"
        )
        assert rows[0]["volume"] == 0
        assert state["AAPL"] == 0

    def test_timestamp_is_converted_to_new_york_and_rounded_to_minute(self):
        rows = rti.process_market_data(
            [_quote("AAPL", timestamp=TS + 29)], {"AAPL"}, {}, feed_label="NASDAQ"
        )
        ts = rows[0]["timestamp"]
        assert str(ts.tzinfo) == "America/New_York"
        assert (ts.hour, ts.minute, ts.second) == (10, 0, 0)
        assert str(rows[0]["date"]) == "2026-09-15"

    def test_empty_feed_returns_no_rows(self):
        assert rti.process_market_data([], {"AAPL"}, {}, feed_label="NASDAQ") == []


class _FakeFMP:
    def __init__(self, feeds):
        self.feeds = feeds
        self.calls = []

    def get_realtime_data(self, exchange):
        self.calls.append(exchange)
        return self.feeds.get(exchange)

    def get_batch_crypto_quotes(self):
        self.calls.append("CRYPTO")
        return self.feeds.get("CRYPTO")

    def get_batch_commodity_quotes(self):
        self.calls.append("COMMODITY")
        return self.feeds.get("COMMODITY")


class _FakeDB:
    def __init__(self, read_result=None):
        self.inserted = []
        self.read_result = read_result

    def bulk_inject_to_db(self, table, data, conflict_columns=None, schema=""):
        self.inserted.append((table, data, conflict_columns))
        return {
            "status": "success",
            "message": "ok",
            "inserted_count": len(data),
            "ignored_count": 0,
        }

    def read_db(self, sql):
        return self.read_result


class TestRunIngestionCycle:
    def test_polls_every_feed_and_assigns_each_feeds_exchange(self):
        fmp = _FakeFMP({
            "NASDAQ": [_quote("AAPL", exchange="NASDAQ")],
            "NYSE": [_quote("A", exchange="NYSE")],
            "AMEX": [],
            "CRYPTO": [_quote("BTCUSD", exchange="CRYPTO")],
            "COMMODITY": [_quote("GCUSD", exchange="COMMODITY")],
        })
        db = _FakeDB()
        uncovered = rti.run_ingestion_cycle(fmp, db, {"AAPL", "A", "BTCUSD", "GCUSD", "GONE"}, {})

        assert fmp.calls == list(rti.QUOTE_FEEDS)
        _, rows, conflict = db.inserted[0]
        assert {r["ticker"]: r["exchange"] for r in rows} == {
            "AAPL": "NASDAQ", "A": "NYSE", "BTCUSD": "CRYPTO", "GCUSD": "COMMODITY",
        }
        assert conflict == ["ticker", "timestamp"]
        assert uncovered == {"GONE"}

    def test_ticker_seen_in_earlier_feed_is_not_reprocessed_by_later_feed(self):
        fmp = _FakeFMP({
            "NASDAQ": [_quote("DUP", exchange="NASDAQ", volume=1000)],
            "NYSE": [_quote("DUP", exchange="NYSE", volume=1000)],
        })
        db = _FakeDB()
        state = {}
        rti.run_ingestion_cycle(fmp, db, {"DUP"}, state)

        _, rows, _ = db.inserted[0]
        assert len(rows) == 1
        assert rows[0]["exchange"] == "NASDAQ"
        assert rows[0]["volume"] == 1000

    def test_failed_feed_is_skipped_and_others_still_insert(self):
        fmp = _FakeFMP({"NASDAQ": None, "NYSE": [_quote("A", exchange="NYSE")]})
        db = _FakeDB()
        uncovered = rti.run_ingestion_cycle(fmp, db, {"A", "AAPL"}, {})

        assert [r["ticker"] for r in db.inserted[0][1]] == ["A"]
        assert uncovered == {"AAPL"}

    def test_no_rows_means_no_insert(self):
        fmp = _FakeFMP({})
        db = _FakeDB()
        rti.run_ingestion_cycle(fmp, db, {"AAPL"}, {})
        assert db.inserted == []


class TestVerifyUpsertConstraint:
    def test_accepts_unique_index_on_ticker_timestamp_in_any_order(self):
        db = _FakeDB({
            "status": "success",
            "data": [
                {"index_name": "market_data_pkey", "columns": ["id"]},
                {"index_name": "market_data_ticker_timestamp_key", "columns": ["timestamp", "ticker"]},
            ],
        })
        assert rti.verify_upsert_constraint(db) is True

    def test_rejects_when_only_non_matching_unique_indexes_exist(self, caplog):
        db = _FakeDB({
            "status": "success",
            "data": [{"index_name": "market_data_pkey", "columns": ["id"]}],
        })
        with caplog.at_level(logging.CRITICAL):
            assert rti.verify_upsert_constraint(db) is False
        assert "CREATE UNIQUE INDEX" in caplog.text

    def test_inconclusive_index_query_proceeds_with_critical_log(self, caplog):
        db = _FakeDB({"status": "error", "message": "boom", "data": None})
        with caplog.at_level(logging.CRITICAL):
            assert rti.verify_upsert_constraint(db) is True
        assert "boom" in caplog.text
