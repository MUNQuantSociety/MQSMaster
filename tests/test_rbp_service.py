# tests/test_rbp_service.py
"""Unit tests for :class:`RBP.service.RBPForecastService`.

These tests mock every external dependency (DB connector, market-data loader,
predictor, RBI calculator) so they can run without a database, without network,
and without real OHLCV data.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from typing import List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Import the service under test. If a parallel agent hasn't landed it yet the
# test module will still load - individual tests are skipped via a marker.
# ---------------------------------------------------------------------------
try:
    from RBP.service import (
        DEFAULT_HORIZON_DAYS,
        TABLE_NAME,
        RBPForecastService,
    )
    from RBP.config import RBPConfig

    RBP_AVAILABLE = True
except Exception as exc:  # pragma: no cover - service not yet in tree
    RBP_AVAILABLE = False
    _IMPORT_ERROR = exc


pytestmark = pytest.mark.skipif(
    not RBP_AVAILABLE,
    reason=f"RBP.service not importable: {_IMPORT_ERROR if not RBP_AVAILABLE else ''}",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_TICKERS: List[str] = ["AAPL", "MSFT", "NVDA"]
FEATURE_COLUMNS: List[str] = ["open_price", "close_price", "rsi_14"]
TARGET_COLUMN: str = "target_return_21d"


@pytest.fixture
def rbp_config():
    """Minimal RBPConfig wired for 3 tickers and a tiny feature set."""
    return RBPConfig(
        tickers=list(TEST_TICKERS),
        feature_columns=list(FEATURE_COLUMNS),
        target_column=TARGET_COLUMN,
        lookback_days=365,
        censoring_quantiles=[0.1, 0.9],
        max_combination_size=1,
    )


@pytest.fixture
def mock_db():
    """A MagicMock standing in for MQSDBConnector.

    ``bulk_inject_to_db`` returns the typical success payload so the service's
    insert path returns ``len(rows)``.
    """
    db = MagicMock()
    db.bulk_inject_to_db.return_value = {
        "status": "success",
        "message": "Successfully inserted or ignored 3 rows.",
    }
    return db


def _make_engineered_df(tickers: List[str], n_rows: int = 60) -> pd.DataFrame:
    """Build a fake engineered DataFrame for the given tickers."""
    base_dates = pd.date_range("2024-01-01", periods=n_rows, freq="D")
    frames = []
    for ticker in tickers:
        df = pd.DataFrame(
            {
                "timestamp": base_dates,
                "ticker": ticker,
                "open_price": 100.0 + pd.Series(range(n_rows), dtype=float),
                "close_price": 101.0 + pd.Series(range(n_rows), dtype=float),
                "rsi_14": 50.0,
                TARGET_COLUMN: 0.01,
            }
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def fake_engineered_df():
    return _make_engineered_df(TEST_TICKERS)


@pytest.fixture
def fake_grid_df():
    """A small RBI grid - shape doesn't matter, the RBI calc is mocked."""
    return pd.DataFrame(
        {
            "feature": ["close_price", "rsi_14", "open_price"],
            "score": [0.5, 0.3, 0.1],
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_service(config, db, engineered, prediction=0.05, grid_df=None):
    """Construct an RBPForecastService with all heavy collaborators mocked.

    Returns ``(service, mocks_dict)`` so tests can introspect call counts.
    """
    service = RBPForecastService(config=config, db=db)

    # Replace collaborators wired in __init__.
    service.loader = MagicMock()
    service.loader.load = MagicMock(return_value=engineered.copy())

    service.engineer = MagicMock()
    # The real engineer drops NaNs / computes targets; for tests, identity is fine.
    service.engineer.engineer = MagicMock(side_effect=lambda df: df)

    service.predictor = MagicMock()
    service.predictor.predict = MagicMock(
        return_value=(prediction, grid_df if grid_df is not None else pd.DataFrame())
    )

    service.rbi_calc = MagicMock()
    service.rbi_calc.calculate = MagicMock(
        return_value=pd.Series(
            {"close_price": 0.5, "rsi_14": 0.3, "open_price": 0.1}
        )
    )

    return service


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRBPForecastServiceRefresh:
    """Behavioural tests for ``RBPForecastService.refresh``."""

    def test_refresh_inserts_one_row_per_ticker(
        self, rbp_config, mock_db, fake_engineered_df, fake_grid_df
    ):
        """Happy path: 3 tickers in -> 3 rows pushed to bulk_inject_to_db."""
        service = _build_service(
            rbp_config, mock_db, fake_engineered_df, prediction=0.05, grid_df=fake_grid_df
        )

        asof = datetime(2024, 3, 15, 16, 0, 0)
        n_inserted = service.refresh(asof=asof)

        assert n_inserted == 3
        assert mock_db.bulk_inject_to_db.call_count == 1

        call_kwargs = mock_db.bulk_inject_to_db.call_args.kwargs
        assert call_kwargs["table"] == TABLE_NAME
        rows = call_kwargs["data"]
        assert len(rows) == 3

        seen_tickers = {row["ticker"] for row in rows}
        assert seen_tickers == set(TEST_TICKERS)

        for row in rows:
            assert row["horizon_days"] == DEFAULT_HORIZON_DAYS == 21
            assert row["y_pred"] == pytest.approx(0.05)
            assert row["model_version"] == service.model_version
            assert "rbi_top" in row
            assert "generated_at" in row
            assert row["asof"] == asof

    def test_refresh_handles_per_ticker_failure(
        self, rbp_config, mock_db, fake_engineered_df, fake_grid_df
    ):
        """If predictor blows up on one ticker, the batch still completes."""
        service = _build_service(
            rbp_config, mock_db, fake_engineered_df, grid_df=fake_grid_df
        )

        call_counter = {"n": 0}

        def flaky_predict(task_features, x_train, y_train):
            call_counter["n"] += 1
            if call_counter["n"] == 2:
                raise RuntimeError("simulated predictor failure")
            return (0.05, fake_grid_df)

        service.predictor.predict = MagicMock(side_effect=flaky_predict)
        # Bulk insert success payload should reflect the surviving rows.
        mock_db.bulk_inject_to_db.return_value = {
            "status": "success",
            "message": "Successfully inserted or ignored 2 rows.",
        }

        n_inserted = service.refresh(asof=datetime(2024, 3, 15, 16, 0, 0))

        assert n_inserted == 2
        rows = mock_db.bulk_inject_to_db.call_args.kwargs["data"]
        assert len(rows) == 2
        # The failing ticker should NOT be in the persisted rows.
        survivors = {row["ticker"] for row in rows}
        assert len(survivors) == 2
        assert survivors.issubset(set(TEST_TICKERS))

    def test_refresh_skips_when_market_data_empty(self, rbp_config, mock_db):
        """Empty market-data frame => no DB writes, return 0."""
        service = RBPForecastService(config=rbp_config, db=mock_db)
        service.loader = MagicMock()
        service.loader.load = MagicMock(return_value=pd.DataFrame())
        service.engineer = MagicMock()
        # engineer.engineer should never be called because the service short-
        # circuits on an empty load(); guard with a side-effect that fails loudly.
        service.engineer.engineer = MagicMock(
            side_effect=AssertionError("engineer should not be called for empty data")
        )

        n_inserted = service.refresh(asof=datetime(2024, 3, 15, 16, 0, 0))

        assert n_inserted == 0
        mock_db.bulk_inject_to_db.assert_not_called()

    def test_rbi_top_is_json_wrapped(
        self, rbp_config, mock_db, fake_engineered_df, fake_grid_df
    ):
        """The ``rbi_top`` field MUST be wrapped in psycopg2.extras.Json.

        Without an adapter, psycopg2 cannot bind a ``dict`` to a JSONB column
        and the insert dies in production. This guards the regression.
        """
        from psycopg2.extras import Json

        service = _build_service(
            rbp_config, mock_db, fake_engineered_df, grid_df=fake_grid_df
        )

        service.refresh(asof=datetime(2024, 3, 15, 16, 0, 0))

        rows = mock_db.bulk_inject_to_db.call_args.kwargs["data"]
        assert rows, "expected at least one row to inspect"
        for row in rows:
            assert isinstance(row["rbi_top"], Json), (
                f"rbi_top must be wrapped in psycopg2.extras.Json (got "
                f"{type(row['rbi_top']).__name__}); JSONB binding will fail "
                f"in production without it."
            )
            # The wrapped payload should be a dict of feature -> score.
            payload = row["rbi_top"].adapted
            assert isinstance(payload, dict)
            for name, value in payload.items():
                assert isinstance(name, str)
                assert isinstance(value, float)

    def test_caching_skips_redundant_training_rebuild(
        self, rbp_config, mock_db, fake_engineered_df, fake_grid_df
    ):
        """Two refreshes with the same data-end date => training matrix is built
        once per ticker, not twice.

        The service caches ``(x_train, y_train, training_end_date)`` in
        ``self._train_cache``; ``_get_or_build_train`` returns the cached tuple
        when ``latest_date`` matches. We verify by spying on that helper.
        """
        service = _build_service(
            rbp_config, mock_db, fake_engineered_df, grid_df=fake_grid_df
        )

        original_builder = service._get_or_build_train
        rebuild_count = {"n": 0}

        def counting_builder(ticker, train_df):
            cached = service._train_cache.get(ticker)
            latest_date = pd.to_datetime(train_df["timestamp"].iloc[-1]).date()
            if cached is None or cached[2] != latest_date:
                rebuild_count["n"] += 1
            return original_builder(ticker, train_df)

        with patch.object(
            service, "_get_or_build_train", side_effect=counting_builder
        ):
            asof = datetime(2024, 3, 15, 16, 0, 0)
            service.refresh(asof=asof)
            first_pass = rebuild_count["n"]
            assert first_pass == len(TEST_TICKERS), (
                "first refresh should rebuild training matrix once per ticker"
            )

            # Second refresh with the same asof / same engineered data.
            service.refresh(asof=asof)
            second_pass = rebuild_count["n"]

        # No additional rebuilds should have occurred on the warm pass.
        assert second_pass == first_pass, (
            f"expected cache hits on second refresh; saw "
            f"{second_pass - first_pass} extra rebuilds"
        )

        # Sanity: the cache should be populated for every ticker.
        assert set(service._train_cache.keys()) == set(TEST_TICKERS)


class TestRBPForecastServiceConstruction:
    """Light sanity tests around construction + model_version stamping."""

    def test_model_version_is_stable_string(self, rbp_config, mock_db):
        service = RBPForecastService(config=rbp_config, db=mock_db)
        assert isinstance(service.model_version, str)
        assert service.model_version.startswith("rbp_v1_combo")
        # Encodes both combo size and quantile count.
        assert "combo1" in service.model_version
        assert "q2" in service.model_version

    def test_refresh_with_no_tickers_returns_zero(self, mock_db):
        empty_config = RBPConfig(tickers=[], feature_columns=list(FEATURE_COLUMNS))
        service = RBPForecastService(config=empty_config, db=mock_db)
        # loader / engineer shouldn't be reached, but mock them to be safe.
        service.loader = MagicMock()
        service.engineer = MagicMock()

        assert service.refresh(asof=datetime(2024, 3, 15)) == 0
        mock_db.bulk_inject_to_db.assert_not_called()
        service.loader.load.assert_not_called()
