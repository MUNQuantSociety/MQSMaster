"""
RBP end-to-end integration smoke test.

Exercises the full RBP pipeline wiring:
    RBPForecastService.refresh()  ->  rbp_forecasts (mock DB)
        |
        v
    RBPOverlay.__call__()         <-- reads rbp_forecasts via mock DB
        |
        v
    tradeExecutor.execute_trade() <-- routes confidence through overlay

The point is to validate WIRING (which SQL fires, which JSON gets wrapped,
which confidence value lands inside the executor), not the SQL itself or
the RBP math. All numeric internals (MarketDataLoader, RBPPredictor,
RBICalculator) are mocked. The DB is a single fake that captures every
SQL call so we can assert on it.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import pytz

# tradeExecutor constructs FMPMarketData on init, which validates FMP_API_KEY.
# Tag the whole module so the unit-test gate (which filters `not api`) skips
# it on hosts without the key (e.g., main.yml unit step).
pytestmark = [pytest.mark.integration, pytest.mark.api]

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psycopg2.extras import Json  # noqa: E402

from RBP.config import RBPConfig  # noqa: E402
from RBP.service import RBPForecastService, TABLE_NAME  # noqa: E402
from src.risk_manager.rbp_overlay import RBPOverlay  # noqa: E402


NY = pytz.timezone("America/New_York")
TEST_TICKERS = ["AAPL", "TSLA"]
RBP_CFG: Dict[str, Any] = {
    "enabled": True,
    "blend_weight": 0.10,
    "tanh_scale": 20.0,
    "stale_after_hours": 24,
    "cache_ttl_seconds": 60,
    "disabled_portfolios": ["5", "8"],
}


# --------------------------------------------------------------------------- #
# Fake DB: captures every SQL call and is fully steerable per-test.
# --------------------------------------------------------------------------- #


class FakeDB:
    """In-memory stand-in for MQSDBConnector.

    Records every call so tests can assert on the wiring. Each handler can
    be overridden per-test by reassigning the `*_handler` attributes.
    """

    def __init__(self) -> None:
        self.bulk_insert_calls: List[Dict[str, Any]] = []
        self.execute_query_calls: List[Tuple[str, Any, bool]] = []
        # Default: pretend the SELECT returns nothing.
        self.query_handler = lambda sql, values, fetch: {
            "status": "success",
            "data": [],
        }
        self.bulk_handler = lambda table, data, conflict_columns: {
            "status": "success",
            "message": f"Successfully inserted or ignored {len(data)} rows.",
        }

    # --- methods used by RBPForecastService -------------------------------- #
    def bulk_inject_to_db(
        self,
        table: str,
        data: List[Dict[str, Any]],
        conflict_columns: List[str],
    ) -> Dict[str, Any]:
        self.bulk_insert_calls.append(
            {"table": table, "data": data, "conflict_columns": conflict_columns}
        )
        return self.bulk_handler(table, data, conflict_columns)

    # --- methods used by RBPOverlay ---------------------------------------- #
    def execute_query(
        self, sql: str, values: Any = None, fetch: bool = False
    ) -> Dict[str, Any]:
        self.execute_query_calls.append((sql, values, fetch))
        # Intercept the overlay's table-existence probe so it never reaches
        # the per-test query_handler (which may raise or return empty).
        if "to_regclass" in sql:
            return {"status": "success", "data": [{"t": "rbp_forecasts"}]}
        return self.query_handler(sql, values, fetch)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_db() -> FakeDB:
    return FakeDB()


@pytest.fixture
def rbp_config() -> RBPConfig:
    return RBPConfig(tickers=list(TEST_TICKERS))


def _make_engineered_frame(tickers: List[str], feature_columns: List[str]) -> pd.DataFrame:
    """Build a tiny engineered DataFrame with enough rows for training+predict."""
    rows: List[Dict[str, Any]] = []
    base = pd.Timestamp("2024-01-01")
    for t in tickers:
        for i in range(30):
            row: Dict[str, Any] = {
                "ticker": t,
                "timestamp": base + pd.Timedelta(days=i),
                "close_price": 100.0 + i,
            }
            # All features and the target column the predictor needs.
            for col in feature_columns:
                row[col] = 0.01 * (i + 1)
            row["target_return_21d"] = 0.02
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def service(
    fake_db: FakeDB, rbp_config: RBPConfig
) -> RBPForecastService:
    """Build the service with internal numeric machinery mocked out."""
    svc = RBPForecastService(config=rbp_config, db=fake_db)

    engineered = _make_engineered_frame(TEST_TICKERS, rbp_config.feature_columns)
    svc.loader = MagicMock()
    svc.loader.load.return_value = engineered

    svc.engineer = MagicMock()
    svc.engineer.engineer.return_value = engineered

    # predictor.predict -> (prediction, grid_df)
    svc.predictor = MagicMock()
    svc.predictor.predict.return_value = (
        0.05,
        pd.DataFrame({c: [0.0] for c in rbp_config.feature_columns}),
    )

    # rbi_calc.calculate -> a small Series of feature scores
    svc.rbi_calc = MagicMock()
    svc.rbi_calc.calculate.return_value = pd.Series(
        {c: 0.1 for c in rbp_config.feature_columns[:5]}
    )
    return svc


# --------------------------------------------------------------------------- #
# Scenario 1: full pipeline smoke
# --------------------------------------------------------------------------- #


def test_full_pipeline_smoke(fake_db: FakeDB, service: RBPForecastService) -> None:
    """RBP refresh -> rbp_forecasts -> RBPOverlay -> tradeExecutor.

    Validates the cross-component wiring end to end.
    """
    # ---- Step B: refresh() writes well-formed rows to rbp_forecasts. ------
    asof = datetime(2024, 2, 1, 16, 0, tzinfo=NY)
    n_inserted = service.refresh(asof=asof)

    assert n_inserted == len(TEST_TICKERS), (
        f"expected {len(TEST_TICKERS)} inserts, got {n_inserted}"
    )
    assert len(fake_db.bulk_insert_calls) == 1, "bulk insert should fire exactly once"

    call = fake_db.bulk_insert_calls[0]
    assert call["table"] == TABLE_NAME == "rbp_forecasts"
    assert call["conflict_columns"] == [
        "ticker",
        "asof",
        "horizon_days",
        "model_version",
    ]
    rows = call["data"]
    assert {r["ticker"] for r in rows} == set(TEST_TICKERS)
    for row in rows:
        assert set(row.keys()) >= {
            "ticker",
            "asof",
            "horizon_days",
            "y_pred",
            "rbi_top",
            "model_version",
            "generated_at",
        }
        assert row["horizon_days"] == 21
        assert isinstance(row["y_pred"], float)
        # JSONB wrap: rbi_top must be a psycopg2 Json adapter, not a raw dict.
        assert isinstance(row["rbi_top"], Json), (
            f"rbi_top must be wrapped in psycopg2 Json, got {type(row['rbi_top'])}"
        )
        assert row["model_version"].startswith("rbp_v1_combo")

    # ---- Step C: make DB now return forecasts for the SELECT. -------------
    aapl_y_pred = 0.05  # matches predictor mock

    def select_handler(sql: str, values: Any, fetch: bool) -> Dict[str, Any]:
        assert "rbp_forecasts" in sql
        assert fetch is True
        ticker = values[0]
        if ticker == "AAPL":
            return {
                "status": "success",
                "data": [{"y_pred": aapl_y_pred, "asof": asof}],
            }
        return {"status": "success", "data": []}

    fake_db.query_handler = select_handler

    # ---- Step D: RBPOverlay blends forecast into confidence. --------------
    overlay = RBPOverlay(fake_db, RBP_CFG)
    original_conf = 0.5
    blended = overlay("1", "AAPL", "BUY", original_conf)

    # Expected math (mirrors RBPOverlay.__call__):
    rbp_mag = abs(math.tanh(aapl_y_pred * RBP_CFG["tanh_scale"]))
    expected = (1.0 - RBP_CFG["blend_weight"]) * original_conf + RBP_CFG[
        "blend_weight"
    ] * rbp_mag
    expected = max(0.0, min(1.0, expected))

    assert blended == pytest.approx(expected, abs=1e-9), (
        f"overlay produced {blended}, expected {expected}"
    )
    # And the SELECT must have actually fired.
    assert any("rbp_forecasts" in sql for sql, _, _ in fake_db.execute_query_calls)

    # ---- Step E: tradeExecutor must apply the SAME overlay value. ---------
    captured: Dict[str, float] = {}

    def fake_update_database(
        self: Any,
        portfolio_id: Any,
        ticker: str,
        signal_type: str,
        quantity_to_trade: int,
        updated_cash: float,
        updated_quantity: float,
        arrival_price: float,
        exec_price: float,
        slippage_bps: float,
        timestamp: Any,
        port_notional: float,
    ) -> Dict[str, Any]:
        # Recover the confidence the executor used: with target_notional
        # large and positions empty, desired = confidence * port_notional.
        captured["quantity"] = float(quantity_to_trade)
        captured["exec_price"] = float(exec_price)
        return {"status": "success"}

    from src.live_trading.executor import tradeExecutor

    with patch.object(tradeExecutor, "get_current_price", return_value=100.0), patch.object(
        tradeExecutor, "update_database", new=fake_update_database
    ):
        executor = tradeExecutor(db_connector=fake_db, rbp_overlay=overlay)
        # Clear the overlay's cache so this call re-reads the FakeDB and
        # uses the SAME blended value computed in Step D.
        overlay._cache.clear()

        positions = pd.DataFrame(columns=["ticker", "quantity"])
        executor.execute_trade(
            portfolio_id="1",
            ticker="AAPL",
            signal_type="BUY",
            confidence=original_conf,
            arrival_price=100.0,
            cash=1_000_000.0,
            positions=positions,
            port_notional=1_000_000.0,
            ticker_weight=1.0,
            timestamp=asof,
        )

    # With port_notional=$1M, weight=1.0, no current position, price=$100:
    #   desired_trade_notional = 1_000_000 * blended_confidence
    #   quantity_to_trade      = floor(desired / 100)
    expected_qty = math.floor(1_000_000.0 * expected / 100.0)
    assert "quantity" in captured, "update_database was not called"
    assert captured["quantity"] == expected_qty, (
        f"executor confidence drifted: qty={captured['quantity']}, expected={expected_qty}"
    )


# --------------------------------------------------------------------------- #
# Scenario 2: disabled portfolio bypasses overlay end-to-end
# --------------------------------------------------------------------------- #


def test_disabled_portfolio_bypasses_overlay_end_to_end(fake_db: FakeDB) -> None:
    """portfolio_id='8' is in disabled_portfolios -> overlay is a no-op."""
    asof = datetime(2024, 2, 1, 16, 0, tzinfo=NY)

    # If the overlay (incorrectly) tried to query, this would record a call;
    # we assert below that it stayed empty.
    fake_db.query_handler = lambda sql, values, fetch: {
        "status": "success",
        "data": [{"y_pred": 0.99, "asof": asof}],  # would massively boost conf
    }

    overlay = RBPOverlay(fake_db, RBP_CFG)
    # Snapshot DB-call count AFTER overlay construction (which may probe
    # for the forecasts table via to_regclass) but BEFORE any overlay
    # invocation or trade. The semantic claim is: the trade itself, not
    # the constructor, must not touch the DB for disabled portfolios.
    calls_after_init = len(fake_db.execute_query_calls)
    original_conf = 0.42

    # Direct overlay assertion: disabled => exact pass-through.
    assert overlay("8", "AAPL", "BUY", original_conf) == original_conf
    assert (
        len(fake_db.execute_query_calls) == calls_after_init
    ), "disabled portfolios must short-circuit BEFORE the DB SELECT"

    # And confirm the executor honors that pass-through end to end.
    captured: Dict[str, float] = {}

    def fake_update_database(
        self: Any,
        portfolio_id: Any,
        ticker: str,
        signal_type: str,
        quantity_to_trade: int,
        updated_cash: float,
        updated_quantity: float,
        arrival_price: float,
        exec_price: float,
        slippage_bps: float,
        timestamp: Any,
        port_notional: float,
    ) -> Dict[str, Any]:
        captured["quantity"] = float(quantity_to_trade)
        return {"status": "success"}

    from src.live_trading.executor import tradeExecutor

    with patch.object(tradeExecutor, "get_current_price", return_value=100.0), patch.object(
        tradeExecutor, "update_database", new=fake_update_database
    ):
        executor = tradeExecutor(db_connector=fake_db, rbp_overlay=overlay)
        positions = pd.DataFrame(columns=["ticker", "quantity"])
        executor.execute_trade(
            portfolio_id="8",
            ticker="AAPL",
            signal_type="BUY",
            confidence=original_conf,
            arrival_price=100.0,
            cash=1_000_000.0,
            positions=positions,
            port_notional=1_000_000.0,
            ticker_weight=1.0,
            timestamp=asof,
        )

    # confidence unchanged -> qty = floor(1_000_000 * 0.42 / 100) = 4200
    expected_qty = math.floor(1_000_000.0 * original_conf / 100.0)
    assert captured["quantity"] == expected_qty, (
        f"disabled portfolio confidence was perturbed: "
        f"qty={captured['quantity']}, expected={expected_qty}"
    )
    # And still no SELECT against rbp_forecasts from the trade path.
    assert len(fake_db.execute_query_calls) == calls_after_init


# --------------------------------------------------------------------------- #
# Scenario 3: overlay DB failure must NOT break trade execution
# --------------------------------------------------------------------------- #


def test_overlay_failure_does_not_break_execution(fake_db: FakeDB) -> None:
    """If the overlay's SELECT raises, the trade still proceeds at the original confidence."""
    asof = datetime(2024, 2, 1, 16, 0, tzinfo=NY)

    def boom(sql: str, values: Any, fetch: bool) -> Dict[str, Any]:
        raise RuntimeError("simulated DB outage on rbp_forecasts SELECT")

    fake_db.query_handler = boom

    overlay = RBPOverlay(fake_db, RBP_CFG)
    original_conf = 0.5

    # Direct overlay: must not raise; must pass confidence through.
    blended = overlay("1", "AAPL", "BUY", original_conf)
    assert blended == original_conf, (
        f"overlay should fail-open on DB error, got {blended}"
    )

    # End-to-end: executor proceeds normally.
    captured: Dict[str, float] = {}

    def fake_update_database(
        self: Any,
        portfolio_id: Any,
        ticker: str,
        signal_type: str,
        quantity_to_trade: int,
        updated_cash: float,
        updated_quantity: float,
        arrival_price: float,
        exec_price: float,
        slippage_bps: float,
        timestamp: Any,
        port_notional: float,
    ) -> Dict[str, Any]:
        captured["quantity"] = float(quantity_to_trade)
        return {"status": "success"}

    from src.live_trading.executor import tradeExecutor

    with patch.object(tradeExecutor, "get_current_price", return_value=100.0), patch.object(
        tradeExecutor, "update_database", new=fake_update_database
    ):
        executor = tradeExecutor(db_connector=fake_db, rbp_overlay=overlay)
        # Clear the cached NaN sentinel from the direct-call above so the
        # executor path actually re-hits (and re-raises through) the
        # FakeDB, exercising the fail-open codepath inside tradeExecutor.
        overlay._cache.clear()

        positions = pd.DataFrame(columns=["ticker", "quantity"])
        # Must not raise even though the overlay's underlying SELECT explodes.
        executor.execute_trade(
            portfolio_id="1",
            ticker="AAPL",
            signal_type="BUY",
            confidence=original_conf,
            arrival_price=100.0,
            cash=1_000_000.0,
            positions=positions,
            port_notional=1_000_000.0,
            ticker_weight=1.0,
            timestamp=asof,
        )

    expected_qty = math.floor(1_000_000.0 * original_conf / 100.0)
    assert "quantity" in captured, "update_database was never reached"
    assert captured["quantity"] == expected_qty, (
        f"trade size drifted under overlay failure: "
        f"qty={captured['quantity']}, expected={expected_qty}"
    )
