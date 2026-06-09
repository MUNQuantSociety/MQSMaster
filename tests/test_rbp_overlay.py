"""tests/test_rbp_overlay.py

Unit tests for ``src/risk_manager/rbp_overlay.py``.

The overlay must:
  * NEVER raise (fail-open contract)
  * Pass-through original confidence whenever it is disabled, the portfolio is
    opted-out, the DB has no fresh forecast, or the forecast is NaN
  * Apply the documented blend math (1 - w)*conf + w * tanh(|y_pred|*scale)
    only when the RBP forecast agrees with the strategy's directional call
  * Honor the in-memory per-ticker TTL cache so we don't hammer the DB

All tests use ``unittest.mock.MagicMock`` for the ``MQSDBConnector`` -- no real
database is required.  The DB seam is the ``execute_query`` method, mirroring
what the production overlay calls.

Contract assumption: the overlay constructs its own connector-typed object via
``RBPOverlay(db, cfg)`` and only ever invokes ``db.execute_query(sql, params,
fetch=True)``.  The returned dict shape matches the rest of the codebase:
``{"status": "success"|"error", "data": [{"y_pred": float, "asof": ...}, ...]}``.
"""

from __future__ import annotations

import math
import os
import sys
from unittest.mock import MagicMock

import pytest

# The overlay module imports ``from common.database.MQSDBConnector import
# MQSDBConnector`` (project convention -- ``src`` is treated as a package
# root).  Ensure ``src`` is importable so the module loads under bare pytest.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from src.risk_manager.rbp_overlay import RBPOverlay  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _ok(y_pred: float) -> dict:
    """Build a successful execute_query response carrying a single row."""
    return {
        "status": "success",
        "data": [{"y_pred": y_pred, "asof": "2026-05-21T12:00:00Z"}],
    }


def _empty_ok() -> dict:
    """Successful query that returned zero rows (the 'DB miss' case)."""
    return {"status": "success", "data": []}


def _base_cfg(**overrides) -> dict:
    """Default cfg matching the production defaults; override per test."""
    cfg = {
        "enabled": True,
        "blend_weight": 0.10,
        "tanh_scale": 20.0,
        "stale_after_hours": 24,
        "disabled_portfolios": [],
        "cache_ttl_seconds": 60,
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def mock_db():
    """Fresh MagicMock DB connector per test (no shared state).

    The overlay constructor runs an existence check (``SELECT to_regclass(...)``)
    that must report the ``rbp_forecasts`` table as present, otherwise the
    overlay self-disables and every legacy test trips.  We install a
    ``side_effect`` dispatcher that:
      * returns a table-exists payload for the ``to_regclass`` probe; and
      * defers to ``execute_query.return_value`` for every other SQL string,
        preserving the historical contract where individual tests just set
        ``db.execute_query.return_value = ...`` to inject a forecast row.

    Tests that need to override the existence-check behavior should construct
    their own dispatcher via ``_dispatching_db`` (see below).
    """
    db = MagicMock()
    db.execute_query = MagicMock()
    db.execute_query.return_value = _empty_ok()

    _table_exists_payload = {"status": "success", "data": [{"t": "rbp_forecasts"}]}

    def _dispatch(sql, *args, **kwargs):
        if "to_regclass" in sql:
            return _table_exists_payload
        return db.execute_query.return_value

    db.execute_query.side_effect = _dispatch
    return db


@pytest.fixture
def make_overlay(mock_db):
    """Factory that builds an RBPOverlay from cfg overrides.

    Returns the (overlay, db) tuple so tests can assert on db.execute_query
    call counts/args.
    """
    def _factory(**cfg_overrides):
        cfg = _base_cfg(**cfg_overrides)
        overlay = RBPOverlay(mock_db, cfg)
        # The constructor consumes one ``to_regclass`` existence-check call.
        # Reset the call log so per-test ``assert_called_once`` / ``call_count``
        # assertions only count the forecast-fetch calls under test.
        mock_db.execute_query.reset_mock()
        return overlay, mock_db
    return _factory


# ---------------------------------------------------------------------------
# 1. Disabled / opt-out paths -- DB must not be touched
# ---------------------------------------------------------------------------

def test_overlay_disabled_returns_original(make_overlay):
    """cfg.enabled=False -> pass-through, no DB call."""
    overlay, db = make_overlay(enabled=False)

    result = overlay("1", "AAPL", "BUY", 0.42)

    assert result == 0.42
    db.execute_query.assert_not_called()


def test_disabled_portfolio_returns_original(make_overlay):
    """portfolio_id in disabled_portfolios -> pass-through, no DB call."""
    overlay, db = make_overlay(disabled_portfolios=["8"])

    result = overlay("8", "AAPL", "BUY", 0.7)

    assert result == 0.7
    db.execute_query.assert_not_called()


def test_hold_signal_returns_original(make_overlay):
    """Non-directional signal (HOLD) -> pass-through, no DB call.

    The overlay can only blend BUY/SELL because directional agreement is
    undefined for HOLD.  It must not even consult the DB.
    """
    overlay, db = make_overlay()

    result = overlay("1", "AAPL", "HOLD", 0.5)

    assert result == 0.5
    db.execute_query.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Happy-path blending math
# ---------------------------------------------------------------------------

def test_buy_with_positive_forecast_blends_up(make_overlay):
    """BUY + y_pred>0 -> agree -> conf nudged toward rbp_mag.

    With y_pred=+0.02, tanh_scale=20, blend_weight=0.10:
        rbp_mag = tanh(0.02 * 20)  = tanh(0.4) ~= 0.379949
        new_conf = 0.9*0.5 + 0.1*0.379949 ~= 0.487995
    """
    overlay, db = make_overlay()
    db.execute_query.return_value = _ok(0.02)

    result = overlay("1", "AAPL", "BUY", 0.5)

    expected_rbp_mag = math.tanh(0.4)
    expected = 0.9 * 0.5 + 0.1 * expected_rbp_mag
    assert result == pytest.approx(expected, abs=1e-4)
    # rbp_mag (~0.38) is below conf (0.5), so blending pulls conf DOWN even
    # though the signal agrees.  The important contract is that 'agree' uses
    # the magnitude (non-zero) instead of zero -- which we cross-check by
    # asserting the result is STRICTLY higher than the disagreeing baseline
    # (0.9*0.5 + 0.1*0.0 = 0.45).
    assert result > 0.45
    db.execute_query.assert_called_once()


def test_buy_with_negative_forecast_shrinks(make_overlay):
    """BUY + y_pred<0 -> DISAGREE -> rbp_conf=0 -> conf decays toward 0.

    With blend_weight=0.10 and conf=0.5:
        new_conf = 0.9*0.5 + 0.1*0.0 = 0.45
    """
    overlay, db = make_overlay()
    db.execute_query.return_value = _ok(-0.02)

    result = overlay("1", "AAPL", "BUY", 0.5)

    assert result == pytest.approx(0.45, abs=1e-4)
    assert result < 0.5  # shrunk because forecast disagrees
    db.execute_query.assert_called_once()


def test_sell_with_negative_forecast_blends_up(make_overlay):
    """SELL + y_pred<0 -> agree -> conf nudged up.

    rbp_mag = |tanh(-0.02 * 20)| = |tanh(-0.4)| = tanh(0.4) ~= 0.379949
        new_conf = 0.9*0.5 + 0.1*0.379949 ~= 0.487995
    """
    overlay, db = make_overlay()
    db.execute_query.return_value = _ok(-0.02)

    result = overlay("1", "AAPL", "SELL", 0.5)

    expected = 0.9 * 0.5 + 0.1 * math.tanh(0.4)
    assert result == pytest.approx(expected, abs=1e-4)
    # Cross-check: agreeing case must end up STRICTLY higher than the
    # disagreeing case (0.45), even though both can sit below conf=0.5.
    assert result > 0.45
    db.execute_query.assert_called_once()


def test_sell_with_positive_forecast_shrinks(make_overlay):
    """SELL + y_pred>0 -> DISAGREE -> rbp_conf=0 -> conf shrinks.

    Symmetric counterpart to ``test_buy_with_negative_forecast_shrinks``;
    documents that disagreement is checked on signed direction, not magnitude.
    """
    overlay, db = make_overlay()
    db.execute_query.return_value = _ok(0.02)

    result = overlay("1", "AAPL", "SELL", 0.5)

    assert result == pytest.approx(0.45, abs=1e-4)
    assert result < 0.5


# ---------------------------------------------------------------------------
# 3. DB failure modes -- never raises, always returns original
# ---------------------------------------------------------------------------

def test_db_miss_returns_original(make_overlay):
    """DB returns status=success but data=[] -> pass-through."""
    overlay, db = make_overlay()
    db.execute_query.return_value = _empty_ok()

    result = overlay("1", "AAPL", "BUY", 0.5)

    assert result == 0.5


def test_db_error_status_returns_original(make_overlay):
    """DB returns status=error -> pass-through (treated like a miss)."""
    overlay, db = make_overlay()
    db.execute_query.return_value = {"status": "error", "data": []}

    result = overlay("1", "AAPL", "BUY", 0.5)

    assert result == 0.5


def test_db_raises_returns_original(make_overlay):
    """DB raises (psycopg2.OperationalError etc.) -> swallowed, pass-through.

    This is the fail-open contract: under no circumstances may the overlay
    bubble up an exception into the live executor.
    """
    overlay, db = make_overlay()
    db.execute_query.side_effect = RuntimeError("connection refused")

    # Must not raise.
    result = overlay("1", "AAPL", "BUY", 0.5)

    assert result == 0.5


def test_nan_forecast_returns_original(make_overlay):
    """y_pred=NaN in the row -> pass-through (no blending on garbage)."""
    overlay, db = make_overlay()
    db.execute_query.return_value = _ok(float("nan"))

    result = overlay("1", "AAPL", "BUY", 0.5)

    assert result == 0.5


# ---------------------------------------------------------------------------
# 4. In-memory TTL cache
# ---------------------------------------------------------------------------

def test_cache_hit_skips_db(make_overlay):
    """Two consecutive calls within TTL -> DB executed once.

    The second call must serve from the in-memory ticker cache.
    """
    overlay, db = make_overlay(cache_ttl_seconds=60)
    db.execute_query.return_value = _ok(0.02)

    r1 = overlay("1", "AAPL", "BUY", 0.5)
    r2 = overlay("1", "AAPL", "BUY", 0.5)

    assert db.execute_query.call_count == 1
    # Same inputs -> deterministic output.
    assert r1 == pytest.approx(r2, abs=1e-9)


def test_cache_miss_after_ttl(make_overlay):
    """cache_ttl_seconds=0 -> every call refetches."""
    overlay, db = make_overlay(cache_ttl_seconds=0)
    db.execute_query.return_value = _ok(0.02)

    overlay("1", "AAPL", "BUY", 0.5)
    overlay("1", "AAPL", "BUY", 0.5)

    assert db.execute_query.call_count == 2


def test_cache_isolated_per_ticker(make_overlay):
    """Different tickers must each hit the DB once (cache is per-ticker)."""
    overlay, db = make_overlay(cache_ttl_seconds=60)
    db.execute_query.return_value = _ok(0.02)

    overlay("1", "AAPL", "BUY", 0.5)
    overlay("1", "MSFT", "BUY", 0.5)
    overlay("1", "AAPL", "BUY", 0.5)  # cache hit on AAPL

    assert db.execute_query.call_count == 2


def test_cached_miss_does_not_rehit_db(make_overlay):
    """An empty-result lookup is cached too -- so the next call doesn't re-hit.

    This prevents hammering the DB for tickers with no recent forecast.
    """
    overlay, db = make_overlay(cache_ttl_seconds=60)
    db.execute_query.return_value = _empty_ok()

    r1 = overlay("1", "AAPL", "BUY", 0.5)
    r2 = overlay("1", "AAPL", "BUY", 0.5)

    assert r1 == 0.5
    assert r2 == 0.5
    # The miss-sentinel must be cached -- exactly one DB call.
    assert db.execute_query.call_count == 1


# ---------------------------------------------------------------------------
# 5. Clamp + pathological inputs
# ---------------------------------------------------------------------------

def test_clamps_to_unit_interval(make_overlay):
    """Pathological cfg (negative blend_weight) -> still clamped to [0, 1].

    A negative blend_weight with a positive rbp_conf could theoretically push
    new_conf below 0; the final result must still respect the unit interval.
    """
    overlay, db = make_overlay(blend_weight=-5.0)
    db.execute_query.return_value = _ok(0.02)

    result = overlay("1", "AAPL", "BUY", 0.5)

    assert 0.0 <= result <= 1.0


def test_clamps_upper_bound(make_overlay):
    """Huge positive blend_weight on agreeing signal -> capped at 1.0."""
    # blend_weight=5 -> (1-5)*0.9 + 5*~0.38 = -3.6 + 1.9 = -1.7 (will clamp to 0)
    # Use a setup that drives the un-clamped result well past 1.0:
    # blend_weight = -1.0, conf = 1.0, rbp_conf = 0.5
    #   raw = (1 - (-1))*1.0 + (-1)*0.5 = 2.0 - 0.5 = 1.5 -> clamp to 1.0
    overlay, db = make_overlay(blend_weight=-1.0)
    db.execute_query.return_value = _ok(0.02)

    result = overlay("1", "AAPL", "BUY", 1.0)

    assert 0.0 <= result <= 1.0
    assert result == pytest.approx(1.0, abs=1e-9)


def test_zero_blend_weight_is_passthrough(make_overlay):
    """blend_weight=0 -> overlay is mathematically a no-op even when agreeing."""
    overlay, db = make_overlay(blend_weight=0.0)
    db.execute_query.return_value = _ok(0.02)

    result = overlay("1", "AAPL", "BUY", 0.5)

    assert result == pytest.approx(0.5, abs=1e-9)


def test_full_blend_weight_replaces_confidence(make_overlay):
    """blend_weight=1.0 -> new_conf == rbp_conf entirely.

    Useful sanity check that the blend math truly is a convex combo and not
    something else (e.g. additive).
    """
    overlay, db = make_overlay(blend_weight=1.0)
    db.execute_query.return_value = _ok(0.02)

    result = overlay("1", "AAPL", "BUY", 0.5)

    expected_rbp_mag = math.tanh(0.4)
    assert result == pytest.approx(expected_rbp_mag, abs=1e-6)


# ---------------------------------------------------------------------------
# 6. Schema-missing self-disable behavior
# ---------------------------------------------------------------------------

def _dispatching_db(to_regclass_response, forecast_response=None):
    """Build a MagicMock DB whose execute_query routes by SQL string.

    The overlay issues two distinct queries:
      * ``SELECT to_regclass('public.rbp_forecasts') AS t`` (constructor)
      * ``LATEST_FORECAST_QUERY`` (per-call forecast lookup)

    This helper dispatches the right canned response to each so tests can
    drive the existence check independently from the forecast fetch.
    """
    db = MagicMock()

    def _side_effect(sql, *args, **kwargs):
        if "to_regclass" in sql:
            if isinstance(to_regclass_response, Exception):
                raise to_regclass_response
            return to_regclass_response
        return forecast_response if forecast_response is not None else _empty_ok()

    db.execute_query = MagicMock(side_effect=_side_effect)
    return db


def test_self_disables_when_table_missing(caplog):
    """to_regclass returns NULL -> overlay self-disables and warns once."""
    db = _dispatching_db(
        to_regclass_response={"status": "success", "data": [{"t": None}]},
    )

    with caplog.at_level("WARNING", logger="RBPOverlay"):
        overlay = RBPOverlay(db, _base_cfg(enabled=True))

    assert overlay.enabled is False
    # Exactly one warning emitted by the overlay during construction.
    overlay_warnings = [
        rec for rec in caplog.records
        if rec.name == "RBPOverlay" and rec.levelname == "WARNING"
    ]
    assert len(overlay_warnings) == 1
    assert "rbp_forecasts" in overlay_warnings[0].getMessage()


def test_self_disables_when_existence_check_errors():
    """to_regclass returns status=error -> overlay self-disables (fail-safe)."""
    db = _dispatching_db(
        to_regclass_response={"status": "error", "message": "boom"},
    )

    overlay = RBPOverlay(db, _base_cfg(enabled=True))

    assert overlay.enabled is False


def test_stays_enabled_when_table_exists():
    """to_regclass returns the table name -> overlay stays enabled and blends.

    Confirms the existence check is permissive when the schema is present:
    construction does not flip ``enabled`` off, and a subsequent ``__call__``
    routes through the normal LATEST_FORECAST_QUERY path and applies the
    documented blend math.
    """
    db = _dispatching_db(
        to_regclass_response={"status": "success", "data": [{"t": "rbp_forecasts"}]},
        forecast_response=_ok(0.02),
    )

    overlay = RBPOverlay(db, _base_cfg(enabled=True))

    assert overlay.enabled is True

    result = overlay("1", "AAPL", "BUY", 0.5)

    # Same math as test_buy_with_positive_forecast_blends_up: agreeing BUY
    # with y_pred=+0.02 and default blend_weight=0.10, tanh_scale=20.0.
    expected = 0.9 * 0.5 + 0.1 * math.tanh(0.4)
    assert result == pytest.approx(expected, abs=1e-6)
    # Existence check + forecast fetch = exactly two DB hits.
    assert db.execute_query.call_count == 2
