"""Unit tests for the market_data retention pruner.

No AWS or DB calls -- should_prune is pure, and run_cycle's boundaries
(SSM client, MQSDBConnector) are exercised via plain mocks/stubs. Exercises
the once-a-year gate logic exhaustively in milliseconds, no year-long wait
needed.
"""

from datetime import date
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from src.orchestrator.retention.prune_market_data import (
    _get_last_run_date,
    _prune_once,
    _today,
    run_cycle,
    should_prune,
)

pytestmark = [pytest.mark.smoke]


# --- should_prune ---------------------------------------------------------


def test_should_prune_never_run_before():
    assert should_prune(None, date(2026, 8, 25)) is True


def test_should_prune_recent_run_skips():
    last_run = date(2026, 8, 1)
    today = date(2026, 8, 25)
    assert should_prune(last_run, today, min_interval_days=365) is False


def test_should_prune_boundary_just_under():
    last_run = date(2025, 1, 1)
    today = date(2025, 12, 31)  # 364 days
    assert should_prune(last_run, today, min_interval_days=365) is False


def test_should_prune_boundary_exact():
    last_run = date(2025, 1, 1)
    today = date(2026, 1, 1)  # 365 days
    assert should_prune(last_run, today, min_interval_days=365) is True


def test_should_prune_well_over():
    last_run = date(2020, 1, 1)
    today = date(2026, 8, 25)
    assert should_prune(last_run, today, min_interval_days=365) is True


# --- _get_last_run_date -----------------------------------------------------


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "x"}}, "GetParameter")


def test_get_last_run_date_parameter_not_found_means_never_ran():
    ssm = MagicMock()
    ssm.get_parameter.side_effect = _client_error("ParameterNotFound")

    result, ok = _get_last_run_date(ssm)

    assert result is None
    assert ok is True


def test_get_last_run_date_transient_error_skips_not_prunes():
    ssm = MagicMock()
    ssm.get_parameter.side_effect = _client_error("ThrottlingException")

    result, ok = _get_last_run_date(ssm)

    # Must NOT look like "never ran" -- that would trigger a full prune
    # attempt on a transient AWS hiccup.
    assert result is None
    assert ok is False


def test_get_last_run_date_parses_stored_value():
    ssm = MagicMock()
    ssm.get_parameter.return_value = {"Parameter": {"Value": "2026-01-15"}}

    result, ok = _get_last_run_date(ssm)

    assert result == date(2026, 1, 15)
    assert ok is True


# --- _prune_once batching ---------------------------------------------------


def _batches_stub(batches):
    """Returns an execute_query stand-in that yields one canned batch per call."""
    calls = {"n": 0}

    def fake_execute_query(sql, params, fetch=False):
        batch = batches[calls["n"]]
        calls["n"] += 1
        return {"status": "success", "data": [{"id": i} for i in range(batch)]}

    return fake_execute_query, calls


def test_prune_once_stops_when_batch_under_size():
    db = MagicMock()
    db.execute_query, calls = _batches_stub([3])  # single partial batch

    deleted = _prune_once(db, date(2026, 8, 25), retention_days=545, batch_size=5)

    assert deleted == 3
    assert calls["n"] == 1


def test_prune_once_loops_across_full_batches():
    db = MagicMock()
    # Two full batches then a partial one -- exercises the loop actually
    # continuing across multiple round-trips.
    db.execute_query, calls = _batches_stub([5, 5, 2])

    deleted = _prune_once(db, date(2026, 8, 25), retention_days=545, batch_size=5)

    assert deleted == 12
    assert calls["n"] == 3


def test_prune_once_raises_on_error():
    db = MagicMock()
    db.execute_query.return_value = {"status": "error", "message": "boom"}

    with pytest.raises(RuntimeError):
        _prune_once(db, date(2026, 8, 25), retention_days=545, batch_size=5)


# --- run_cycle order-of-operations -----------------------------------------


def test_run_cycle_skips_when_recently_run():
    ssm = MagicMock()
    ssm.get_parameter.return_value = {
        "Parameter": {"Value": _today().isoformat()}
    }
    db = MagicMock()

    run_cycle(db, ssm)

    db.execute_query.assert_not_called()
    ssm.put_parameter.assert_not_called()


def test_run_cycle_does_not_record_state_when_delete_fails():
    ssm = MagicMock()
    ssm.get_parameter.side_effect = _client_error("ParameterNotFound")
    db = MagicMock()
    db.execute_query.return_value = {"status": "error", "message": "boom"}

    with pytest.raises(RuntimeError):
        run_cycle(db, ssm)

    ssm.put_parameter.assert_not_called()


def test_run_cycle_records_state_only_after_successful_prune():
    ssm = MagicMock()
    ssm.get_parameter.side_effect = _client_error("ParameterNotFound")
    db = MagicMock()
    db.execute_query.return_value = {"status": "success", "data": []}  # empty table, one no-op batch

    run_cycle(db, ssm)

    ssm.put_parameter.assert_called_once()
    kwargs = ssm.put_parameter.call_args.kwargs
    assert kwargs["Value"] == _today().isoformat()
    assert kwargs["Overwrite"] is True
