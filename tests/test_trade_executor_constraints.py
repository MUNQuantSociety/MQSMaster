import logging
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.live_trading.executor import tradeExecutor

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.workflow_live,
]


def _build_executor(price_map, record):
    executor = tradeExecutor.__new__(tradeExecutor)
    executor.dbconn = object()
    executor.leverage = 2.0
    executor.rbp_overlay = None
    executor.logger = logging.getLogger("test_trade_executor")

    def fake_get_current_price(ticker):
        return float(price_map.get(ticker, 0.0))

    def fake_update_database(
        portfolio_id,
        ticker,
        signal_type,
        quantity_to_trade,
        updated_cash,
        updated_quantity,
        arrival_price,
        exec_price,
        slippage_bps,
        timestamp,
        port_notional,
    ):
        record.append(
            {
                "portfolio_id": portfolio_id,
                "ticker": ticker,
                "signal_type": signal_type,
                "quantity": quantity_to_trade,
                "updated_cash": updated_cash,
                "updated_quantity": updated_quantity,
                "exec_price": exec_price,
                "port_notional": port_notional,
                "timestamp": timestamp,
            }
        )
        return {
            "status": "success",
            "quantity": quantity_to_trade,
            "updated_cash": updated_cash,
        }

    executor.get_current_price = fake_get_current_price
    executor.update_database = fake_update_database
    return executor


def _empty_positions():
    return pd.DataFrame(columns=["ticker", "quantity"])


def test_buying_power_calculation():
    record = []
    executor = _build_executor({"MSFT": 50.0}, record)
    positions = pd.DataFrame(
        [
            {"ticker": "AAPL", "quantity": 5},
            {"ticker": "MSFT", "quantity": 10},
        ]
    )

    buying_power = executor._calculate_buying_power(
        portfolio_equity=1000.0,
        positions_df=positions,
        current_ticker="AAPL",
        current_ticker_price=100.0,
    )

    assert buying_power == pytest.approx(1000.0)


@pytest.mark.parametrize("signal_type", ["BUY", "SELL"])
def test_order_sizing_limits_skip_small_trade(signal_type):
    record = []
    executor = _build_executor({"AAPL": 100.0}, record)

    result = executor.execute_trade(
        portfolio_id="p1",
        ticker="AAPL",
        signal_type=signal_type,
        confidence=1.0,
        arrival_price=100.0,
        cash=10000.0,
        positions=_empty_positions(),
        port_notional=1000.0,
        ticker_weight=0.0005,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    assert result is None
    assert record == []


@pytest.mark.parametrize("signal_type", ["BUY", "SELL"])
def test_position_weight_limits_respected(signal_type):
    record = []
    executor = _build_executor({"AAPL": 100.0}, record)

    result = executor.execute_trade(
        portfolio_id="p1",
        ticker="AAPL",
        signal_type=signal_type,
        confidence=1.0,
        arrival_price=100.0,
        cash=10000.0,
        positions=_empty_positions(),
        port_notional=10000.0,
        ticker_weight=0.25,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    assert result is not None
    assert record
    trade = record[0]
    trade_value = trade["quantity"] * trade["exec_price"]
    assert trade_value <= 2500.0
    assert trade["signal_type"] == signal_type


@pytest.mark.parametrize(
    "signal_type, expected_updated_cash, expected_updated_quantity",
    [
        ("BUY", 500.0, 50.0),
        ("SELL", 1500.0, -50.0),
    ],
)
def test_trade_execution_happy_path(
    signal_type, expected_updated_cash, expected_updated_quantity
):
    record = []
    executor = _build_executor({"AAPL": 10.0}, record)

    result = executor.execute_trade(
        portfolio_id="p1",
        ticker="AAPL",
        signal_type=signal_type,
        confidence=1.0,
        arrival_price=10.0,
        cash=1000.0,
        positions=_empty_positions(),
        port_notional=1000.0,
        ticker_weight=0.5,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    assert result is not None
    assert result["quantity"] == 50
    assert result["updated_cash"] == pytest.approx(expected_updated_cash)
    assert record[0]["signal_type"] == signal_type
    assert record[0]["quantity"] == 50
    assert record[0]["updated_cash"] == pytest.approx(expected_updated_cash)
    assert record[0]["updated_quantity"] == pytest.approx(expected_updated_quantity)
    assert record
