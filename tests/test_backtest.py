# tests/test_backtest.py

from datetime import datetime, timedelta

import pytest

from src.main_backtest import _init_backtest
from src.portfolios.portfolio_1.strategy import VolMomentum
from src.portfolios.portfolio_2.strategy import MomentumStrategy
from src.portfolios.portfolio_3.strategy import RegimeAdaptiveStrategy

_BASE_START = datetime(2024, 1, 1)

@pytest.fixture
def start_date():
    return _BASE_START.strftime("%Y-%m-%d")



@pytest.fixture
def end_date():
    start = _BASE_START
    end = start + timedelta(days=90)
    return end.strftime("%Y-%m-%d")


@pytest.mark.parametrize(
    "portfolio_class",
    [VolMomentum, MomentumStrategy, RegimeAdaptiveStrategy],
)
def test_init_backtest_preserves_all_inputs(
    portfolio_class,
    start_date,
    end_date,
):
    (
        portfolio_classes,
        resolved_start_date,
        resolved_end_date,
        initial_capital,
        slippage,
        backtest_mode,
        resolved_fast_config,
    ) = _init_backtest(
        portfolio_classes=[portfolio_class],
        start_date=start_date,
        end_date=end_date,
        initial_capital=1000000.0,
        slippage=0.0,
        backtest_mode="event",
    )

    assert portfolio_classes == [portfolio_class]
    assert resolved_start_date == start_date
    assert resolved_end_date == end_date
    assert initial_capital == 1000000.0
    assert slippage == 0.0
    assert backtest_mode == "event"
    assert resolved_fast_config is None
