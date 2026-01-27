# tests/test_backtest.py

import pytest
from datetime import datetime, timedelta
from src.portfolios.portfolio_1.strategy import VolMomentum
from src.portfolios.portfolio_2.strategy import MomentumStrategy
from src.portfolios.portfolio_3.strategy import RegimeAdaptiveStrategy
from src.main_backtest import main


@pytest.fixture
def start_date():
    return datetime(2024, 1, 1).strftime("%Y-%m-%d")

@pytest.fixture
def end_date():
    start = datetime(2024, 1, 1)
    end = start + timedelta(days=90)
    return end.strftime("%Y-%m-%d")


def test_portfolio_1_strategy(start_date, end_date):
    test = main(
        portfolio_classes=[VolMomentum],
        start_date=start_date,
        end_date=end_date,
        initial_capital=1000000.0,
        slippage=0
    )
    
    assert test is not None, "Backtest failed for Portfolio 1's strategy"

def test_portfolio_2_strategy(start_date, end_date):
    test = main(
        portfolio_classes=[MomentumStrategy],
        start_date=start_date,
        end_date=end_date,
        initial_capital=1000000.0,
        slippage=0
    )
    
    assert test is not None, "Backtest failed for Portfolio 2's strategy"

def test_portfolio_3_strategy(start_date, end_date):
    test = main(
        portfolio_classes=[RegimeAdaptiveStrategy],
        start_date=start_date,
        end_date=end_date,
        initial_capital=1000000.0,
        slippage=0
    )
    
    assert test is not None, "Backtest failed for Portfolio 3's strategy"

