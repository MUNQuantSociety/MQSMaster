# tests/test_api_endpoints.py
import pytest
from datetime import datetime, timedelta
from src.orchestrator.marketData.fmpMarketData import FMPMarketData

@pytest.fixture
def start_date():
    start_date = (datetime.now() - timedelta(days=7))
    return start_date.strftime("%Y-%m-%d")

@pytest.fixture
def end_date():
    return datetime.now().strftime("%Y-%m-%d")

@pytest.fixture
def fmp():
    return FMPMarketData()


def test_historical_data(fmp, start_date, end_date):
    data = fmp.get_historical_data("AAPL", start_date, end_date)
    assert data is not None and isinstance(data, list) and len(data) > 0, "Historical data api failed"

def test_intraday_data(fmp, start_date, end_date):
    data = fmp.get_intraday_data("AAPL", start_date, end_date, 5)
    assert data is not None and isinstance(data, list) and len(data) > 0, "Intraday data api failed"

def test_realtime_data(fmp, start_date, end_date):
    data = fmp.get_realtime_data("NASDAQ")
    assert data is not None and isinstance(data, list) and len(data) > 0, "Realtime data api failed"

def test_curent_price(fmp):
    price = fmp.get_current_price("AAPL")
    assert isinstance(price, float) and price > 0, "Current Price api failed"

