#tests/test_api_endpoints.py

from datetime import datetime, timedelta
from src.orchestrator.marketData.fmpMarketData import FMPMarketData

def fmp():
    return FMPMarketData

def test_dates():
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    return start_date, end_date

def test_historical_data(fmp, start_date, end_date):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/AAPL?from={start_date}&to={end_date}&apikey={fmp.fmp_api_key}"
    response = requests.get(url)
    assert response.status_code == 200, f"API returned {response.status_code}: {response.text}"

    data = fmp.get_historical_data("AAPL", start_date, end_date)
    assert data is not None and isinstance(data, list) and len(data) > 0, "Historical data api failed"

def test_intraday_data(fmp, start_date, end_date):
    url = f"https://financialmodelingprep.com/api/v3/historical-chart/{fmp.interval_str}/AAPL?from={start_date}&to={end_date}&apikey={fmp.fmp_api_key}"
    response = requests.get(url)
    assert response.status_code == 200, f"API returned {response.status_code}: {response.text}"
    data = fmp.get_intraday_data("AAPL", start_date, end_date)
    assert data is not None and isinstance(data, list) and len(data) > 0, "Intraday data api failed"

def test_realtime_data(fmp, start_date, end_date):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/AAPL?from={start_date}&to={end_date}&apikey={fmp.fmp_api_key}"
    response = requests.get(url)
    assert response.status_code == 200, f"API returned {response.status_code}: {response.text}"
    
    data = fmp.get_realtime_data("AAPL", start_date, end_date)
    assert data is not None and isinstance(data, list) and len(data) > 0, "Realtime data api failed"

def test_curent_price(fmp, start_date, end_date):
    url = f"https://financialmodelingprep.com/stable/quote?symbol=AAPL"
    params = {"apikey": self.fmp_api_key}
    response = requests.get(url, params)
    assert response.status_code == 200, f"API returned {response.status_code}: {response.text}"

    price = fmp.get_current_price("AAPl")
    assert isinstance(data, list) and len(data) > 0, "Current Price api failed"
 
