#API gateway for 'fetch_articles' and 'fetch_alt_articles'

import requests
import os
from src.common.auth.apiAuth import APIAuth

class ArticlesGateway:
    #init the API keys for articles
    def __init__(self):
        api = APIAuth()
        self.fmp_key = api.get_fmp_api_key() #fetch_articles (FMP)
        self.alpha_key = os.getenv("ALPHA_KEY") #fetch_alt_articles (Alpha)

    # ---------------- FMP NEWS / fetch_articles----------------
    def fetch_fmp_news(self, ticker, page=0):
        url = (
            f"https://financialmodelingprep.com/api/v3/stock_news"
            f"?tickers={ticker}&page={page}&apikey={self.fmp_key}"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json() 

    # ---------------- ALPHA NEWS / fetch_alt_articles----------------
    def fetch_alpha_news(self, ticker_list, time_from, time_to):
        tickers = ",".join(ticker_list)
        url = (
            f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
            f"&tickers={tickers}&time_from={time_from}&time_to={time_to}"
            f"&apikey={self.alpha_key}"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
