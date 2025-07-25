# fetch_articles.py

import sys
import os
import time
import requests
import pandas as pd
import argparse
from datetime import datetime, timedelta

# insert project root into your path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from data_infra.authentication.apiAuth import APIAuth

# ─── CONFIG ───────────────────────────────────────────────────────────────────
api       = APIAuth()
API_KEY   = api.get_fmp_api_key()    # ← CALL the method!
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL"]
DAYS_BACK = 3130
MAX_PAGES = 50
RATE_LIMIT = 0.2

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "articles")


# ─── ARGPARSE TO PICK UP COMMAND‐LINE TICKERS ─────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch and update stock‐news CSVs for one or more tickers."
    )
    p.add_argument(
        "tickers",
        nargs="*",
        help=(
            "Ticker symbols to fetch (e.g. AAPL MSFT), "
            "or comma‐separated (e.g. AAPL,MSFT). "
            "If none given, uses default list."
        )
    )
    return p.parse_args()


# ─── FUNCTIONS ────────────────────────────────────────────────────────────────

def fetch_news(symbol, days=DAYS_BACK):
    """Fetch recent stock news from FMP for a given symbol."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    all_articles = []
    page = 0

    while page < MAX_PAGES:
        url = (
            f"https://financialmodelingprep.com/api/v3/stock_news"
            f"?symbol={symbol}&page={page}&apikey={API_KEY}"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            news = resp.json()
        except requests.RequestException as e:
            print(f"[{symbol}] Request error on page {page}: {e}")
            break

        if not news:
            break  # no more data

        # filter & normalize
        for art in news:
            pd_str = art.get("publishedDate")
            try:
                art_date = datetime.strptime(pd_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            if art_date < cutoff:
                continue

            all_articles.append({
                "publishedDate": art_date,
                "title": art.get("title", "").strip(),
                "content": art.get("text", art.get("content", "")).strip(),
                "site": art.get("site", "").strip()
            })

        page += 1
        time.sleep(RATE_LIMIT)

    return all_articles


def update_ticker_csv(symbol):
    """Fetch new articles for a symbol and append to its CSV under articles/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"{symbol}.csv")

    # load existing
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path, parse_dates=["publishedDate"])
    else:
        old_df = pd.DataFrame(
            columns=["publishedDate", "title", "content", "site"]
        )

    # fetch & build DataFrame
    articles = fetch_news(symbol)
    if not articles:
        print(f"[{symbol}] No new articles found.")
        return

    new_df = pd.DataFrame(articles)
    # drop exact duplicates
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined.drop_duplicates(
        subset=["publishedDate", "title"], keep="first", inplace=True
    )
    combined.sort_values("publishedDate", inplace=True)

    # save
    combined.to_csv(csv_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    added = len(combined) - len(old_df)
    print(f"[{symbol}] {added} new, total {len(combined)} articles.")


def main():
    args = parse_args()

    # build ticker list from args or fallback
    if args.tickers:
        # allow both space‑ and comma‑separated
        tickers = []
        for tok in args.tickers:
            tickers += [t.strip().upper() for t in tok.split(",") if t.strip()]
    else:
        tickers = DEFAULT_TICKERS

    for sym in tickers:
        update_ticker_csv(sym)


if __name__ == "__main__":
    main()
