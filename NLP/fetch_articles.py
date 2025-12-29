import sys
import os
import time
import requests
import pandas as pd
import argparse
from datetime import datetime
import json

# insert project root into your path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.common.auth.apiAuth import APIAuth

# ─── CONFIG ───────────────────────────────────────────────────────────────────
api = APIAuth()
API_KEY = api.get_fmp_api_key()  # ← CALL the method!
MAX_PAGES_PER_RUN = 50  # Max pages to fetch in a single cycle
RATE_LIMIT = 0.2  # Seconds to wait between API calls

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "articles")
STATE_DIR = os.path.join(BASE_DIR, "fetch_state")


# ─── ARGPARSE TO PICK UP COMMAND‐LINE ARGUMENTS ───────────────────────────────
def parse_args():
    """Parse command-line arguments for ticker and date range."""
    p = argparse.ArgumentParser(
        description="Fetch and update stock-news CSVs for a ticker within a date range."
    )
    p.add_argument(
        "ticker", help="The ticker symbol to fetch (e.g., AAPL)."
    )
    p.add_argument(
        "start_date", help="Start date for fetching articles, in YYYY-MM-DD format."
    )
    p.add_argument(
        "end_date", help="End date for fetching articles, in YYYY-MM-DD format."
    )
    return p.parse_args()


# ─── FUNCTIONS ────────────────────────────────────────────────────────────────
# TODO: Implement content scraping
#############################################################################
def fetch_news(symbol, start_date, end_date, start_page=0):
    """
    Fetch stock news from FMP for a symbol within a date range.
    Returns articles found, flag if page limit was hit, and next start page.
    """
    print(f"[{symbol}] Fetching articles from page {start_page}...")
    all_articles = []
    page = start_page
    reached_start_date = False

    while page < start_page + MAX_PAGES_PER_RUN:
        url = (
            f"https://financialmodelingprep.com/api/v3/stock_news"
            f"?tickers={symbol}&page={page}&apikey={API_KEY}"
        )
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            news = resp.json()
        except requests.RequestException as e:
            print(f"[{symbol}] Request error on page {page}: {e}")
            break

        if not news:
            print(f"[{symbol}] No more articles found from API.")
            break

        for art in news:
            pd_str = art.get("publishedDate")
            try:
                art_date = datetime.strptime(pd_str, "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                continue

            if art_date < start_date:
                reached_start_date = True
                break

            if art_date <= end_date:
                # ⬇️ FIX APPLIED HERE ⬇️
                all_articles.append({
                    "publishedDate": art_date,
                    "title": (art.get("title") or "").strip(),
                    "content": (art.get("text") or art.get("content") or "").strip(),
                    "site": (art.get("url") or "").strip(),
                })

        if reached_start_date:
            print(f"[{symbol}] Reached the start date boundary.")
            break

        page += 1
        time.sleep(RATE_LIMIT)
    
    hit_max_pages = (page == start_page + MAX_PAGES_PER_RUN)
    next_start_page = page
    
    if hit_max_pages:
        print(f"[{symbol}] Reached page limit for this run. Next start page: {next_start_page}")
    else:
        print(f"[{symbol}] Finished fetching available pages. Total pages: {page - start_page}")

    return all_articles, hit_max_pages, next_start_page
def save_fetch_state(ticker, next_start_page, start_date, end_date):
    """Save fetch state to JSON file"""
    os.makedirs(STATE_DIR, exist_ok=True)
    state_path = os.path.join(STATE_DIR, f"{ticker}_state.json")
    
    state = {
        "next_start_page": next_start_page,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d")
    }
    
    with open(state_path, 'w') as f:
        json.dump(state, f)


def load_fetch_state(ticker, user_start, user_end):
    """Load fetch state from JSON file if dates match"""
    state_path = os.path.join(STATE_DIR, f"{ticker}_state.json")
    
    if not os.path.exists(state_path):
        return 0  # Start from page 0 if no state
    
    try:
        with open(state_path, 'r') as f:
            state = json.load(f)
        
        # Reset state if date range changed
        if (state["start_date"] != user_start.strftime("%Y-%m-%d") or
            state["end_date"] != user_end.strftime("%Y-%m-%d")):
            return 0
        
        return state["next_start_page"]
    except Exception:
        return 0


def update_ticker_csv(symbol, start_date_str, end_date_str):
    """
    Fetch articles incrementally by tracking last fetched page in state file
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
    
    try:
        user_start = datetime.strptime(start_date_str, "%Y-%m-%d")
        user_end = datetime.strptime(end_date_str, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59
        )
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format.")
        return
    
    # Load or initialize fetch state
    start_page = load_fetch_state(symbol, user_start, user_end)
    has_more_pages = True
    run_count = 1
    
    while has_more_pages:
        print(f"\n--- Starting fetch cycle #{run_count} (starting page: {start_page}) ---")
        
        # Fetch articles from current start page
        articles, hit_max_pages, next_page = fetch_news(
            symbol, 
            user_start, 
            user_end, 
            start_page
        )
        
        # Update CSV if we found articles
        if articles:
            new_df = pd.DataFrame(articles)
            
            if os.path.exists(csv_path):
                try:
                    old_df = pd.read_csv(csv_path, parse_dates=["publishedDate"])
                    combined = pd.concat([old_df, new_df], ignore_index=True)
                except pd.errors.EmptyDataError:
                    combined = new_df
            else:
                combined = new_df
                
            # Remove duplicates and sort
            initial_count = len(combined)
            combined.drop_duplicates(
                subset=["publishedDate", "title"], 
                keep="first", 
                inplace=True
            )
            combined.sort_values("publishedDate", ascending=False, inplace=True)
            combined.to_csv(csv_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
            
            added = len(combined) - (initial_count - len(new_df))
            duplicates_removed = initial_count - len(combined)
            print(f"[{symbol}] Added {added} new articles, removed {duplicates_removed} duplicates")
        else:
            print(f"[{symbol}] No new articles found in this batch")
        
        # Update state and determine if we should continue
        start_page = next_page
        has_more_pages = hit_max_pages
        save_fetch_state(symbol, start_page, user_start, user_end)
        
        run_count += 1
    
    print(f"\n[{symbol}] Fetching completed successfully!")


def main():
    """Main function to parse arguments and initiate the fetch."""
    args = parse_args()
    update_ticker_csv(args.ticker.upper(), args.start_date, args.end_date)


if __name__ == "__main__":
    main()
    