"""
crypto_tickers.py
-----------------
Fetches top 500 cryptocurrencies by market cap and formats them for FMP API.
Automatically updates tickers.json with the latest crypto tickers each run.

Usage:
    python crypto_tickers.py
"""

import requests
import json
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TICKERS_JSON_PATH = os.path.join(SCRIPT_DIR, "tickers.json")
CRYPTO_TICKERS_PATH = os.path.join(SCRIPT_DIR, "crypto_tickers.json")


def get_top_crypto_from_coingecko(limit=500):
    """
    Fetch top cryptocurrencies by market cap from CoinGecko API (free, no key needed).
    Automatically gets fresh data every time it runs.
    """
    print(f"🔄 Fetching top {limit} cryptocurrencies from CoinGecko (live)...")
    
    all_coins = []
    per_page = 250
    pages_needed = (limit // per_page) + 1
    
    for page in range(1, pages_needed + 1):
        url = (
            f"https://api.coingecko.com/api/v3/coins/markets"
            f"?vs_currency=usd&order=market_cap_desc&per_page={per_page}&page={page}"
        )
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            coins = response.json()
            all_coins.extend(coins)
            print(f"  Page {page}: Got {len(coins)} coins")
            time.sleep(1)  # Rate limit: be nice to free API
        except Exception as e:
            print(f"  Error on page {page}: {e}")
            break
    
    return all_coins[:limit]


def format_for_fmp(coins):
    """
    Convert coin symbols to FMP format (e.g., BTC -> BTCUSD).
    """
    fmp_tickers = []
    
    for coin in coins:
        symbol = coin.get('symbol', '').upper()
        name = coin.get('name', '')
        market_cap = coin.get('market_cap', 0)
        
        if symbol and len(symbol) <= 10:  # Skip weird long symbols
            fmp_tickers.append({
                "ticker": f"{symbol}USD",
                "symbol": symbol,
                "name": name,
                "market_cap": market_cap
            })
    
    return fmp_tickers


def update_tickers_json(crypto_tickers):
    """
    Update the main tickers.json file with crypto tickers.
    Removes old crypto entries and adds fresh ones.
    """
    # Load existing tickers.json
    if os.path.exists(TICKERS_JSON_PATH):
        with open(TICKERS_JSON_PATH, 'r') as f:
            existing_tickers = json.load(f)
        print(f"📂 Loaded {len(existing_tickers)} existing tickers from tickers.json")
    else:
        existing_tickers = []
        print("📂 No existing tickers.json found, creating new one")
    
    # Remove old crypto tickers (anything ending in USD that's crypto)
    crypto_endings = ['USD']
    non_crypto_tickers = []
    removed_count = 0
    
    for ticker in existing_tickers:
        # Keep if it's a stock ticker (doesn't look like crypto)
        is_crypto = (
            isinstance(ticker, str) and 
            ticker.endswith('USD') and 
            len(ticker) <= 10 and
            ticker not in ['JPYUSD', 'EURUSD', 'GBPUSD']  # Keep forex if any
        )
        if not is_crypto:
            non_crypto_tickers.append(ticker)
        else:
            removed_count += 1
    
    print(f"🗑️  Removed {removed_count} old crypto tickers")
    
    # Add new crypto tickers
    new_crypto_list = [c["ticker"] for c in crypto_tickers]
    updated_tickers = non_crypto_tickers + new_crypto_list
    
    # Save updated tickers.json
    with open(TICKERS_JSON_PATH, 'w') as f:
        json.dump(updated_tickers, f, indent=4)
    
    print(f"✅ Updated tickers.json: {len(non_crypto_tickers)} stocks + {len(new_crypto_list)} crypto = {len(updated_tickers)} total")
    
    return updated_tickers


def save_crypto_details(crypto_tickers):
    """
    Save detailed crypto info to separate file for reference.
    """
    with open(CRYPTO_TICKERS_PATH, 'w') as f:
        json.dump(crypto_tickers, f, indent=4)
    
    print(f"📝 Saved detailed crypto info to crypto_tickers.json")


def main():
    print("=" * 60)
    print("🚀 CRYPTO TICKERS UPDATER")
    print("=" * 60)
    print("This script fetches live data and updates tickers.json")
    print()
    
    # Step 1: Fetch top 500 crypto (live data)
    coins = get_top_crypto_from_coingecko(500)
    
    if not coins:
        print("❌ Failed to fetch crypto data")
        return
    
    print(f"\n✅ Fetched {len(coins)} cryptocurrencies")
    
    # Step 2: Format for FMP
    fmp_tickers = format_for_fmp(coins)
    print(f"✅ Formatted {len(fmp_tickers)} tickers for FMP")
    
    # Step 3: Save detailed crypto info
    save_crypto_details(fmp_tickers)
    
    # Step 4: Update main tickers.json
    update_tickers_json(fmp_tickers)
    
    # Step 5: Print preview
    print("\n" + "=" * 60)
    print("📋 TOP 20 CRYPTO TICKERS (by market cap):")
    print("=" * 60)
    for i, coin in enumerate(fmp_tickers[:20], 1):
        print(f"  {i:2}. {coin['ticker']:12} - {coin['name']}")
    
    print("\n✅ Done! Run this script anytime to refresh crypto list.")


if __name__ == "__main__":
    main()
