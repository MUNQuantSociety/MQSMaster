#!/usr/bin/env python3
"""
dynamic_ticker_fetch.py
this is a script that will combine the tickers.json with a dynamically loaded list of tickers from the S&P500 and the top 500 crypto currencies.
it will use a cron job to run weekly that scrapes wikipedia and coinmarketcap for the latest tickers and update the tickers.json file accordingly.
it will update the database during this run filling/removing tickers as necessary. to maintain data integrity.
"""
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

def set_drivers():
    """
    Sets up Selenium WebDrivers for scraping S&P 500 and crypto tickers.
    Returns two configured WebDriver instances.
    """
    SP_500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    crypto_url = "https://coinmarketcap.com/"

    # Configure Chrome to be stable for scraping.
    chrome_options = Options()
    # Uncomment to run headless (no window) if desired.
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver1 = None
    driver2 = None
    try:
        driver1 = webdriver.Chrome(options=chrome_options)
        driver2 = webdriver.Chrome(options=chrome_options)

        driver1.get(SP_500_url)
        driver2.get(crypto_url)
        return driver1, driver2

    except Exception as e:
        if driver1:
            driver1.quit()
        if driver2:
            driver2.quit()
        raise e

def _load_more_content(driver, max_scrolls=10, pause_seconds=0.5):
    """Scrolls the page to trigger lazy-loaded content.
    Stops early if no additional content loads.
    """
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_seconds)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def scrape_tickers(max_scrolls=10):
    """
    Scrapes top 500 tickers from S&P 500 and crypto pages.
    Returns a list of ticker data.
    """
    driver1, driver2 = set_drivers()
    drivers = [driver1]
    try:
        for driver in drivers:
            wait = WebDriverWait(driver, 30)
            wait.until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        "#constituents > tbody > tr:nth-child(1) > td:nth-child(1) > a",
                    )
                )
            )
            # Load more items by scrolling.
            _load_more_content(driver, max_scrolls=max_scrolls, pause_seconds=0.5)

        # Collect story cards.
        elements = driver.find_elements(
            By.CLASS_NAME, 'external text'
        )
        ticker = elements[0]
        # get ticker
        symbol = ticker.find_element(By.TAG_NAME, "a").get_attribute("textContent").strip()
        tickers = [{"symbol": symbol, "source": "dynamic_scrape"}]
        return tickers
    except Exception:
        print("An error occurred during scraping.")
        return []
    finally:
        driver1.quit()
        driver2.quit()

def main():
    """
    Main function to execute the ticker scraping and update tickers.csv.
    """
    tickers = scrape_tickers(max_scrolls=20)
    if tickers:
        tickers_path = Path(__file__).parent / "tickers.csv"
        existing_tickers = pd.read_json('tickers.json').to_dict()
        # Combine and deduplicate tickers
        combined_tickers = {t['symbol']: t for t in existing_tickers + tickers}.values()
        df = pd.DataFrame(combined_tickers)
        df.to_csv(tickers_path, index=False)
        print(f"Updated tickers.csv with {len(combined_tickers)} unique tickers.")
    else:
        print("No tickers were scraped.")


if __name__ == "__main__":
    main()
