#!/usr/bin/env python3
"""
news_scraper.py
Scrape latest Yahoo Finance news items for a symbol using Selenium.
Reliably loads content by scrolling and waits; saves CSV per symbol.
"""

import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def _load_more_content(driver, max_scrolls=10, pause_seconds=2.5):
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


def scrape_article_content(symbol, max_scrolls=10):
    """
    Scrape latest news items for `symbol` from Yahoo Finance.
    Returns a DataFrame with columns [title, link]. Also writes CSV under NLP/articles.
    """
    url = f"https://finance.yahoo.com/quote/{symbol}/latest-news/"

    # Configure Chrome to be stable for scraping.
    chrome_options = Options()
    # Uncomment to run headless (no window) if desired.
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)

        wait = WebDriverWait(driver, 30)
        wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'section[data-testid="storyitem"]')
            )
        )

        # Load more items by scrolling.
        _load_more_content(driver, max_scrolls=max_scrolls, pause_seconds=2.5)

        # Collect story cards.
        stories = driver.find_elements(
            By.CSS_SELECTOR, 'section[data-testid="storyitem"]'
        )
        story = stories[0]
        # Get story title within first item
        title = story.find_element(By.TAG_NAME, "h3").text
        # get link to story
        link = story.find_element(By.TAG_NAME, "a").get_attribute("href")

    except Exception as e:
        print(f"Session error: {e}. Cannot scrape article content.")
        return ""
    try:
        if driver:
            for x in stories:
                link = x.find_element(By.TAG_NAME, "a").get_attribute("href")
                title = x.find_element(By.TAG_NAME, "h3").text
                date = x.find_element(By.CLASS_NAME, "publishing").text
                yield {"date": date, "title": title, "link": link}
        else:
            driver.quit()
            return link, title
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


def main():
    symbols = ["AAPL", "MSFT", "GOOGL"]
    for symbol in symbols:
        row= scrape_article_content(symbol, max_scrolls=12)
        article_df = pd.DataFrame(row)
        output_path = Path(__file__).parent / "articles" / f"{symbol}_yahoo_news.csv"
        article_df.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
        if article_df is not None and not article_df.empty:
            print(f"Scraped articles for {symbol}")
            print(article_df[["date", "title", "link"]].head(5), "\n")
        else:
            print(f"No articles found for {symbol}")
    return


if __name__ == "__main__":
    main()
