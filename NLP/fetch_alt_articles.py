import logging
import os

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
ALPHA_KEY = os.getenv("ALPHA_KEY")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
PATH = os.path.dirname(__file__) + "/articles"


class ArticleScraper:
    def __init__(self, symbol):
        self.symbol = symbol

    # Dynamically fetch content up to the last complete sentence
    def get_complete_sentences(self, text, min_chars=200, max_chars=1000):
        """Extract text up to the last complete sentence within the range."""
        if len(text) <= min_chars:
            return text
        # Get a chunk that's at least min_chars but no more than max_chars
        chunk = text[:max_chars]

        # Find sentence boundaries (., !, ?)
        import re

        sentence_ends = list(re.finditer(r"[.!?](?:\s|$)", chunk))

        if not sentence_ends:
            # No sentence boundary found, return up to min_chars
            return chunk[:min_chars]

        # Find the last sentence that ends after min_chars
        for match in reversed(sentence_ends):
            if match.end() >= min_chars:
                return chunk[: match.end()].strip()

        # If all sentences end before min_chars, return up to the last sentence
        return chunk[: sentence_ends[-1].end()].strip()

    async def fetch_and_parse(self, session, url, row, semaphore):
        """Fetch URL and extract content with rate limiting via semaphore."""
        from bs4 import BeautifulSoup

        async with semaphore:
            try:
                async with session.get(url, headers=HEADERS, timeout=10) as response:
                    response.raise_for_status()
                    content = await response.text()

                soup = BeautifulSoup(content, "html.parser")
                paragraphs = soup.find_all("p")
                full_content = " ".join([p.get_text().strip() for p in paragraphs])

                # Remove common prefixes
                remove = {
                    "Oops": 26,
                    "This article first": 40,
                    "抱歉，發生錯誤": 9,
                    "Credit": 7,
                }
                if full_content.startswith(tuple(remove.keys())):
                    for prefix, length in remove.items():
                        if full_content.startswith(prefix):
                            full_content = full_content[length:]
                            break

                content = self.get_complete_sentences(full_content)
                logging.debug(content)
                return {
                    "publishedDate": row["Date"],
                    "title": row["Title"],
                    "content": content,
                    "site": url,
                }
            except Exception as e:
                logging.debug(f"Unexpected error for {url}: {e}")
                return None

    async def fetch_all(self, valid_rows=[]):
        """Fetch all URLs concurrently with rate limiting."""
        import asyncio

        import aiohttp

        # Semaphore limits concurrent requests (3 simultaneous) + 0.1s delay = ~10 req/sec
        semaphore = asyncio.Semaphore(3)
        connector = aiohttp.TCPConnector(limit=5)

        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self.fetch_and_parse(session, url, row, semaphore)
                for url, row in valid_rows
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

    def scrape_yahoo(self):
        """
        Scrape latest news items for `symbol` from Yahoo Finance.
        Returns a DataFrame with columns [publishedDate,title, content, site]. Also writes CSV under NLP/articles.
        """
        import yfinance as yf

        asset = yf.Ticker(self.symbol)
        news = asset.news
        # Extract fields from each article
        for article in tqdm(
            news, desc="Scraping Yahoo News articles...", total=len(news)
        ):
            content = article.get("content", {})
            title = content.get("title", "N/A")
            summary = content.get("summary", "N/A")
            pub_date = content.get("pubDate", "N/A")
            canonical_url = content.get("canonicalUrl", {}).get("url", "N/A")

            yield {
                "publishedDate": pub_date,
                "title": title,
                "content": summary,
                "site": canonical_url,
            }

    def scrape_finviz(self):
        """
        Fetch news articles for a given stock symbol from Finviz using concurrent requests.
        Uses asyncio with a semaphore to limit concurrent requests and respect rate limits.
        """
        import asyncio
        from urllib.parse import urlparse

        import finvizfinance.quote as ff

        fnews = ff.finvizfinance(self.symbol)
        news_data = fnews.ticker_news()

        # Validate and prepare URLs first
        valid_rows = []
        for _, row in news_data.iterrows():
            url = row["Link"]

            if not url or pd.isna(url):
                logging.debug(f"Skipping invalid URL: {url}")
                continue

            # Fix relative URLs
            if url.startswith("/"):
                url = "https://finviz.com" + url

            # Validate URL has scheme
            parsed = urlparse(url)
            if not parsed.scheme:
                logging.debug(f"Skipping URL without scheme: {url}")
                continue

            valid_rows.append((url, row))

        # Run async operations and yield results with progress bar
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(self.fetch_all(valid_rows=valid_rows))
            for result in tqdm(
                results,
                desc="Scraping Finviz News articles...",
                total=len(results),
            ):
                if result and not isinstance(result, Exception):
                    yield result
        finally:
            loop.close()

    def scrape_alpha(
        self,
        ticker=["AAPL"],
        time_from="20251201T1200",
        time_to="20251231T1200",
        apikey=ALPHA_KEY,
    ):
        """Scrape news articles from Alpha Vantage for given tickers and time range."""
        from datetime import datetime

        import requests

        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={','.join(ticker)}&time_from={time_from}&time_to={time_to}&apikey={apikey}"

        r = requests.get(url)
        news = r.json()
        for news in tqdm(
            news.get("feed", []),
            desc="Scraping Alpha Vantage News articles...",
            total=len(news.get("feed", [])),
        ):
            title = news.get("title", "N/A")
            summary = news.get("summary", "N/A")
            pub_date = news.get("time_published", "N/A")

            # Format the timestamp from API format (e.g., "20251231T120000Z") to readable format
            if pub_date != "N/A":
                try:
                    pub_date = datetime.strptime(pub_date, "%Y%m%dT%H%M%S").strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                except ValueError as e:
                    print(f"Error parsing date {pub_date}: {e}")

            url = news.get("url", "N/A")

            yield {
                "publishedDate": pub_date,
                "title": title,
                "content": summary,
                "site": url,
            }


def main():
    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        scraper = ArticleScraper(symbol)
        logging.info(f"Fetching articles for {symbol}...")
        yahoo = scraper.scrape_yahoo()
        finviz = scraper.scrape_finviz()
        alpha = scraper.scrape_alpha()
        print('--------------------\n')
        print(f"Fetching articles for {symbol}.")

        # Convert to DataFrame for further analysis if needed
        yahoo_news_df = pd.DataFrame(yahoo)
        finviz_news_df = pd.DataFrame(finviz)
        alpha_news_df = pd.DataFrame(alpha)

        with open(f"{PATH}/{symbol}_yahoo_news.csv", "w", encoding="utf-8") as f:
            yahoo_news_df.to_csv(f, index=False, date_format="%Y-%m-%d %H:%M:%S")
        with open(f"{PATH}/{symbol}_finviz_news.csv", "w", encoding="utf-8") as f:
            finviz_news_df.to_csv(f, index=False, date_format="%Y-%m-%d %H:%M:%S")
        with open(f"{PATH}/{symbol}_alpha_news.csv", "w", encoding="utf-8") as f:
            alpha_news_df.to_csv(f, index=False, date_format="%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    main()
