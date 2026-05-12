"""Yahoo Finance news scraper backed by the ``yfinance`` library."""

from __future__ import annotations

import time as _time
from typing import Iterator

from tqdm import tqdm

from NLP.core import get_logger, normalize_timestamp
from NLP.scrapers.base import ArticleRecord, BaseNewsScraper

logger = get_logger(__name__)


class YahooNewsScraper(BaseNewsScraper):
    """Pull the recent news feed for a ticker from Yahoo Finance.

    yfinance is rate-limited so we retry up to three times with
    exponential backoff. Non rate-limit errors fail fast.
    """

    MAX_RETRIES: int = 3
    BASE_BACKOFF_SECONDS: int = 30

    def scrape(self) -> Iterator[ArticleRecord]:  # type: ignore[override]
        import yfinance as yf

        news = []
        for attempt in range(self.MAX_RETRIES):
            try:
                asset = yf.Ticker(self.symbol)
                news = asset.news
                break
            except Exception as exc:
                msg = str(exc).lower()
                if "rate limit" in msg or "too many requests" in msg:
                    wait = self.BASE_BACKOFF_SECONDS * (2 ** attempt)
                    logger.warning(
                        f"[{self.symbol}] Yahoo rate limited, waiting {wait}s "
                        f"(attempt {attempt + 1}/{self.MAX_RETRIES})..."
                    )
                    _time.sleep(wait)
                else:
                    logger.warning(f"[{self.symbol}] Yahoo scrape error: {exc}")
                    return
        else:
            logger.warning(
                f"[{self.symbol}] Yahoo rate limit persists after "
                f"{self.MAX_RETRIES} attempts, skipping Yahoo source"
            )
            return

        for article in tqdm(
            news, desc="Scraping Yahoo News articles...", total=len(news)
        ):
            content = article.get("content", {}) or {}
            title = content.get("title", "N/A")
            summary = content.get("summary", "N/A")
            pub_date_raw = content.get("pubDate", "N/A")
            pub_date = normalize_timestamp(
                None if pub_date_raw == "N/A" else pub_date_raw
            )

            canonical_url = (content.get("canonicalUrl") or {}).get("url", "N/A")

            yield {
                "publishedDate": pub_date,
                "title": title,
                "content": summary,
                "site": canonical_url,
            }
