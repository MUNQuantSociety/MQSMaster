"""Finviz news scraper using async HTTP for fan-out fetches."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Iterator, List, Tuple
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm

from NLP.core import get_logger, normalize_timestamp
from NLP.scrapers.base import ArticleRecord, BaseNewsScraper

logger = get_logger(__name__)

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

_LEADING_PREFIXES = {
    "Oops": 26,
    "This article first": 40,
    "抱歉，發生錯誤": 9,
    "Credit": 7,
}


class FinvizNewsScraper(BaseNewsScraper):
    """Fetch story metadata from Finviz and follow each link concurrently."""

    CONCURRENCY: int = 3
    CONNECTION_LIMIT: int = 5
    REQUEST_TIMEOUT_SECONDS: int = 10
    CONTENT_MAX_CHARS: int = 500
    SUMMARY_MIN_CHARS: int = 200
    SUMMARY_MAX_CHARS: int = 1000

    def __init__(self, symbol: str, headers: dict | None = None):
        super().__init__(symbol)
        self.headers = headers or _DEFAULT_HEADERS

    @staticmethod
    def _trim_to_complete_sentence(
        text: str, min_chars: int, max_chars: int
    ) -> str:
        """Return text bounded to the last complete sentence within range."""
        if len(text) <= min_chars:
            return text

        chunk = text[:max_chars]
        sentence_ends = list(re.finditer(r"[.!?](?:\s|$)", chunk))

        if not sentence_ends:
            return chunk[:min_chars]

        for match in reversed(sentence_ends):
            if match.end() >= min_chars:
                return chunk[: match.end()].strip()

        return chunk[: sentence_ends[-1].end()].strip()

    @staticmethod
    def _strip_known_prefixes(text: str) -> str:
        if not text.startswith(tuple(_LEADING_PREFIXES.keys())):
            return text
        for prefix, length in _LEADING_PREFIXES.items():
            if text.startswith(prefix):
                return text[length:]
        return text

    async def _fetch_and_parse(self, session, url, row, semaphore):
        from bs4 import BeautifulSoup

        async with semaphore:
            try:
                async with session.get(
                    url, headers=self.headers, timeout=self.REQUEST_TIMEOUT_SECONDS
                ) as response:
                    response.raise_for_status()
                    body = await response.text()

                soup = BeautifulSoup(body, "html.parser")
                paragraphs = soup.find_all("p")
                full_content = " ".join(
                    p.get_text().strip() for p in paragraphs
                )[: self.CONTENT_MAX_CHARS]
                full_content = self._strip_known_prefixes(full_content)
                content = self._trim_to_complete_sentence(
                    full_content,
                    self.SUMMARY_MIN_CHARS,
                    self.SUMMARY_MAX_CHARS,
                )
                logging.debug(content)
                return {
                    "publishedDate": normalize_timestamp(row.get("Date")),
                    "title": row["Title"],
                    "content": content,
                    "site": url,
                }
            except Exception as exc:
                logging.debug(f"Unexpected error for {url}: {exc}")
                return None

    async def _fetch_all(
        self, valid_rows: List[Tuple[str, pd.Series]]
    ) -> List[ArticleRecord]:
        import aiohttp

        semaphore = asyncio.Semaphore(self.CONCURRENCY)
        connector = aiohttp.TCPConnector(limit=self.CONNECTION_LIMIT)

        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self._fetch_and_parse(session, url, row, semaphore)
                for url, row in valid_rows
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

    def _collect_valid_rows(self, news_data: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
        valid: List[Tuple[str, pd.Series]] = []
        for _, row in news_data.iterrows():
            url = row["Link"]
            if not url or pd.isna(url):
                logging.debug(f"Skipping invalid URL: {url}")
                continue

            if url.startswith("/"):
                url = "https://finviz.com" + url

            if not urlparse(url).scheme:
                logging.debug(f"Skipping URL without scheme: {url}")
                continue

            valid.append((url, row))
        return valid

    def scrape(self) -> Iterator[ArticleRecord]:  # type: ignore[override]
        import finvizfinance.quote as ff

        fnews = ff.finvizfinance(self.symbol)
        news_data = fnews.ticker_news()
        valid_rows = self._collect_valid_rows(news_data)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(self._fetch_all(valid_rows))
            for result in tqdm(
                results,
                desc="Scraping Finviz News articles...",
                total=len(results),
            ):
                if result and not isinstance(result, Exception):
                    yield result
        finally:
            loop.close()
