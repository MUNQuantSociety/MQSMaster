"""Alpha Vantage NEWS_SENTIMENT endpoint scraper."""

from __future__ import annotations

from typing import Iterable, Iterator, Optional

from tqdm import tqdm

from NLP.core import (
    ensure_project_root_on_path,
    get_logger,
    normalize_timestamp,
)
from NLP.scrapers.base import ArticleRecord, BaseNewsScraper

ensure_project_root_on_path()
from src.common.articles_gateway import ArticlesGateway  # noqa: E402

logger = get_logger(__name__)


class AlphaVantageNewsScraper(BaseNewsScraper):
    """Fetch news + sentiment from Alpha Vantage for a list of tickers."""

    DEFAULT_TIME_FROM = "20251201T1200"
    DEFAULT_TIME_TO = "20251231T1200"

    def __init__(
        self,
        symbol: str,
        gateway: Optional[ArticlesGateway] = None,
    ):
        super().__init__(symbol)
        self.gateway = gateway or ArticlesGateway()

    def scrape(  # type: ignore[override]
        self,
        ticker: Optional[Iterable[str]] = None,
        time_from: str = DEFAULT_TIME_FROM,
        time_to: str = DEFAULT_TIME_TO,
    ) -> Iterator[ArticleRecord]:
        tickers = list(ticker) if ticker is not None else [self.symbol]
        payload = self.gateway.fetch_alpha_news(tickers, time_from, time_to)
        feed = (payload or {}).get("feed", []) or []

        for item in tqdm(
            feed,
            desc="Scraping Alpha Vantage News articles...",
            total=len(feed),
        ):
            title = item.get("title", "N/A")
            summary = item.get("summary", "N/A")
            pub_date_raw = item.get("time_published", "N/A")
            pub_date = normalize_timestamp(
                None if pub_date_raw == "N/A" else pub_date_raw
            )
            url = item.get("url", "N/A")

            yield {
                "publishedDate": pub_date,
                "title": title,
                "content": summary,
                "site": url,
            }
