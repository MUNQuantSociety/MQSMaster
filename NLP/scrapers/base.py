"""Abstract base class shared by every per-source news scraper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, TypedDict


class ArticleRecord(TypedDict, total=False):
    """Schema produced by every scraper.

    ``publishedDate`` is expected to be a tz-naive ``pd.Timestamp`` after
    going through :func:`NLP.core.timestamps.normalize_timestamp`.
    """

    publishedDate: Any
    title: str
    content: str
    site: str


class BaseNewsScraper(ABC):
    """Common interface for news scrapers tied to a single ticker symbol."""

    def __init__(self, symbol: str):
        if not symbol:
            raise ValueError("symbol must be a non-empty string")
        self.symbol = symbol.upper()

    @abstractmethod
    def scrape(self, *args: Any, **kwargs: Any) -> Iterator[ArticleRecord]:
        """Yield article records for ``self.symbol``."""
        raise NotImplementedError
