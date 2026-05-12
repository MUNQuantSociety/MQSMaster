"""Backwards-compatible module for the legacy ``ArticleScraper`` facade.

Per-source scrapers now live under :mod:`NLP.scrapers`. This module
keeps a thin facade so external imports such as
``from NLP.fetch_alt_articles import ArticleScraper, normalize_published_date_column``
continue to resolve.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Mapping

from NLP.core import (
    ensure_project_root_on_path,
    get_logger,
    normalize_published_date_column,
    normalize_timestamp,
)
from NLP.scrapers.aggregator import ArticleAggregator
from NLP.scrapers.alpha_vantage import AlphaVantageNewsScraper
from NLP.scrapers.finviz import FinvizNewsScraper
from NLP.scrapers.truth_social import TruthSocialScraper
from NLP.scrapers.yahoo import YahooNewsScraper

ensure_project_root_on_path()

logger = get_logger(__name__)

# Re-exported for callers that imported these alongside ArticleScraper.
__all__ = [
    "ArticleScraper",
    "normalize_published_date_column",
    "normalize_timestamp",
]


class ArticleScraper:
    """Multi-source scraper facade.

    The legacy class did everything in one place. The new layout has one
    class per source under :mod:`NLP.scrapers`; this facade keeps the
    public surface (``scrape_yahoo``, ``scrape_finviz``, ``scrape_alpha``,
    ``trump_tracker``) so external callers keep working.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self._yahoo = YahooNewsScraper(self.symbol)
        self._finviz = FinvizNewsScraper(self.symbol)
        self._alpha = AlphaVantageNewsScraper(self.symbol)
        self._truth = TruthSocialScraper(self.symbol)

    # Each scrape_* helper is a thin generator delegate that yields the
    # same dict schema as the legacy implementation.
    def scrape_yahoo(self) -> Iterator[Mapping]:
        yield from self._yahoo.scrape()

    def scrape_finviz(self) -> Iterator[Mapping]:
        yield from self._finviz.scrape()

    def scrape_alpha(
        self,
        ticker: Iterable[str] | None = None,
        time_from: str = AlphaVantageNewsScraper.DEFAULT_TIME_FROM,
        time_to: str = AlphaVantageNewsScraper.DEFAULT_TIME_TO,
    ) -> Iterator[Mapping]:
        yield from self._alpha.scrape(
            ticker=ticker if ticker is not None else [self.symbol],
            time_from=time_from,
            time_to=time_to,
        )

    def trump_tracker(self) -> Iterator[Mapping]:
        yield from self._truth.scrape()

    # Helpers preserved verbatim from the legacy class.

    def get_complete_sentences(
        self, text: str, min_chars: int = 200, max_chars: int = 1000
    ) -> str:
        return FinvizNewsScraper._trim_to_complete_sentence(text, min_chars, max_chars)

    def check_duplicates(self):
        return ArticleAggregator.find_duplicates_across_per_source_csvs(self.symbol)
