"""Per-source news article scrapers and a multi-source aggregator.

All scrapers expose ``scrape(...)`` which yields ``ArticleRecord`` dicts
with the keys ``publishedDate``, ``title``, ``content``, ``site``.
"""

from NLP.scrapers.aggregator import ArticleAggregator
from NLP.scrapers.alpha_vantage import AlphaVantageNewsScraper
from NLP.scrapers.base import ArticleRecord, BaseNewsScraper
from NLP.scrapers.finviz import FinvizNewsScraper
from NLP.scrapers.fmp import FmpFetchStateStore, FmpNewsScraper
from NLP.scrapers.truth_social import TruthSocialScraper
from NLP.scrapers.yahoo import YahooNewsScraper

__all__ = [
    "ArticleAggregator",
    "AlphaVantageNewsScraper",
    "ArticleRecord",
    "BaseNewsScraper",
    "FinvizNewsScraper",
    "FmpFetchStateStore",
    "FmpNewsScraper",
    "TruthSocialScraper",
    "YahooNewsScraper",
]
