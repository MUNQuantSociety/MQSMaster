"""Backwards-compatible module for ``python -m NLP.fetch_articles``.

The implementation moved to :mod:`NLP.scrapers.fmp` (FMP scraping +
fetch state) and :mod:`NLP.scrapers.aggregator` (multi-source merge).
This module re-exports the public surface and keeps the legacy CLI
entrypoint so external callers, tests, and the daemon's
``subprocess.run([..., "-m", "NLP.fetch_articles", ...])`` keep working.
"""

from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from NLP.core import (
    ARTICLES_DIR,
    STATE_DIR,
    ensure_project_root_on_path,
    get_logger,
)
from NLP.core.timestamps import normalize_published_date
from NLP.scrapers.aggregator import ArticleAggregator, remove_duplicate_articles
from NLP.scrapers.fmp import FmpFetchStateStore, FmpNewsScraper

ensure_project_root_on_path()
from src.common.articles_gateway import ArticlesGateway  # noqa: E402

logger = get_logger(__name__)

# Legacy module-level constants kept for callers that imported them.
MAX_PAGES_PER_RUN = FmpNewsScraper.MAX_PAGES_PER_RUN
RATE_LIMIT = FmpNewsScraper.RATE_LIMIT_SECONDS
OUTPUT_DIR = str(ARTICLES_DIR)

gateway = ArticlesGateway()


def parse_args() -> argparse.Namespace:
    """Argparse entrypoint preserved for tests and the legacy CLI."""
    p = argparse.ArgumentParser(
        description="Fetch and update stock-news CSVs for a ticker within a date range."
    )
    p.add_argument("ticker", help="The ticker symbol to fetch (e.g., AAPL).")
    p.add_argument(
        "start_date", help="Start date for fetching articles, in YYYY-MM-DD format."
    )
    p.add_argument(
        "end_date", help="End date for fetching articles, in YYYY-MM-DD format."
    )
    p.add_argument(
        "--trump_tracker",
        help="Include Trump truth social post?(True/False)",
        type=bool,
        default=False,
        required=False,
    )
    return p.parse_args()


def remove_duplicates(*frames: pd.DataFrame) -> pd.DataFrame:
    """Back-compat alias for :func:`NLP.scrapers.aggregator.remove_duplicate_articles`."""
    return remove_duplicate_articles(*frames)


def fetch_news(symbol: str, start_date: datetime, end_date: datetime, start_page: int = 0):
    """Back-compat wrapper around :meth:`FmpNewsScraper.fetch_page_window`."""
    return FmpNewsScraper(symbol).fetch_page_window(start_date, end_date, start_page)


def save_fetch_state(ticker: str, next_start_page: int, start_date: datetime, end_date: datetime) -> None:
    FmpFetchStateStore().save(ticker, next_start_page, start_date, end_date)


def load_fetch_state(ticker: str, user_start: datetime, user_end: datetime) -> int:
    return FmpFetchStateStore().load(ticker, user_start, user_end)


def update_ticker_csv(symbol: str, start_date_str: str, end_date_str: str) -> None:
    FmpNewsScraper(symbol).update_csv(start_date_str, end_date_str)


def merge_all_sources(args, yahoo_news_df, finviz_news_df, alpha_news_df, tmnt_news_df):
    """Back-compat shim that mirrors the old free-function path.

    Defers to :class:`ArticleAggregator` for the FMP load + dedup +
    save logic.
    """
    aggregator = ArticleAggregator(
        args.ticker.upper(),
        include_trump_tracker=bool(getattr(args, "trump_tracker", False)),
    )
    return aggregator._merge(yahoo_news_df, finviz_news_df, alpha_news_df, tmnt_news_df)


def main() -> None:
    args = parse_args()
    aggregator = ArticleAggregator(
        args.ticker.upper(),
        include_trump_tracker=bool(args.trump_tracker),
    )
    final_path = aggregator.run(args.start_date, args.end_date)
    print(f"All articles merged into single file: {final_path}")
    print(f"Processing completed for {args.ticker.upper()}")


if __name__ == "__main__":
    main()


__all__ = [
    "ArticlesGateway",
    "MAX_PAGES_PER_RUN",
    "OUTPUT_DIR",
    "RATE_LIMIT",
    "STATE_DIR",
    "fetch_news",
    "gateway",
    "load_fetch_state",
    "main",
    "merge_all_sources",
    "normalize_published_date",
    "parse_args",
    "remove_duplicates",
    "save_fetch_state",
    "update_ticker_csv",
]
