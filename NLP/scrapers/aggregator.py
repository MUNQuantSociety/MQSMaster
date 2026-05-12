"""Combine FMP + alt-source articles into a single deduped CSV per ticker."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from NLP.core import (
    ARTICLES_DIR,
    get_logger,
    normalize_published_date_column,
)
from NLP.scrapers.alpha_vantage import AlphaVantageNewsScraper
from NLP.scrapers.finviz import FinvizNewsScraper
from NLP.scrapers.fmp import FmpNewsScraper
from NLP.scrapers.truth_social import TruthSocialScraper
from NLP.scrapers.yahoo import YahooNewsScraper

logger = get_logger(__name__)

_FMP_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def remove_duplicate_articles(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatenate frames, drop NaT rows, dedup on (publishedDate, title)."""
    normalized = [normalize_published_date_column(df) for df in frames if df is not None]
    combined = pd.concat(normalized, ignore_index=True)
    combined = normalize_published_date_column(combined)

    before = len(combined)
    combined.drop_duplicates(
        subset=["publishedDate", "title"], keep="first", inplace=True
    )
    after = len(combined)
    logger.info(
        f"Removed {before - after} duplicates; {after} unique articles remain."
    )
    return combined


class ArticleAggregator:
    """Drive every per-source scraper for a ticker and produce one CSV.

    The merged CSV is the FMP file under ``NLP/articles/<TICKER>.csv``;
    additional sources are deduped against it and appended in place. The
    same filename is what :class:`SentimentProcessor` reads downstream,
    so callers do not need to know which sources contributed.
    """

    def __init__(
        self,
        symbol: str,
        articles_dir: Path | str = ARTICLES_DIR,
        include_trump_tracker: bool = False,
    ):
        self.symbol = symbol.upper()
        self.articles_dir = Path(articles_dir)
        self.include_trump_tracker = include_trump_tracker

    @property
    def merged_csv_path(self) -> Path:
        return self.articles_dir / f"{self.symbol}.csv"

    def run(self, start_date: str, end_date: str) -> Path:
        """Fetch FMP + alt sources, merge, dedup, save the merged CSV."""
        fmp_scraper = FmpNewsScraper(self.symbol, articles_dir=self.articles_dir)
        fmp_scraper.update_csv(start_date, end_date)

        time_from = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%dT%H%M")
        time_to = (
            datetime.strptime(end_date, "%Y-%m-%d")
            .replace(hour=23, minute=59)
            .strftime("%Y%m%dT%H%M")
        )

        yahoo_df = self._collect(YahooNewsScraper(self.symbol))
        finviz_df = self._collect(FinvizNewsScraper(self.symbol))
        alpha_df = self._collect(
            AlphaVantageNewsScraper(self.symbol),
            ticker=[self.symbol],
            time_from=time_from,
            time_to=time_to,
        )
        truth_df = (
            self._collect(TruthSocialScraper(self.symbol))
            if self.include_trump_tracker
            else pd.DataFrame()
        )

        return self._merge(yahoo_df, finviz_df, alpha_df, truth_df)

    def _collect(self, scraper, **kwargs) -> pd.DataFrame:
        try:
            records = list(scraper.scrape(**kwargs))
        except RuntimeError as exc:
            logger.warning(f"Skipping {scraper.__class__.__name__}: {exc}")
            return pd.DataFrame()
        df = pd.DataFrame(records)
        return normalize_published_date_column(df) if not df.empty else df

    def _merge(
        self,
        yahoo_df: pd.DataFrame,
        finviz_df: pd.DataFrame,
        alpha_df: pd.DataFrame,
        truth_df: pd.DataFrame,
    ) -> Path:
        path = self.merged_csv_path

        try:
            fmp_df = pd.read_csv(path, parse_dates=["publishedDate"])
            fmp_df = normalize_published_date_column(fmp_df)
            logger.info(f"Loaded existing FMP data: {len(fmp_df)} articles")
        except FileNotFoundError:
            logger.info(f"No existing FMP data found for {self.symbol}")
            fmp_df = pd.DataFrame()

        all_frames = []
        if fmp_df is not None and not fmp_df.empty:
            all_frames.append(fmp_df)
            logger.info(f"FMP articles: {len(fmp_df)}")
        if not yahoo_df.empty:
            all_frames.append(yahoo_df)
            logger.info(f"Yahoo articles: {len(yahoo_df)}")
        if not finviz_df.empty:
            all_frames.append(finviz_df)
            logger.info(f"Finviz articles: {len(finviz_df)}")
        if not alpha_df.empty:
            all_frames.append(alpha_df)
            logger.info(f"Alpha Vantage articles: {len(alpha_df)}")
        if not truth_df.empty and self.include_trump_tracker:
            all_frames.append(truth_df)
            logger.info(f"Trump Tracker articles: {len(truth_df)}")

        if not all_frames:
            logger.info(f"No articles found for {self.symbol}")
            return path

        combined = remove_duplicate_articles(*all_frames)
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(path, index=False, date_format=_FMP_DATE_FMT)
        logger.info(
            f"Saved merged articles to {path}: {len(combined)} total articles"
        )
        return path

    @staticmethod
    def find_duplicates_across_per_source_csvs(
        symbol: str, articles_dir: Path | str = ARTICLES_DIR
    ) -> set[str]:
        """Diagnostic: count duplicate titles between merged + per-source CSVs.

        Mirrors the legacy ``ArticleScraper.check_duplicates`` helper so
        ad-hoc analysis scripts keep working. Reads
        ``<symbol>.csv``, ``<symbol>_alpha_news.csv``,
        ``<symbol>_finviz_news.csv``, ``<symbol>_yahoo_news.csv``.
        """
        import re

        articles_dir = Path(articles_dir)
        merged = pd.read_csv(articles_dir / f"{symbol}.csv")
        alpha = pd.read_csv(articles_dir / f"{symbol}_alpha_news.csv")
        finviz = pd.read_csv(articles_dir / f"{symbol}_finviz_news.csv")
        yahoo = pd.read_csv(articles_dir / f"{symbol}_yahoo_news.csv")

        def normalize(title: object) -> str:
            return re.sub(r"[^a-zA-Z0-9]", "", str(title)).lower()

        sets = [
            {normalize(t) for t in df["title"]}
            for df in (merged, alpha, finviz, yahoo)
        ]
        duplicates = sets[0].intersection(*sets[1:])
        total_titles = sum(len(s) for s in sets)
        logger.info(f"Found {len(duplicates)}/{total_titles} duplicate titles.")
        return duplicates
