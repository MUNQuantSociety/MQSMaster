"""Financial Modeling Prep paged news scraper with persistent fetch state."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd
import requests

from NLP.core import (
    ARTICLES_DIR,
    STATE_DIR,
    ensure_project_root_on_path,
    get_logger,
    normalize_published_date_column,
)
from NLP.scrapers.base import ArticleRecord, BaseNewsScraper

ensure_project_root_on_path()
from src.common.articles_gateway import ArticlesGateway  # noqa: E402

logger = get_logger(__name__)

_DATE_FMT = "%Y-%m-%d"
_FMP_DATE_FMT = "%Y-%m-%d %H:%M:%S"


class FmpFetchStateStore:
    """Persist the next-page cursor for each ticker between fetch runs."""

    def __init__(self, state_dir: Path | str = STATE_DIR):
        self.state_dir = Path(state_dir)

    def _path(self, ticker: str) -> Path:
        return self.state_dir / f"{ticker}_state.json"

    def load(self, ticker: str, user_start: datetime, user_end: datetime) -> int:
        """Return the next page to fetch, or 0 if state is missing/stale."""
        path = self._path(ticker)
        if not path.exists():
            return 0

        try:
            with open(path) as f:
                state = json.load(f)
        except Exception:
            return 0

        if state.get("start_date") != user_start.strftime(_DATE_FMT) or state.get(
            "end_date"
        ) != user_end.strftime(_DATE_FMT):
            return 0

        return int(state.get("next_start_page", 0))

    def save(
        self,
        ticker: str,
        next_start_page: int,
        user_start: datetime,
        user_end: datetime,
    ) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with open(self._path(ticker), "w") as f:
            json.dump(
                {
                    "next_start_page": next_start_page,
                    "start_date": user_start.strftime(_DATE_FMT),
                    "end_date": user_end.strftime(_DATE_FMT),
                },
                f,
            )


class FmpNewsScraper(BaseNewsScraper):
    """Paginated news scraper backed by Financial Modeling Prep.

    Handles incremental fetching (page state persists between runs) and
    appends rows to the per-ticker CSV under ``NLP/articles/``.
    """

    MAX_PAGES_PER_RUN: int = 50
    RATE_LIMIT_SECONDS: float = 0.2

    def __init__(
        self,
        symbol: str,
        gateway: Optional[ArticlesGateway] = None,
        state_store: Optional[FmpFetchStateStore] = None,
        articles_dir: Path | str = ARTICLES_DIR,
    ):
        super().__init__(symbol)
        self.gateway = gateway or ArticlesGateway()
        self.state_store = state_store or FmpFetchStateStore()
        self.articles_dir = Path(articles_dir)

    @property
    def csv_path(self) -> Path:
        return self.articles_dir / f"{self.symbol}.csv"

    def fetch_page_window(
        self, user_start: datetime, user_end: datetime, start_page: int = 0
    ) -> tuple[List[ArticleRecord], bool, int]:
        """Fetch up to ``MAX_PAGES_PER_RUN`` pages.

        Returns ``(articles, hit_max_pages, next_start_page)`` so callers
        can checkpoint state between runs.
        """
        logger.info(f"[{self.symbol}] Fetching articles from page {start_page}...")
        all_articles: List[ArticleRecord] = []
        page = start_page
        reached_start_date = False

        while page < start_page + self.MAX_PAGES_PER_RUN:
            try:
                news = self.gateway.fetch_fmp_news(self.symbol, page)
            except requests.RequestException as e:
                logger.warning(f"[{self.symbol}] Request error on page {page}: {e}")
                break

            if not news:
                logger.info(f"[{self.symbol}] No more articles found from API.")
                break

            for art in news:
                pd_str = art.get("publishedDate")
                try:
                    art_date = datetime.strptime(pd_str, _FMP_DATE_FMT)
                except (ValueError, TypeError):
                    continue

                if art_date < user_start:
                    reached_start_date = True
                    break

                if art_date <= user_end:
                    all_articles.append(
                        {
                            "publishedDate": art_date,
                            "title": (art.get("title") or "").strip(),
                            "content": (
                                art.get("text") or art.get("content") or ""
                            ).strip(),
                            "site": (art.get("url") or "").strip(),
                        }
                    )

            if reached_start_date:
                logger.info(f"[{self.symbol}] Reached the start date boundary.")
                break

            page += 1
            time.sleep(self.RATE_LIMIT_SECONDS)

        hit_max_pages = page == start_page + self.MAX_PAGES_PER_RUN
        next_start_page = page

        if hit_max_pages:
            logger.info(
                f"[{self.symbol}] Reached page limit for this run. Next start page: {next_start_page}"
            )
        else:
            logger.info(
                f"[{self.symbol}] Finished fetching available pages. Total pages: {page - start_page}"
            )

        return all_articles, hit_max_pages, next_start_page

    def update_csv(self, start_date_str: str, end_date_str: str) -> Path:
        """Run fetch cycles until pagination is exhausted, append to CSV."""
        self.articles_dir.mkdir(parents=True, exist_ok=True)

        try:
            user_start = datetime.strptime(start_date_str, _DATE_FMT)
            user_end = datetime.strptime(end_date_str, _DATE_FMT).replace(
                hour=23, minute=59, second=59
            )
        except ValueError as exc:
            raise ValueError("Dates must be in YYYY-MM-DD format.") from exc

        start_page = self.state_store.load(self.symbol, user_start, user_end)
        has_more_pages = True
        run_count = 1

        while has_more_pages:
            logger.info(
                f"\n--- Starting fetch cycle #{run_count} (starting page: {start_page}) ---"
            )

            articles, hit_max_pages, next_page = self.fetch_page_window(
                user_start, user_end, start_page
            )

            if articles:
                self._merge_and_save(articles)
            else:
                logger.info(f"[{self.symbol}] No new articles found in this batch")

            start_page = next_page
            has_more_pages = hit_max_pages
            self.state_store.save(self.symbol, start_page, user_start, user_end)
            run_count += 1

        logger.info(f"\n[{self.symbol}] Fetching completed successfully!")
        return self.csv_path

    def _merge_and_save(self, articles: List[ArticleRecord]) -> None:
        new_df = normalize_published_date_column(pd.DataFrame(articles))

        if self.csv_path.exists():
            try:
                old_df = pd.read_csv(self.csv_path, parse_dates=["publishedDate"])
                old_df = normalize_published_date_column(old_df)
                combined = pd.concat([old_df, new_df], ignore_index=True)
            except pd.errors.EmptyDataError:
                combined = new_df
        else:
            combined = new_df

        combined = normalize_published_date_column(combined)

        initial_count = len(combined)
        combined.drop_duplicates(
            subset=["publishedDate", "title"], keep="first", inplace=True
        )
        combined.sort_values("publishedDate", ascending=False, inplace=True)
        combined.to_csv(self.csv_path, index=False, date_format=_FMP_DATE_FMT)

        added = len(combined) - (initial_count - len(new_df))
        duplicates_removed = initial_count - len(combined)
        logger.info(
            f"[{self.symbol}] Added {added} new articles, removed {duplicates_removed} duplicates"
        )

    def scrape(  # type: ignore[override]
        self, start_date_str: str, end_date_str: str
    ) -> Iterator[ArticleRecord]:
        """Generator interface: re-yield rows from the merged CSV.

        ``update_csv`` already performs deduping + persistence; this
        method just streams the rows for callers that prefer iteration.
        """
        path = self.update_csv(start_date_str, end_date_str)
        if not path.exists():
            return
        df = pd.read_csv(path, parse_dates=["publishedDate"])
        for record in df.to_dict(orient="records"):
            yield record
