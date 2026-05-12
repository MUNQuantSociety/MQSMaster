"""Truth Social posts scraped via the Apify dataset actor."""

from __future__ import annotations

import os
from typing import Any, Iterator, Optional

from NLP.core import get_logger, normalize_timestamp
from NLP.scrapers.base import ArticleRecord, BaseNewsScraper

logger = get_logger(__name__)

_DEFAULT_ACTOR_ID = "sTDLfdZAmte0aYlxg"
_DEFAULT_USERNAME = "realDonaldTrump"


class TruthSocialScraper(BaseNewsScraper):
    """Pull recent Truth Social posts for a target account.

    The ``symbol`` field is informational only - Apify is queried by
    ``username``. Set ``APIFY_KEY`` in the environment before use.
    """

    DEFAULT_MAX_POSTS = 20

    def __init__(
        self,
        symbol: str = "TRUMP",
        username: str = _DEFAULT_USERNAME,
        actor_id: str = _DEFAULT_ACTOR_ID,
        api_key: Optional[str] = None,
        max_posts: int = DEFAULT_MAX_POSTS,
    ):
        super().__init__(symbol)
        self.username = username
        self.actor_id = actor_id
        self.api_key = api_key or os.getenv("APIFY_KEY")
        self.max_posts = max_posts

    def _build_run_input(self) -> dict:
        return {
            "username": self.username,
            "maxPosts": self.max_posts,
            "useLastPostId": False,
            "onlyReplies": False,
            "onlyMedia": False,
            "cleanContent": True,
            "startFromId": None,
            "singlePostId": None,
        }

    @staticmethod
    def _extract_post(post_dict: Any) -> Optional[ArticleRecord]:
        if not isinstance(post_dict, dict):
            return None

        content = str(post_dict.get("content") or "").strip()
        if not content:
            return None

        return {
            "publishedDate": normalize_timestamp(post_dict.get("created_at")),
            "title": "Truth Social Post",
            "content": content,
            "site": post_dict.get("url"),
        }

    def scrape(self) -> Iterator[ArticleRecord]:  # type: ignore[override]
        from apify_client import ApifyClient

        if not self.api_key:
            raise RuntimeError(
                "Missing APIFY_KEY environment variable. Set APIFY_KEY in your "
                ".env file before running TruthSocialScraper."
            )

        client = ApifyClient(self.api_key)

        try:
            run = client.actor(self.actor_id).call(run_input=self._build_run_input())
        except Exception as exc:
            logger.error(f"Failed to run Apify actor: {exc}")
            return

        if not run or "defaultDatasetId" not in run:
            logger.warning("No dataset found in the run result")
            return

        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            if isinstance(item, dict):
                extracted = self._extract_post(item)
                if extracted:
                    yield extracted
            elif isinstance(item, list):
                for sub in item:
                    extracted = self._extract_post(sub)
                    if extracted:
                        yield extracted
