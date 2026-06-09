#!/usr/bin/env python3
"""backfill_NLP.py
Historical NLP backfill for a date range.

Fetches articles (FMP all pages + Yahoo + Finviz + Alpha Vantage) for
each ticker in the universe between ``--start`` and ``--end``, scores
them with FinBERT, and writes them to ``news_sentiment``.

Hard rule: backfill MUST NOT run during US equities cash session
(09:30-16:00 ET, Mon-Fri). The runtime guard refuses to start while the
market is open. Pass ``--wait`` to block until the close instead of
exiting.

Usage:
    python NLP/backfill_NLP.py --start 2025-01-01 --end 2025-12-31
    python NLP/backfill_NLP.py --start 2025-12-01 --end 2025-12-31 --tickers AAPL MSFT
    python NLP/backfill_NLP.py --start 2025-01-01 --end 2025-12-31 --wait
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, Sequence

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from NLP.orchestration import TickerUniverse
from NLP.persistence.repository import NewsSentimentRepository
from NLP.scrapers.aggregator import ArticleAggregator
from NLP.sentiment.pipeline import SentimentPipeline
from NLP.sentiment.scorer import FinBertSentimentScorer
from src.common.market_hours import is_market_open, seconds_until_market_close

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("NLP.backfill")


_DATE_FMT = "%Y-%m-%d"


def _parse_date(value: str) -> str:
    """Validate ``value`` is a YYYY-MM-DD date; return it unchanged."""
    datetime.strptime(value, _DATE_FMT)
    return value


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NLP backfill for a historical date range."
    )
    parser.add_argument(
        "--start",
        required=True,
        type=_parse_date,
        help="Start date (YYYY-MM-DD), inclusive.",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=_parse_date,
        help="End date (YYYY-MM-DD), inclusive.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Ticker subset. Defaults to all of src/orchestrator/backfill/tickers.json.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="If market is open, sleep until 16:00 ET then start. Default: abort.",
    )
    parser.add_argument(
        "--include-trump-tracker",
        action="store_true",
        help="Also scrape Truth Social posts (requires APIFY_KEY in env).",
    )
    return parser.parse_args(argv)


class NLPBackfill:
    """Drive the article-fetch + FinBERT scoring + DB-write loop for a date range."""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        tickers: Optional[List[str]] = None,
        include_trump_tracker: bool = False,
    ):
        if start_date > end_date:
            raise ValueError(
                f"start_date {start_date} must be <= end_date {end_date}"
            )
        self.start_date = start_date
        self.end_date = end_date
        self.include_trump_tracker = include_trump_tracker

        if tickers is not None:
            self.tickers = list(tickers)
        else:
            self.tickers = TickerUniverse().load_tickers()

        logger.info("Initializing FinBERT scorer (one-time model load)...")
        self.scorer = FinBertSentimentScorer()
        self.scorer.load_model()
        self.repository = NewsSentimentRepository()
        self.pipeline = SentimentPipeline(
            scorer=self.scorer, repository=self.repository
        )

    def run(self) -> int:
        """Iterate the ticker universe, return the count of successful tickers."""
        logger.info(
            "Backfill range %s -> %s for %d tickers (trump_tracker=%s)",
            self.start_date,
            self.end_date,
            len(self.tickers),
            self.include_trump_tracker,
        )
        successes = 0
        for idx, ticker in enumerate(self.tickers, start=1):
            logger.info(
                "[%d/%d] %s: fetching all sources...",
                idx,
                len(self.tickers),
                ticker,
            )
            t0 = time.time()
            try:
                aggregator = ArticleAggregator(
                    ticker, include_trump_tracker=self.include_trump_tracker
                )
                aggregator.run(self.start_date, self.end_date)
            except Exception as exc:
                logger.error("[%s] aggregator failed: %s", ticker, exc)
                continue

            try:
                ok = self.pipeline.process_ticker_complete(ticker=ticker)
                if ok:
                    successes += 1
                    logger.info(
                        "[%s] done in %ds (success #%d)",
                        ticker,
                        int(time.time() - t0),
                        successes,
                    )
                else:
                    logger.warning("[%s] sentiment pipeline returned False", ticker)
            except Exception as exc:
                logger.error("[%s] sentiment/persist failed: %s", ticker, exc)

        logger.info(
            "Backfill complete: %d / %d tickers persisted", successes, len(self.tickers)
        )
        return successes


def _enforce_market_hours_guard(wait: bool) -> None:
    """Hard rule: backfill must not run during the US equities cash session.

    Without ``--wait`` we abort immediately when the market is open.
    With ``--wait`` we sleep until 16:00 ET (today) and resume.
    """
    if not is_market_open():
        return

    if not wait:
        logger.error(
            "Market is currently open. Backfill is blocked by the market-hours guard. "
            "Either rerun after 16:00 ET or pass --wait to block until close."
        )
        sys.exit(2)

    seconds = seconds_until_market_close()
    logger.warning(
        "Market is open. --wait given; sleeping %d seconds (%.1f min) until close...",
        seconds,
        seconds / 60.0,
    )
    time.sleep(seconds + 60)  # +60s buffer past the bell

    if is_market_open():
        logger.error(
            "Still inside market hours after waiting; aborting to be safe."
        )
        sys.exit(2)
    logger.info("Market is closed. Resuming backfill.")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _enforce_market_hours_guard(wait=args.wait)

    backfill = NLPBackfill(
        start_date=args.start,
        end_date=args.end,
        tickers=args.tickers,
        include_trump_tracker=args.include_trump_tracker,
    )
    try:
        backfill.run()
    except Exception:
        logger.critical("Backfill failed.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
