"""NLP pipeline runner.

In-process loop that sweeps the active ticker universe every 5 minutes,
scrapes the newest articles from FMP, scores them with FinBERT, and
writes results to ``news_sentiment``. Slow alt sources
(Yahoo / Finviz / Alpha Vantage) are round-robin'd across cycles so each
ticker still gets a multi-source refresh roughly hourly without blowing
the 5-minute budget.

Architecture choices that matter:

* Model loaded once at runner startup, shared across all tickers in the
  process. (Subprocess-per-ticker reloaded FinBERT ~1900x per cycle.)
* :class:`FMPRateLimiter.for_nlp` enforces a 2000 req/min cap; the
  realtime ingestor holds the other 1000 of the 3000 account budget via
  :meth:`FMPRateLimiter.for_realtime`.
* Backfill (historical date range) lives in ``NLP/backfill_NLP.py`` and
  is gated by market hours; this runner is intended to run continuously.
"""

from __future__ import annotations

import gc
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import psutil

from NLP.core import (
    DAEMON_LOG_FILE,
    PROJECT_ROOT,
    ensure_project_root_on_path,
)
from NLP.core.logging_config import configure_rotating_file_logger
from NLP.orchestration import TickerUniverse
from NLP.persistence.repository import NewsSentimentRepository
from NLP.scrapers.aggregator import ArticleAggregator
from NLP.sentiment.pipeline import SentimentPipeline
from NLP.sentiment.scorer import FinBertSentimentScorer

ensure_project_root_on_path()


SCRAPE_INTERVAL = 300        # seconds between cycles
MAX_LOG_SIZE = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 1
MEMORY_CLEANUP_INTERVAL = 10
SKIP_THRESHOLD = 0.8
DEFAULT_NUM_BATCHES = 4
LIVE_LOOKBACK_DAYS = 7


_logger = configure_rotating_file_logger(
    DAEMON_LOG_FILE, max_bytes=MAX_LOG_SIZE, backup_count=LOG_BACKUP_COUNT
)


def log_message(message: str) -> None:
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
    _logger.info(f"{message} [Memory: {memory_usage:.1f}MB]")


def cleanup_memory() -> None:
    gc.collect()
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
    log_message(f"Memory cleanup completed - Current usage: {memory_usage:.1f}MB")


class SkipStatsTracker:
    """Track per-ticker article-found history to drive intelligent skipping."""

    def __init__(self, threshold: float = SKIP_THRESHOLD, history: int = 10):
        self.threshold = threshold
        self.history = history
        self._stats: Dict[str, List[bool]] = {}

    def record(self, ticker: str, had_new_articles: bool) -> None:
        bucket = self._stats.setdefault(ticker, [])
        if len(bucket) >= self.history:
            bucket.pop(0)
        bucket.append(had_new_articles)

    def should_skip(self, ticker: str) -> bool:
        bucket = self._stats.get(ticker, [])
        if len(bucket) < 5:
            return False
        recent_new = sum(bucket[-5:])
        new_rate = recent_new / 5
        if new_rate < (1 - self.threshold):
            return True
        return False

    def summary(self) -> Tuple[int, int]:
        total_skips = sum(
            len([x for x in bucket if not x]) for bucket in self._stats.values()
        )
        total_processes = sum(len(bucket) for bucket in self._stats.values())
        return total_skips, total_processes


class TickerProcessor:
    """In-process fetch + score + DB-write for one ticker."""

    def __init__(
        self,
        pipeline: SentimentPipeline,
        skip_tracker: SkipStatsTracker,
        lookback_days: int = LIVE_LOOKBACK_DAYS,
    ):
        self.pipeline = pipeline
        self.skip_tracker = skip_tracker
        self.lookback_days = lookback_days

    def _date_range(self) -> Tuple[str, str]:
        end = datetime.now()
        start = end - timedelta(days=self.lookback_days)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    def process(self, ticker: str, include_alt_sources: bool) -> None:
        aggregator = ArticleAggregator(ticker)
        csv_before = self._row_count(aggregator.merged_csv_path)

        try:
            if include_alt_sources:
                start, end = self._date_range()
                aggregator.run(start, end)
            else:
                aggregator.run_fmp_only(days_back=self.lookback_days)
        except Exception as exc:
            log_message(f"ERROR fetching for {ticker}: {exc}")
            self.skip_tracker.record(ticker, False)
            return

        csv_after = self._row_count(aggregator.merged_csv_path)
        had_new = csv_after > csv_before
        self.skip_tracker.record(ticker, had_new)

        if not had_new:
            return

        if self.skip_tracker.should_skip(ticker):
            log_message(
                f"INTELLIGENT SKIP: sentiment skipped for {ticker} (low recent activity)"
            )
            return

        try:
            ok = self.pipeline.process_ticker_complete(ticker=ticker)
            if ok:
                log_message(f"Scored + persisted sentiment for {ticker}")
            else:
                log_message(f"Sentiment pipeline returned False for {ticker}")
        except Exception as exc:
            log_message(f"ERROR scoring/persisting {ticker}: {exc}")

    @staticmethod
    def _row_count(path: Path) -> int:
        if not path.exists():
            return 0
        # Cheap row count: total lines minus header. Safe enough for the
        # "had_new" signal even though it counts all rows, not just rows
        # added this cycle (dedup happens inside the aggregator).
        try:
            with open(path) as f:
                return max(0, sum(1 for _ in f) - 1)
        except OSError:
            return 0


class NLPRunner:
    """Long-running NLP scraper / sentiment loop.

    Drives ticker discovery → article fetch → FinBERT scoring →
    ``news_sentiment`` insert → ``market_data`` sentiment sync.

    Designed to be instantiated and ``.run()``-driven from
    :mod:`NLP.main_NLP`; mirrors the role of
    :class:`live_trading.engine.RunEngine` on the trading side.
    """

    def __init__(
        self,
        universe: Optional[TickerUniverse] = None,
        scrape_interval: int = SCRAPE_INTERVAL,
        memory_cleanup_interval: int = MEMORY_CLEANUP_INTERVAL,
        skip_tracker: Optional[SkipStatsTracker] = None,
        num_batches: int = DEFAULT_NUM_BATCHES,
        lookback_days: int = LIVE_LOOKBACK_DAYS,
    ):
        self.universe = universe or TickerUniverse()
        self.scrape_interval = scrape_interval
        self.memory_cleanup_interval = memory_cleanup_interval
        self.skip_tracker = skip_tracker or SkipStatsTracker()
        self.num_batches = num_batches
        self.cycle_count = 0
        self.alt_rotation_idx = 0

        log_message("Initializing FinBERT scorer (one-time model load)...")
        self.scorer = FinBertSentimentScorer()
        self.scorer.load_model()
        self.repository = NewsSentimentRepository()
        self.pipeline = SentimentPipeline(
            scorer=self.scorer, repository=self.repository
        )
        self.processor = TickerProcessor(
            self.pipeline, self.skip_tracker, lookback_days=lookback_days
        )
        log_message("Runner initialized")

    def run_cycle(self, tickers_override: Optional[Sequence[str]] = None) -> bool:
        """Execute one full scraping cycle. Returns True on success."""
        self.cycle_count += 1
        cycle_start = time.time()
        log_message(f"=== Starting cycle #{self.cycle_count} ===")

        try:
            tickers = (
                list(tickers_override)
                if tickers_override is not None
                else self.universe.load_tickers()
            )
            if not tickers:
                log_message("No tickers available; skipping cycle")
                return True

            batches = TickerUniverse.build_batches(tickers, num_batches=self.num_batches)
            alt_batch_idx = self.alt_rotation_idx % len(batches)
            self.alt_rotation_idx += 1

            log_message(
                f"Cycle plan: {len(tickers)} tickers in {len(batches)} batches; "
                f"alt-source batch this cycle = batch {alt_batch_idx + 1}"
            )

            for idx, batch in enumerate(batches):
                include_alt = idx == alt_batch_idx
                log_message(
                    f"Batch {idx + 1}/{len(batches)} ({len(batch)} tickers) "
                    f"include_alt={include_alt}"
                )
                batch_start = time.time()
                for ticker in batch:
                    self.processor.process(ticker, include_alt_sources=include_alt)
                log_message(
                    f"Batch {idx + 1} done in {int(time.time() - batch_start)}s"
                )

            cycle_duration = int(time.time() - cycle_start)
            log_message(f"=== Cycle #{self.cycle_count} done in {cycle_duration}s ===")

            if self.cycle_count % self.memory_cleanup_interval == 0:
                cleanup_memory()

            total_skips, total_processes = self.skip_tracker.summary()
            if total_processes > 0:
                skip_rate = (total_skips / total_processes) * 100
                log_message(
                    f"Skip rate: {skip_rate:.1f}% ({total_skips}/{total_processes})"
                )
            return True

        except Exception as exc:
            log_message(f"ERROR in cycle: {exc}")
            return False

    def run(self, tickers_override: Optional[Sequence[str]] = None) -> None:
        """Run :meth:`run_cycle` forever, sleeping ``scrape_interval`` between."""
        log_message("Starting NLP pipeline runner")
        log_message(
            f"Config: interval={self.scrape_interval}s, num_batches={self.num_batches}, "
            f"log_rotation={MAX_LOG_SIZE // (1024 * 1024)}MB, "
            f"memory_cleanup_every={self.memory_cleanup_interval} cycles"
        )
        cleanup_memory()

        while True:
            try:
                cycle_start = time.time()
                self.run_cycle(tickers_override=tickers_override)
                elapsed = int(time.time() - cycle_start)
                sleep_time = self.scrape_interval - elapsed
                if sleep_time > 0:
                    log_message(f"Sleeping {sleep_time}s until next cycle...")
                    time.sleep(sleep_time)
                else:
                    log_message(
                        f"WARNING: Cycle took {elapsed}s > interval "
                        f"{self.scrape_interval}s; consider raising SCRAPE_INTERVAL"
                    )
            except KeyboardInterrupt:
                log_message("Runner stopped by user")
                break
            except Exception as exc:
                log_message(f"ERROR in runner loop: {exc}; retry in 60s")
                time.sleep(60)


__all__ = [
    "LIVE_LOOKBACK_DAYS",
    "MAX_LOG_SIZE",
    "MEMORY_CLEANUP_INTERVAL",
    "NLPRunner",
    "SCRAPE_INTERVAL",
    "SKIP_THRESHOLD",
    "SkipStatsTracker",
    "TickerProcessor",
    "cleanup_memory",
    "log_message",
]
