#!/usr/bin/env python3
"""NLP article scraper daemon.

Long-running entrypoint. Wraps :class:`TickerUniverse` + per-ticker
fetch / sentiment subprocess calls in a class with explicit memory and
log-rotation hooks. Keeps the legacy CLI: ``python NLP/daemon.py start``.
"""

from __future__ import annotations

import gc
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil

# When invoked as a script (``python NLP/daemon.py``) Python does not put
# the repo root on sys.path, so ``import NLP.core`` fails. Bootstrap the
# path before any NLP imports.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from NLP.core import (
    DAEMON_LOG_FILE,
    PROJECT_ROOT,
    ensure_project_root_on_path,
)
from NLP.core.logging_config import configure_rotating_file_logger
from NLP.orchestration import TickerUniverse

ensure_project_root_on_path()


# ----------------------------------------------------------------------
# Module-level constants kept for back-compat with monitor_daemon and
# any external callers that imported them.
# ----------------------------------------------------------------------

NUM_PORTFOLIOS = 4
EXCLUDED_TICKERS = {"^VIX"}

SCRAPE_INTERVAL = 300  # seconds between cycles
BATCH_INTERVAL = 120   # seconds between batches inside a cycle

MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 1
MEMORY_CLEANUP_INTERVAL = 10
SKIP_THRESHOLD = 0.8

SCRIPT_DIR = Path(__file__).parent
LOG_FILE = DAEMON_LOG_FILE


# ----------------------------------------------------------------------
# Module-level helpers retained for backward compatibility (imported by
# monitor_daemon and a few existing scripts).
# ----------------------------------------------------------------------

_universe = TickerUniverse()


def load_tickers_from_portfolios() -> list:
    """Legacy free-function wrapper around :class:`TickerUniverse`."""
    return _universe.load_tickers()


def build_batches(tickers: list) -> list:
    """Legacy free-function wrapper around :meth:`TickerUniverse.build_batches`."""
    return TickerUniverse.build_batches(tickers, num_batches=4)


_logger = configure_rotating_file_logger(
    LOG_FILE, max_bytes=MAX_LOG_SIZE, backup_count=LOG_BACKUP_COUNT
)


# Ensure subsequent module-level prints / log_message calls have something
# to land on if downstream code directly imports them.
_all_tickers = load_tickers_from_portfolios()
_batches = build_batches(_all_tickers)
BATCH_1 = _batches[0] if len(_batches) > 0 else []
BATCH_2 = _batches[1] if len(_batches) > 1 else []
BATCH_3 = _batches[2] if len(_batches) > 2 else []
BATCH_4 = _batches[3] if len(_batches) > 3 else []


def log_message(message: str) -> None:
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
    _logger.info(f"{message} [Memory: {memory_usage:.1f}MB]")


def cleanup_memory() -> None:
    gc.collect()
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
    log_message(f"Memory cleanup completed - Current usage: {memory_usage:.1f}MB")


# ----------------------------------------------------------------------
# Class-based implementation
# ----------------------------------------------------------------------


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
            log_message(
                f"Intelligent skip for {ticker} - only {recent_new}/5 recent cycles had new articles"
            )
            return True
        return False

    def summary(self) -> Tuple[int, int]:
        total_skips = sum(
            len([x for x in bucket if not x]) for bucket in self._stats.values()
        )
        total_processes = sum(len(bucket) for bucket in self._stats.values())
        return total_skips, total_processes


_NEW_ARTICLE_RE_TEMPLATE = r"\[{ticker}\] Added (\d+) new articles"


def check_for_new_articles(fetch_output: str, ticker: str) -> bool:
    """Parse fetch_articles.py stdout to determine if new articles were found."""
    pattern = re.compile(_NEW_ARTICLE_RE_TEMPLATE.format(ticker=re.escape(ticker)))
    for line in fetch_output.split("\n"):
        match = pattern.search(line)
        if match:
            count = int(match.group(1))
            log_message(f"{ticker}: {count} new articles found")
            return count > 0

        if f"[{ticker}] 0 new articles" in line:
            log_message(f"{ticker}: 0 new articles (duplicates)")
            return False

        if f"[{ticker}] No new articles found" in line:
            log_message(f"{ticker}: no new articles")
            return False

    log_message(f"{ticker}: could not parse fetch output, skipping")
    return False


class TickerBatchProcessor:
    """Run fetch + sentiment subprocesses for one batch of tickers."""

    FETCH_TIMEOUT_SECONDS = 300
    SENTIMENT_TIMEOUT_SECONDS = 600
    INTER_TICKER_PAUSE_SECONDS = 5

    def __init__(self, skip_tracker: SkipStatsTracker, project_root: Path = PROJECT_ROOT):
        self.skip_tracker = skip_tracker
        self.project_root = project_root

    @staticmethod
    def _get_date_range() -> Tuple[str, str]:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        return start_date, end_date

    def _run_fetch(self, ticker: str, start_date: str, end_date: str) -> Optional[subprocess.CompletedProcess]:
        cmd = [
            sys.executable,
            "-m",
            "NLP.fetch_articles",
            ticker,
            start_date,
            end_date,
        ]
        try:
            return subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.FETCH_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            log_message(f"ERROR: Timeout processing {ticker}")
            self.skip_tracker.record(ticker, False)
            return None

    def _run_sentiment(self, ticker: str) -> bool:
        cmd = [
            sys.executable,
            "-m",
            "NLP.process_sentiment_pipeline",
            ticker,
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.SENTIMENT_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            log_message(f"ERROR: Timeout running sentiment for {ticker}")
            return False

        if result.returncode == 0:
            log_message(
                f"Successfully processed sentiment and updated database for {ticker}"
            )
            return True
        log_message(f"ERROR: Failed to process sentiment for {ticker}")
        log_message(f"Error output: {result.stderr}")
        return False

    def process(self, batch_name: str, tickers: List[str]) -> None:
        log_message(f"Starting batch {batch_name} with tickers: {tickers}")
        start_date, end_date = self._get_date_range()
        batch_start_time = time.time()

        for ticker in tickers:
            ticker_start_time = time.time()
            log_message(f"Processing ticker: {ticker}")

            try:
                log_message(f"Step 1: Fetching articles for {ticker}")
                fetch_result = self._run_fetch(ticker, start_date, end_date)
                if fetch_result is None:
                    continue

                if fetch_result.returncode != 0:
                    log_message(f"ERROR: Failed to fetch articles for {ticker}")
                    log_message(f"Error output: {fetch_result.stderr}")
                    self.skip_tracker.record(ticker, False)
                else:
                    new_articles_found = check_for_new_articles(
                        fetch_result.stdout, ticker
                    )
                    self.skip_tracker.record(ticker, new_articles_found)

                    if new_articles_found:
                        log_message(
                            f"Successfully fetched articles for {ticker} - NEW ARTICLES FOUND"
                        )
                        if self.skip_tracker.should_skip(ticker):
                            log_message(
                                f"INTELLIGENT SKIP: Skipping sentiment processing for {ticker} based on recent history"
                            )
                        else:
                            log_message(
                                f"Step 2: Processing sentiment and updating database for {ticker}"
                            )
                            self._run_sentiment(ticker)
                    else:
                        log_message(
                            f"SKIPPING {ticker} - No new articles found since last cycle"
                        )
            except Exception as exc:
                log_message(f"ERROR: Exception processing {ticker}: {exc}")
                self.skip_tracker.record(ticker, False)

            ticker_duration = int(time.time() - ticker_start_time)
            log_message(f"Ticker {ticker} processed in {ticker_duration}s")
            time.sleep(self.INTER_TICKER_PAUSE_SECONDS)

        batch_duration = int(time.time() - batch_start_time)
        log_message(f"Completed batch {batch_name} in {batch_duration}s")


class NLPDaemon:
    """Long-running NLP scraper / sentiment loop."""

    def __init__(
        self,
        universe: Optional[TickerUniverse] = None,
        scrape_interval: int = SCRAPE_INTERVAL,
        batch_interval: int = BATCH_INTERVAL,
        memory_cleanup_interval: int = MEMORY_CLEANUP_INTERVAL,
        skip_tracker: Optional[SkipStatsTracker] = None,
    ):
        self.universe = universe or TickerUniverse()
        self.scrape_interval = scrape_interval
        self.batch_interval = batch_interval
        self.memory_cleanup_interval = memory_cleanup_interval
        self.skip_tracker = skip_tracker or SkipStatsTracker()
        self.batch_processor = TickerBatchProcessor(self.skip_tracker)
        self.cycle_count = 0

    def run_cycle(self, batches: Optional[List[List[str]]] = None) -> bool:
        """Execute one full scraping cycle. Returns True on success."""
        self.cycle_count += 1
        log_message(f"=== Starting scraping cycle #{self.cycle_count} ===")
        cycle_start = time.time()

        try:
            cycle_batches = (
                batches if batches is not None else self.universe.load_batches()
            )

            if not cycle_batches:
                log_message("No tickers available for this cycle")

            for idx, batch in enumerate(cycle_batches, start=1):
                self.batch_processor.process(str(idx), batch)
                if idx < len(cycle_batches):
                    next_batch = idx + 1
                    log_message(
                        f"Waiting {self.batch_interval} seconds before batch {next_batch}..."
                    )
                    time.sleep(self.batch_interval)

            cycle_duration = int(time.time() - cycle_start)
            log_message(
                f"=== Scraping cycle #{self.cycle_count} completed in {cycle_duration} seconds ==="
            )

            if self.cycle_count % self.memory_cleanup_interval == 0:
                cleanup_memory()

            total_skips, total_processes = self.skip_tracker.summary()
            if total_processes > 0:
                skip_rate = (total_skips / total_processes) * 100
                log_message(
                    f"Current skip rate: {skip_rate:.1f}% ({total_skips}/{total_processes})"
                )
            return True

        except Exception as exc:
            log_message(f"ERROR: Exception in scraping cycle: {exc}")
            return False

    def start(self, batches: Optional[List[List[str]]] = None) -> None:
        """Run :meth:`run_cycle` forever."""
        startup_batches = batches if batches is not None else self.universe.load_batches()

        log_message(
            "Starting NLP scraper daemon with batched processing and VM optimization..."
        )
        if startup_batches:
            log_message(f"Startup batch plan: {len(startup_batches)} batches")
            for idx, batch in enumerate(startup_batches, start=1):
                log_message(f"Batch {idx}: {batch}")
        else:
            log_message("Startup batch plan: no tickers available")

        log_message(f"Interval: Every {self.scrape_interval} seconds")
        log_message(f"Batch delay: {self.batch_interval} seconds between batches")
        log_message(
            f"VM Optimizations: Log rotation ({MAX_LOG_SIZE / 1024 / 1024:.0f}MB), "
            f"Memory cleanup every {self.memory_cleanup_interval} cycles"
        )

        cleanup_memory()

        while True:
            try:
                cycle_start = time.time()
                if self.run_cycle(batches=batches):
                    log_message("Scraping cycle completed successfully")
                else:
                    log_message("ERROR: Scraping cycle failed")

                cycle_duration = int(time.time() - cycle_start)
                sleep_time = self.scrape_interval - cycle_duration

                if sleep_time > 0:
                    log_message(f"Sleeping for {sleep_time} seconds until next cycle...")
                    time.sleep(sleep_time)
                else:
                    log_message(
                        f"WARNING: Cycle took longer than interval "
                        f"({cycle_duration} > {self.scrape_interval} seconds)"
                    )
                    log_message(
                        "Consider increasing SCRAPE_INTERVAL or optimizing processing time"
                    )
            except KeyboardInterrupt:
                log_message("Daemon stopped by user")
                break
            except Exception as exc:
                log_message(f"ERROR: Exception in daemon loop: {exc}")
                log_message("Waiting 60 seconds before retry...")
                time.sleep(60)


# ----------------------------------------------------------------------
# Backwards-compatible free-function wrappers (used by tests, scripts,
# and a couple of legacy callers in this repo).
# ----------------------------------------------------------------------


_default_daemon = NLPDaemon()


def run_ticker_batch(batch_name: str, tickers: List[str]) -> None:
    _default_daemon.batch_processor.process(batch_name, tickers)


def update_skip_stats(ticker: str, had_new_articles: bool) -> None:
    _default_daemon.skip_tracker.record(ticker, had_new_articles)


def should_skip_sentiment(ticker: str) -> bool:
    return _default_daemon.skip_tracker.should_skip(ticker)


def get_date_range() -> Tuple[str, str]:
    return TickerBatchProcessor._get_date_range()


def run_scraping_cycle(batches: Optional[List[List[str]]] = None) -> bool:
    return _default_daemon.run_cycle(batches=batches)


def start_daemon(batches: Optional[List[List[str]]] = None) -> None:
    _default_daemon.start(batches=batches)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "start":
        start_daemon()
        return

    print("NLP Article Scraper Daemon - VM Optimized")
    print("=" * 50)
    print("Usage: python daemon.py start")
    print()
    print("This daemon runs 24/7 with the following optimizations:")
    help_batches = TickerUniverse().load_batches()
    print(f"• Batched processing ({len(help_batches)} batches): {help_batches}")
    print(
        f"• Runs every {SCRAPE_INTERVAL} seconds with {BATCH_INTERVAL}s between batches"
    )
    print("• Intelligent skip logic to avoid unnecessary processing")
    print(
        f"• Log rotation ({MAX_LOG_SIZE / 1024 / 1024:.0f}MB max, {LOG_BACKUP_COUNT} backups)"
    )
    print(f"• Memory cleanup every {MEMORY_CLEANUP_INTERVAL} cycles")
    print(f"• Skip threshold: {SKIP_THRESHOLD * 100:.0f}% for sentiment processing")


if __name__ == "__main__":
    main()
