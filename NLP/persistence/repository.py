"""Repository for the ``news_sentiment`` table.

Wraps :class:`MQSDBConnector` with the schema-management, insert, and
read helpers used by the NLP pipeline.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from NLP.core import (
    ARTICLES_DIR,
    SCORES_DIR,
    ensure_project_root_on_path,
    get_logger,
)

ensure_project_root_on_path()

try:
    from src.common.database.MQSDBConnector import MQSDBConnector
except ImportError:
    logger = get_logger(__name__)
    logger.error(
        "Could not import MQSDBConnector. Make sure the database module is available."
    )
    sys.exit(1)

logger = get_logger(__name__)

CONTENT_SUMMARY_MAX_LENGTH = 1000

# Freshness-decay half-life for the daily weighted mean of sentiment.
# Each article's weight = content_length * 0.5^(age_seconds / SENTIMENT_HALF_LIFE_SECONDS).
# 6.5 hours == one trading session.
SENTIMENT_HALF_LIFE_SECONDS = int(6.5 * 3600)

# Default content_length for rows that pre-date the column. Backfilled rows
# already have it populated from LENGTH(content_summary); this constant is
# the fallback used by the weighted-mean SQL via COALESCE.
DEFAULT_CONTENT_LENGTH = 100

# Lookback window for the weighted-mean UPDATE. Articles older than this
# have weight ~ length * 0.5^(168h / 6.5h) ~ length * 1.6e-8 and cannot
# move the daily weighted mean by more than float noise, so we skip them
# in the SQL to keep the per-ticker UPDATE cheap. Without the cap, scoring
# scales O(all-news-ever * #tickers) every cycle. Days that fall out of
# the cap retain whatever sentiment_score they held last write.
SENTIMENT_UPDATE_LOOKBACK_DAYS = 7


class NewsSentimentRepository:
    """CRUD-style accessor for the ``news_sentiment`` table.

    Replaces the legacy ``SentimentDatabaseUpdater``. Owns table /
    index creation, idempotent bulk inserts, and the daily-mean sync
    into ``market_data.sentiment_score``.
    """

    def __init__(self, db: Optional[MQSDBConnector] = None):
        self.db = db or MQSDBConnector()
        self._ensure_table_exists()
        self._create_indexes()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def _ensure_table_exists(self) -> None:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10),
            article_url TEXT,
            published_at TIMESTAMP,
            sentiment_score FLOAT,
            content_summary TEXT,
            content_length INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        result = self.db.execute_query(create_table_sql)
        if result["status"] == "error":
            logger.error(f"Failed to create news_sentiment table: {result['message']}")
            raise Exception("Database table creation failed")

        # content_length was added after the original schema; ensure it
        # exists on legacy databases.
        alter_sql = "ALTER TABLE news_sentiment ADD COLUMN IF NOT EXISTS content_length INTEGER"
        alter_result = self.db.execute_query(alter_sql)
        if alter_result["status"] == "error":
            logger.warning(
                f"content_length column add failed (may be fine if it already exists): {alter_result['message']}"
            )

        logger.info("news_sentiment table verified/created")

    def _create_indexes(self) -> None:
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_news_sentiment_ticker_date ON news_sentiment(ticker, published_at);",
            "CREATE INDEX IF NOT EXISTS idx_news_sentiment_published_at ON news_sentiment(published_at);",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_news_sentiment_url ON news_sentiment(article_url);",
        ]
        for index_sql in indexes:
            result = self.db.execute_query(index_sql)
            if result["status"] == "error":
                logger.warning(f"Failed to create index: {result['message']}")

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def check_article_exists(self, article_url: str) -> bool:
        query = "SELECT COUNT(*) as count FROM news_sentiment WHERE article_url = %s"
        result = self.db.execute_query(query, (article_url,), fetch=True)

        if result["status"] == "error":
            logger.error(f"Error checking article existence: {result['message']}")
            return False
        if result["data"]:
            return result["data"][0]["count"] > 0
        return False

    def get_sentiment_data(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch sentiment rows for ``ticker`` filtered by published_at."""
        query = "SELECT * FROM news_sentiment WHERE ticker = %s"
        params: List = [ticker]

        if start_date:
            query += " AND published_at >= %s"
            params.append(start_date)
        if end_date:
            query += " AND published_at <= %s"
            params.append(end_date)

        query += " ORDER BY published_at DESC"

        result = self.db.execute_query(query, params, fetch=True)

        if result["status"] == "error":
            logger.error(f"Error retrieving sentiment data: {result['message']}")
            return pd.DataFrame()
        if not result["data"]:
            return pd.DataFrame()

        df = pd.DataFrame(result["data"])
        df["published_at"] = pd.to_datetime(df["published_at"])
        df["created_at"] = pd.to_datetime(df["created_at"])
        return df

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def insert_sentiment_record(
        self,
        ticker: str,
        article_url: str,
        published_at: datetime,
        sentiment_score: float,
        content_summary: str,
    ) -> bool:
        """Insert a single record. Idempotent on ``article_url``."""
        if not -1.0 <= sentiment_score <= 1.0:
            logger.warning(
                f"Invalid sentiment score {sentiment_score} for {ticker}. Skipping."
            )
            return False

        if self.check_article_exists(article_url):
            logger.debug(f"Article already exists: {article_url}")
            return True

        if len(content_summary) > CONTENT_SUMMARY_MAX_LENGTH:
            content_summary = content_summary[:CONTENT_SUMMARY_MAX_LENGTH]

        insert_sql = """
        INSERT INTO news_sentiment (ticker, article_url, published_at, sentiment_score, content_summary)
        VALUES (%s, %s, %s, %s, %s)
        """
        try:
            result = self.db.execute_query(
                insert_sql,
                (ticker, article_url, published_at, sentiment_score, content_summary),
            )
            if result["status"] == "error":
                logger.error(
                    f"Failed to insert sentiment record for {ticker}: {result['message']}"
                )
                logger.error(
                    f"Data: URL={article_url[:100]}..., score={sentiment_score}, date={published_at}"
                )
                return False
            return True
        except Exception as exc:
            logger.error(
                f"Exception during sentiment record insertion for {ticker}: {exc}"
            )
            logger.error(
                f"Data: URL={article_url[:100]}..., score={sentiment_score}, date={published_at}"
            )
            return False

    def process_articles_with_sentiment(
        self, ticker: str, articles_df: pd.DataFrame, sentiment_df: pd.DataFrame
    ) -> Dict[str, int]:
        """Bulk-insert articles + sentiment scores. Single DB round trip."""
        stats = {"processed": 0, "inserted": 0, "skipped": 0, "errors": 0}

        if len(articles_df) != len(sentiment_df):
            logger.error(
                f"Mismatch: {len(articles_df)} articles vs {len(sentiment_df)} sentiment scores for {ticker}"
            )
            return stats

        articles_df = articles_df.reset_index(drop=True)
        sentiment_df = sentiment_df.reset_index(drop=True)

        logger.info(
            f"Processing {len(articles_df)} articles with sentiment for {ticker}"
        )

        rows = []
        for i in range(len(articles_df)):
            try:
                article_row = articles_df.iloc[i]
                sentiment_score = float(sentiment_df.iloc[i]["sentiment"])
                if not -1.0 <= sentiment_score <= 1.0:
                    stats["skipped"] += 1
                    continue

                content = str(article_row.get("content", ""))
                title = str(article_row.get("title", ""))
                full_text = (title + " " + content).strip()
                content_summary = full_text[:CONTENT_SUMMARY_MAX_LENGTH].strip()
                # Word count of the full pre-truncation text drives the
                # length-weighted daily mean downstream.
                content_length = len(full_text.split())

                rows.append(
                    {
                        "ticker": ticker,
                        "article_url": str(article_row.get("site", "")),
                        "published_at": pd.to_datetime(article_row["publishedDate"]),
                        "sentiment_score": sentiment_score,
                        "content_summary": content_summary,
                        "content_length": content_length,
                    }
                )
                stats["processed"] += 1
            except Exception as exc:
                logger.error(f"Error preparing article {i} for {ticker}: {exc}")
                stats["errors"] += 1

        if not rows:
            logger.info(f"{ticker}: no valid rows to insert")
            return stats

        result = self.db.bulk_inject_to_db(
            table="news_sentiment", data=rows, conflict_columns=["article_url"]
        )

        if result["status"] == "error":
            logger.error(f"Bulk insert failed for {ticker}: {result['message']}")
            stats["errors"] += len(rows)
        else:
            stats["inserted"] = result.get("rowcount", 0)
            logger.info(f"Completed {ticker}: bulk insert done — {result['message']}")

        return stats

    def update_market_data_sentiment(self, ticker: str) -> None:
        """Sync length-weighted, freshness-decayed daily mean into ``market_data.sentiment_score``.

        Per-article weight = ``COALESCE(content_length, %(default_len)s) *
        POWER(0.5, age_seconds / %(half_life)s)`` where ``age_seconds =
        EXTRACT(EPOCH FROM (NOW() - published_at))``. Per-day score =
        ``SUM(score * weight) / SUM(weight)``.

        Because the decay uses ``NOW()``, historical days' scores shift
        every cycle as older articles' weights drop. That is intentional
        ("freshness" semantics) but it means this UPDATE rewrites the
        full sentiment history for the ticker on each invocation. The
        ``ABS(...) > 1e-4`` guard suppresses no-op writes from float
        precision jitter.
        """
        sql = """
        UPDATE market_data md
        SET sentiment_score = ns.weighted_score
        FROM (
            SELECT
                ticker,
                DATE(published_at) AS day,
                SUM(sentiment_score * weight) / NULLIF(SUM(weight), 0) AS weighted_score
            FROM (
                SELECT
                    ticker,
                    published_at,
                    sentiment_score,
                    COALESCE(content_length, %s)
                        * POWER(0.5, EXTRACT(EPOCH FROM (NOW() - published_at)) / %s::float)
                        AS weight
                FROM news_sentiment
                WHERE ticker = %s
                  AND published_at >= NOW() - (%s || ' days')::interval
            ) weighted
            GROUP BY ticker, DATE(published_at)
        ) ns
        WHERE md.ticker = ns.ticker
          AND md.date = ns.day
          AND ns.weighted_score IS NOT NULL
          AND (
            md.sentiment_score IS NULL
            OR ABS(md.sentiment_score - ns.weighted_score) > 0.0001
          );
        """
        params = (
            DEFAULT_CONTENT_LENGTH,
            SENTIMENT_HALF_LIFE_SECONDS,
            ticker,
            SENTIMENT_UPDATE_LOOKBACK_DAYS,
        )
        max_attempts = 5
        for i in range(max_attempts):
            try:
                result = self.db.execute_query(sql, params)
                if result["status"] == "error":
                    logger.warning(
                        "Attempt %s/%s failed to update market_data sentiment for %s: %s",
                        i + 1,
                        max_attempts,
                        ticker,
                        result.get("message", "<no message>"),
                    )
                    time.sleep(2 ** i)
                    continue
                logger.info(f"Updated market_data.sentiment_score for {ticker}")
                return
            except Exception as exc:
                logger.warning(
                    "Attempt %s/%s raised while updating market_data sentiment for %s: %s",
                    i + 1,
                    max_attempts,
                    ticker,
                    exc,
                )

        logger.warning(
            "Failed to update market_data sentiment for %s after %s attempts",
            ticker,
            max_attempts,
        )

    # ------------------------------------------------------------------
    # CSV-driven flows
    # ------------------------------------------------------------------

    def update_from_csv_files(
        self,
        ticker: str,
        articles_dir: str = str(ARTICLES_DIR),
        sentiment_dir: str = str(SCORES_DIR),
    ) -> bool:
        """Load CSVs for ``ticker``, bulk-insert, then sync market_data."""
        articles_path = Path(articles_dir) / f"{ticker}.csv"
        sentiment_path = Path(sentiment_dir) / f"{ticker}_article_scores.csv"

        try:
            if not articles_path.exists():
                logger.warning(f"Articles file not found: {articles_path}")
                return False

            articles_df = pd.read_csv(articles_path, parse_dates=["publishedDate"])
            logger.info(f"Loaded {len(articles_df)} articles for {ticker}")
            articles_df = articles_df.drop_duplicates(
                subset=["publishedDate", "title"], keep="first"
            )

            if not sentiment_path.exists():
                logger.warning(f"Sentiment file not found: {sentiment_path}")
                return False

            sentiment_df = pd.read_csv(sentiment_path, parse_dates=["date"])
            logger.info(f"Loaded {len(sentiment_df)} sentiment scores for {ticker}")

            stats = self.process_articles_with_sentiment(
                ticker, articles_df, sentiment_df
            )
            self.update_market_data_sentiment(ticker)
            return stats["inserted"] > 0 or stats["skipped"] > 0
        except Exception as exc:
            logger.error(f"Error updating database for {ticker}: {exc}")
            return False

    def update_multiple_tickers(
        self,
        tickers: List[str],
        articles_dir: str = str(ARTICLES_DIR),
        sentiment_dir: str = str(SCORES_DIR),
    ) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        total_stats = {
            "total_tickers": len(tickers),
            "successful_tickers": 0,
            "failed_tickers": 0,
        }

        for ticker in tickers:
            logger.info(f"Updating database for ticker: {ticker}")
            success = self.update_from_csv_files(ticker, articles_dir, sentiment_dir)
            results[ticker] = success
            if success:
                total_stats["successful_tickers"] += 1
            else:
                total_stats["failed_tickers"] += 1
            logger.info(
                "Ticker %s update %s",
                ticker,
                "succeeded" if success else "failed",
            )

        successful = total_stats["successful_tickers"]
        total = len(results)
        logger.info(
            f"Database update complete: {successful}/{total} tickers successful"
        )
        logger.info(
            "Run stats: total=%s, successful=%s, failed=%s",
            total_stats["total_tickers"],
            total_stats["successful_tickers"],
            total_stats["failed_tickers"],
        )
        return results
