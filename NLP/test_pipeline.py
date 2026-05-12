"""Smoke-test entrypoint for the sentiment pipeline.

Kept at this path for backwards compatibility - the legacy NLP README
documented ``python -m NLP.test_pipeline``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from NLP.core import ARTICLES_DIR, SCORES_DIR, get_logger
from NLP.sentiment.pipeline import SentimentPipeline

logger = get_logger(__name__)


def test_pipeline(test_ticker: str = "AAPL") -> bool:
    """Run the full pipeline for one ticker against on-disk fixtures."""
    articles_dir = str(ARTICLES_DIR)
    sentiment_dir = str(SCORES_DIR)

    logger.info("Starting pipeline test")

    articles_path = Path(articles_dir) / f"{test_ticker}.csv"
    if not articles_path.exists():
        logger.error(f"Test articles file not found: {articles_path}")
        logger.info("Please run the article scraper first to generate test data")
        return False

    try:
        pipeline = SentimentPipeline()
        logger.info(f"Testing pipeline with {test_ticker}")
        success = pipeline.process_ticker_complete(
            ticker=test_ticker,
            articles_dir=articles_dir,
            sentiment_dir=sentiment_dir,
        )

        if success:
            logger.info("Pipeline test PASSED")
            sentiment_path = Path(sentiment_dir) / f"{test_ticker}_article_scores.csv"
            daily_path = Path(sentiment_dir) / f"{test_ticker}_daily_scores.csv"

            if sentiment_path.exists() and daily_path.exists():
                logger.info("Sentiment CSV files created successfully")
                db_data = pipeline.db_updater.get_sentiment_data(test_ticker)
                if not db_data.empty:
                    logger.info(
                        f"Database contains {len(db_data)} sentiment records for {test_ticker}"
                    )
                    logger.info("Database integration test PASSED")
                else:
                    logger.warning("No data found in database")
            return True

        logger.error("Pipeline test FAILED")
        return False

    except Exception as exc:
        logger.error(f"Pipeline test error: {exc}")
        return False


def main() -> None:
    success = test_pipeline()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


__all__ = ["main", "test_pipeline"]
