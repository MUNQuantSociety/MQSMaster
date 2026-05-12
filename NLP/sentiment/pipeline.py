"""End-to-end sentiment pipeline: scoring + database persistence."""

from __future__ import annotations

from typing import Dict, List, Optional

from NLP.core import ARTICLES_DIR, SCORES_DIR, get_logger
from NLP.persistence.repository import NewsSentimentRepository
from NLP.sentiment.scorer import FinBertSentimentScorer

logger = get_logger(__name__)


class SentimentPipeline:
    """Drive scoring + database updates for one or more tickers.

    Composed of a :class:`FinBertSentimentScorer` and
    :class:`NewsSentimentRepository`. Defaults pick the local
    safetensors model when available.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        chunk_size: int = FinBertSentimentScorer.DEFAULT_CHUNK_SIZE,
        scorer: Optional[FinBertSentimentScorer] = None,
        repository: Optional[NewsSentimentRepository] = None,
    ):
        self.processor = scorer or FinBertSentimentScorer(
            model_dir=model_dir, chunk_size=chunk_size
        )
        self.db_updater = repository or NewsSentimentRepository()
        logger.info("Initialized SentimentPipeline")

    def process_ticker_complete(
        self,
        ticker: str,
        articles_dir: str = str(ARTICLES_DIR),
        sentiment_dir: str = str(SCORES_DIR),
    ) -> bool:
        """Score, then push results to the news_sentiment table for one ticker."""
        logger.info(f"Starting complete processing pipeline for {ticker}")

        try:
            logger.info(f"Step 1: Processing sentiment for {ticker}")
            sentiment_success = self.processor.process_ticker(
                ticker=ticker, articles_dir=articles_dir, output_dir=sentiment_dir
            )
            if not sentiment_success:
                logger.error(f"Sentiment processing failed for {ticker}")
                return False

            logger.info(f"Step 2: Updating database for {ticker}")
            db_success = self.db_updater.update_from_csv_files(
                ticker=ticker, articles_dir=articles_dir, sentiment_dir=sentiment_dir
            )
            if not db_success:
                logger.error(f"Database update failed for {ticker}")
                return False

            logger.info(f"Complete processing pipeline successful for {ticker}")
            return True

        except Exception as exc:
            logger.error(f"Error in complete processing pipeline for {ticker}: {exc}")
            return False

    def process_multiple_tickers_complete(
        self,
        tickers: List[str],
        articles_dir: str = str(ARTICLES_DIR),
        sentiment_dir: str = str(SCORES_DIR),
    ) -> Dict[str, bool]:
        """Run :meth:`process_ticker_complete` for each ticker."""
        results: Dict[str, bool] = {}
        for ticker in tickers:
            results[ticker] = self.process_ticker_complete(
                ticker=ticker,
                articles_dir=articles_dir,
                sentiment_dir=sentiment_dir,
            )

        successful = sum(1 for ok in results.values() if ok)
        total = len(results)
        logger.info(
            f"Pipeline processing complete: {successful}/{total} tickers successful"
        )
        return results
