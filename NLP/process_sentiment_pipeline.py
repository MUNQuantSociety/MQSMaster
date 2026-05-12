"""Backwards-compatible module for ``python -m NLP.process_sentiment_pipeline``.

Implementation lives in :mod:`NLP.sentiment.pipeline`. This shim only
re-exports the class and the CLI entrypoint that the daemon's
``subprocess.run`` call relies on.
"""

from __future__ import annotations

import sys

from NLP.core import ARTICLES_DIR, SCORES_DIR, get_logger
from NLP.persistence.repository import NewsSentimentRepository
from NLP.sentiment.pipeline import SentimentPipeline
from NLP.sentiment.scorer import FinBertSentimentScorer

# Legacy alias preserved for callers that imported it from this module.
SentimentDatabaseUpdater = NewsSentimentRepository

logger = get_logger(__name__)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run complete sentiment processing pipeline"
    )
    parser.add_argument("tickers", nargs="+", help="Ticker symbols to process")
    parser.add_argument(
        "--articles-dir",
        default=str(ARTICLES_DIR),
        help="Directory containing article CSV files",
    )
    parser.add_argument(
        "--sentiment-dir",
        default=str(SCORES_DIR),
        help="Directory to save sentiment CSV files",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "Directory containing the FinBERT model or HuggingFace model name. "
            "If not provided, uses local safetensors model if available, else HuggingFace."
        ),
    )
    parser.add_argument(
        "--chunk-size", type=int, default=FinBertSentimentScorer.DEFAULT_CHUNK_SIZE,
        help="Batch size for processing articles",
    )

    args = parser.parse_args()

    pipeline = SentimentPipeline(model_dir=args.model_dir, chunk_size=args.chunk_size)
    results = pipeline.process_multiple_tickers_complete(
        tickers=args.tickers,
        articles_dir=args.articles_dir,
        sentiment_dir=args.sentiment_dir,
    )

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "FinBertSentimentScorer",
    "NewsSentimentRepository",
    "SentimentDatabaseUpdater",
    "SentimentPipeline",
    "main",
]
