"""Backwards-compatible module for ``python -m NLP.update_database``.

The repository implementation lives in :mod:`NLP.persistence.repository`
as :class:`NewsSentimentRepository`. ``SentimentDatabaseUpdater`` is
preserved here as an alias so existing imports keep working.
"""

from __future__ import annotations

import sys

import pandas as pd

from NLP.core import ARTICLES_DIR, SCORES_DIR, get_logger
from NLP.persistence.repository import (
    CONTENT_SUMMARY_MAX_LENGTH,
    NewsSentimentRepository,
)

# Legacy class name kept as an alias for back-compat.
SentimentDatabaseUpdater = NewsSentimentRepository

logger = get_logger(__name__)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Update news_sentiment database table")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols to update")
    parser.add_argument(
        "--articles-dir",
        default=str(ARTICLES_DIR),
        help="Directory containing article CSV files",
    )
    parser.add_argument(
        "--sentiment-dir",
        default=str(SCORES_DIR),
        help="Directory containing sentiment CSV files",
    )
    parser.add_argument(
        "--query",
        action="store_true",
        help="Query and display sentiment data instead of updating",
    )

    args = parser.parse_args()

    updater = SentimentDatabaseUpdater()

    if args.query:
        for ticker in args.tickers:
            result = updater.db.execute_query(
                """
                SELECT
                    DATE(published_at) AS date,
                    ticker,
                    ROUND(AVG(sentiment_score)::numeric, 4) AS avg_score,
                    COUNT(*) AS articles
                FROM news_sentiment
                WHERE ticker = %s
                GROUP BY DATE(published_at), ticker
                ORDER BY date DESC
                """,
                (ticker,),
                fetch=True,
            )
            if result["status"] == "success" and result["data"]:
                df = pd.DataFrame(result["data"])
                print(f"\n{ticker} daily sentiment:")
                print(df.to_string(index=False))
            else:
                print(f"\n{ticker} - no data found")
    else:
        results = updater.update_multiple_tickers(
            tickers=args.tickers,
            articles_dir=args.articles_dir,
            sentiment_dir=args.sentiment_dir,
        )
        if not all(results.values()):
            sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "CONTENT_SUMMARY_MAX_LENGTH",
    "NewsSentimentRepository",
    "SentimentDatabaseUpdater",
    "main",
]
