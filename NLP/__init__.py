"""NLP package.

Public subpackages:
    core            - shared helpers (paths, timestamps, logging)
    scrapers        - per-source article scrapers + aggregator
    sentiment       - FinBERT scorer and end-to-end pipeline
    persistence     - news_sentiment table repository
    orchestration   - ticker universe loader + batch builder

Backward-compatible top-level modules (`NLP.fetch_articles`,
`NLP.fetch_alt_articles`, `NLP.sentiment_processor`,
`NLP.process_sentiment_pipeline`, `NLP.update_database`,
`NLP.test_pipeline`) re-export the public surface from the subpackages.
"""

__all__ = [
    "core",
    "scrapers",
    "sentiment",
    "persistence",
    "orchestration",
]
