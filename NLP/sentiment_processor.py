"""Backwards-compatible module exposing :class:`SentimentProcessor`.

The implementation lives in :mod:`NLP.sentiment.scorer`
(``FinBertSentimentScorer``). The :class:`SentimentProcessor` subclass
defined here exists so legacy tests that monkey-patch
``NLP.sentiment_processor.AutoTokenizer`` and
``NLP.sentiment_processor.AutoModelForSequenceClassification`` keep
working - bare-name lookups inside the overridden ``load_model`` resolve
through this module's globals.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from NLP.core import ARTICLES_DIR, MODEL_DIR, SCORES_DIR, get_logger
from NLP.sentiment.scorer import (
    DEFAULT_LOCAL_MODEL_DIR,
    HUGGINGFACE_FALLBACK,
    FinBertSentimentScorer,
)

logger = get_logger(__name__)

# Re-export the legacy module-level constant the original file exposed.
DEFAULT_MODEL_DIR = str(MODEL_DIR)


class SentimentProcessor(FinBertSentimentScorer):
    """Legacy alias for :class:`FinBertSentimentScorer`.

    Overrides :meth:`load_model` so the bare-name lookups land in this
    module's globals - the test suite monkey-patches
    ``AutoTokenizer`` and ``AutoModelForSequenceClassification`` here.
    """

    def load_model(self) -> None:
        try:
            logger.info(f"Loading model from {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(
                    self.model_dir,
                    trust_remote_code=False,
                    torch_dtype=(
                        torch.float16 if self.device.type == "cuda" else torch.float32
                    ),
                )
                .to(self.device)
                .eval()
            )
            logger.info("Model loaded successfully")
        except Exception as exc:
            logger.error(f"Failed to load model from {self.model_dir}: {exc}")
            raise


def main() -> None:
    """Legacy CLI: ``python -m NLP.sentiment_processor TICKER [...]``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process articles for sentiment analysis"
    )
    parser.add_argument("tickers", nargs="+", help="Ticker symbols to process")
    parser.add_argument(
        "--articles-dir",
        default=str(ARTICLES_DIR),
        help="Directory containing article CSV files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCORES_DIR),
        help="Directory to save sentiment score CSV files",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_LOCAL_MODEL_DIR,
        help="Directory containing the fine-tuned FinBERT model. Missing dir = hard failure (no HuggingFace fallback).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=32, help="Batch size for processing articles"
    )

    args = parser.parse_args()

    processor: Optional[SentimentProcessor] = SentimentProcessor(
        model_dir=args.model_dir, chunk_size=args.chunk_size
    )

    results = processor.process_multiple_tickers(
        tickers=args.tickers,
        articles_dir=args.articles_dir,
        output_dir=args.output_dir,
    )

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "AutoModelForSequenceClassification",
    "AutoTokenizer",
    "DEFAULT_LOCAL_MODEL_DIR",
    "DEFAULT_MODEL_DIR",
    "FinBertSentimentScorer",
    "HUGGINGFACE_FALLBACK",
    "SentimentProcessor",
    "main",
]
