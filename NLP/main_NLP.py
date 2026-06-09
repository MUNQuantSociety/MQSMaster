#!/usr/bin/env python3
"""main_NLP.py
Entry point for the NLP sentiment pipeline.

main_NLP.py is the NLP counterpart of ``src/main.py``: it discovers the
ticker universe, fetches articles, scores sentiment with FinBERT, and
updates the ``news_sentiment`` table. All orchestration lives in
:class:`NLP.runner.NLPRunner`; this file is the thin entrypoint.
"""

from __future__ import annotations

import logging
import os
import sys

# When invoked as a script (``python NLP/main_NLP.py``) Python does not
# put the repo root on sys.path, so ``import NLP`` fails. Bootstrap the
# path before any NLP imports.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from NLP.runner import NLPRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main() -> None:
    """Run the NLP pipeline.

    Loads the ticker universe from
    ``src/orchestrator/backfill/tickers.json``, then runs the
    fetch → score → database-update loop continuously.
    """
    runner = NLPRunner()
    try:
        runner.run()
    except Exception:
        logging.critical("A critical error occurred in main_NLP.", exc_info=True)
    finally:
        logging.info("NLP pipeline is shutting down.")


if __name__ == "__main__":
    main()
