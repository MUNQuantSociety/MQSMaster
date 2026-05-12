"""CLI entry point for running the RBP pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RBP.config import RBPConfig
from RBP.pipeline import RBPPipeline


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = RBPConfig()
    pipeline = RBPPipeline(config)
    predictions_df, rbi_df = pipeline.run()

    print("\n=== Predictions (head) ===")
    print(predictions_df.head())
    print("\n=== RBI scores (head) ===")
    print(rbi_df.head())

    predictions_df.to_csv("rbp_predictions.csv")
    rbi_df.to_csv("rbp_rbi_scores.csv")
    print("\nSaved: rbp_predictions.csv, rbp_rbi_scores.csv")


if __name__ == "__main__":
    main()
