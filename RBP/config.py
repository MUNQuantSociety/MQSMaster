"""Default configuration for the RBP pipeline.

Tweak ticker universe, lookback window, target column, and grid sizing here
without touching pipeline code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from RBP.features import DEFAULT_FEATURES, TARGET_COLUMN
from RBP.models import DEFAULT_CENSORING_QUANTILES


DEFAULT_TICKERS: List[str] = [
    "AAPL",
    "TSLA",
    "AMD",
    "MSFT",
    "NVDA",
]


@dataclass
class RBPConfig:
    """All knobs needed to run an end-to-end RBP experiment."""

    tickers: List[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    feature_columns: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    target_column: str = TARGET_COLUMN
    lookback_days: int = 365 * 5
    train_test_split_date: str = "2023-01-01"
    censoring_quantiles: List[float] = field(
        default_factory=lambda: list(DEFAULT_CENSORING_QUANTILES)
    )
    max_combination_size: Optional[int] = 1
    max_test_tasks: Optional[int] = 200
    n_jobs: int = -1
