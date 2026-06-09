"""Load the active ticker universe from ``tickers.json`` and batch it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from NLP.core import PROJECT_ROOT, get_logger

logger = get_logger(__name__)

DEFAULT_TICKERS_PATH: Path = (
    PROJECT_ROOT / "src" / "orchestrator" / "backfill" / "tickers.json"
)
DEFAULT_EXCLUDED_TICKERS: frozenset = frozenset({"^VIX"})


class TickerUniverse:
    """Resolve and batch the active set of tickers.

    Source of truth is ``src/orchestrator/backfill/tickers.json`` so the
    NLP pipeline scrapes the same universe the backfill pipeline ingests.
    """

    def __init__(
        self,
        tickers_path: Optional[Path | str] = None,
        excluded_tickers: Iterable[str] = DEFAULT_EXCLUDED_TICKERS,
    ):
        self.tickers_path = (
            Path(tickers_path) if tickers_path is not None else DEFAULT_TICKERS_PATH
        )
        self.excluded_tickers = frozenset(excluded_tickers)

    def load_tickers(self) -> List[str]:
        """Return deduplicated tickers in file order, minus the excluded set."""
        try:
            with open(self.tickers_path) as f:
                raw = json.load(f)
        except Exception as exc:
            logger.warning(
                f"Could not load tickers from {self.tickers_path}: {exc}"
            )
            return []

        if not isinstance(raw, list):
            logger.warning(
                f"tickers.json at {self.tickers_path} is not a JSON list; got {type(raw).__name__}"
            )
            return []

        seen: set[str] = set()
        tickers: List[str] = []
        for ticker in raw:
            if not isinstance(ticker, str):
                continue
            if ticker in seen or ticker in self.excluded_tickers:
                continue
            seen.add(ticker)
            tickers.append(ticker)
        return tickers

    @staticmethod
    def build_batches(tickers: Sequence[str], num_batches: int = 4) -> List[List[str]]:
        """Split ``tickers`` into ``num_batches`` roughly-equal batches."""
        if num_batches <= 0:
            raise ValueError("num_batches must be > 0")

        n = len(tickers)
        if n == 0:
            return []

        size = max(1, (n + num_batches - 1) // num_batches)
        return [list(tickers[i : i + size]) for i in range(0, n, size)]

    def load_batches(self, num_batches: int = 4) -> List[List[str]]:
        """Convenience: load tickers and split them into ``num_batches``."""
        return self.build_batches(self.load_tickers(), num_batches=num_batches)
