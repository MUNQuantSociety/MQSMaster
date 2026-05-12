"""Load the active ticker universe from portfolio configs and batch it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from NLP.core import PROJECT_ROOT, get_logger

logger = get_logger(__name__)

DEFAULT_NUM_PORTFOLIOS: int = 4
DEFAULT_EXCLUDED_TICKERS: frozenset = frozenset({"^VIX"})


class TickerUniverse:
    """Resolve and batch the active set of tickers.

    The legacy daemon hard-coded ``portfolio_1..3`` config paths and a
    ``^VIX`` skip list at module scope. This class makes both injectable
    so the daemon, tests, and ad-hoc scripts share one source of truth.
    """

    def __init__(
        self,
        config_paths: Optional[Sequence[Path | str]] = None,
        excluded_tickers: Iterable[str] = DEFAULT_EXCLUDED_TICKERS,
    ):
        if config_paths is None:
            config_paths = self._default_config_paths()
        self.config_paths: List[Path] = [Path(p) for p in config_paths]
        self.excluded_tickers = frozenset(excluded_tickers)

    @staticmethod
    def _default_config_paths(num_portfolios: int = DEFAULT_NUM_PORTFOLIOS) -> List[Path]:
        return [
            PROJECT_ROOT / "src" / "portfolios" / f"portfolio_{n}" / "config.json"
            for n in range(1, num_portfolios)
        ]

    def load_tickers(self) -> List[str]:
        """Return deduplicated tickers in first-seen order."""
        seen: set[str] = set()
        tickers: List[str] = []

        for path in self.config_paths:
            try:
                with open(path) as f:
                    config = json.load(f)
            except Exception as exc:
                logger.warning(f"Could not load portfolio config {path}: {exc}")
                continue

            for ticker in config.get("TICKERS", []):
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
