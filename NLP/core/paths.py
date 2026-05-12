"""Filesystem paths used across the NLP package."""

from __future__ import annotations

import os
import sys
from pathlib import Path

NLP_DIR: Path = Path(__file__).resolve().parent.parent
PROJECT_ROOT: Path = NLP_DIR.parent

ARTICLES_DIR: Path = NLP_DIR / "articles"
SCORES_DIR: Path = NLP_DIR / "sentiment_scores"
STATE_DIR: Path = NLP_DIR / "fetch_state"
MODEL_DIR: Path = NLP_DIR / "finbert-combined-final"
DAEMON_LOG_FILE: Path = NLP_DIR / "daemon.log"


def ensure_project_root_on_path() -> None:
    """Make sure the repo root is importable.

    Replaces the sys.path hack that was duplicated at the top of every
    legacy NLP script.
    """
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


# Run on import so legacy scripts and CLI entrypoints keep working.
ensure_project_root_on_path()


def ensure_dir(path: os.PathLike | str) -> Path:
    """Create ``path`` (and parents) if missing and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
