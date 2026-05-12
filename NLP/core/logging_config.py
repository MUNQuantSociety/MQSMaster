"""Single source of truth for NLP logging setup."""

from __future__ import annotations

import logging
import sys
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_BASIC_CONFIGURED = False


def _ensure_basic_config(level: int = logging.INFO) -> None:
    global _BASIC_CONFIGURED
    if _BASIC_CONFIGURED:
        return
    logging.basicConfig(level=level, format=_DEFAULT_FORMAT)
    _BASIC_CONFIGURED = True


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> Logger:
    """Return a logger with a consistent format. Idempotent."""
    _ensure_basic_config(level=level)
    return logging.getLogger(name)


def configure_rotating_file_logger(
    log_file: Path,
    *,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 1,
    level: int = logging.INFO,
) -> Logger:
    """Configure root logger with both stdout and a rotating file sink.

    Used by the long-running daemon. Replacing handlers is intentional so
    repeated calls remain idempotent.
    """
    handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    root = logging.getLogger()
    root.handlers = [handler, logging.StreamHandler(sys.stdout)]
    root.setLevel(level)
    return root
