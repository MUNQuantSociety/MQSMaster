"""Timestamp normalization helpers shared by every scraper.

Two flavours exist for historical reasons:

* :func:`normalize_timestamp` - scalar input -> tz-naive ``pd.Timestamp``.
* :func:`normalize_published_date_column` - DataFrame with a
  ``publishedDate`` column. Coerces to datetime and drops rows that
  could not be parsed.

:func:`normalize_published_date` is kept as an alias for the column
helper so legacy ``NLP.fetch_articles`` callers keep working.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def normalize_timestamp(value: Any) -> Optional[pd.Timestamp]:
    """Convert assorted date-like inputs to a tz-naive ``pd.Timestamp``.

    Handles ``arrow.Arrow``-style objects (which carry a ``.datetime``
    attribute), strings, ``datetime``/``date``, and pandas timestamps.
    Returns ``pd.NaT`` when parsing fails so callers can drop bad rows.
    """
    if value is None:
        return pd.NaT

    if hasattr(value, "datetime"):
        value = value.datetime

    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert(None)


def normalize_published_date_column(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Coerce the ``publishedDate`` column to datetime and drop NaT rows."""
    if df is None or df.empty or "publishedDate" not in df.columns:
        return df

    out = df.copy()
    out["publishedDate"] = pd.to_datetime(out["publishedDate"], errors="coerce")
    return out.dropna(subset=["publishedDate"])


# Legacy alias from fetch_articles.py.
normalize_published_date = normalize_published_date_column
