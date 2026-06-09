"""Shared market-hours helper.

US equities cash session, America/New_York timezone. No holiday calendar
(weekend filter only); callers that need holiday-accurate gating should
plug in ``pandas_market_calendars`` or a similar dependency.
"""

from __future__ import annotations

from datetime import datetime, time as dtime

import pytz

MARKET_TIMEZONE = pytz.timezone("America/New_York")
MARKET_OPEN_TIME: dtime = dtime(9, 30)
MARKET_CLOSE_TIME: dtime = dtime(16, 0)


def is_market_open(now: datetime | None = None) -> bool:
    """Return True when US equities are in the regular cash session.

    Treats Sat/Sun as closed. ``now`` may be naive (assumed America/New_York)
    or tz-aware; tz-aware values are converted to America/New_York before
    the time-of-day check.
    """
    if now is None:
        current = datetime.now(MARKET_TIMEZONE)
    elif now.tzinfo is None:
        current = MARKET_TIMEZONE.localize(now)
    else:
        current = now.astimezone(MARKET_TIMEZONE)

    if current.weekday() >= 5:
        return False

    return MARKET_OPEN_TIME <= current.time() <= MARKET_CLOSE_TIME


def seconds_until_market_close(now: datetime | None = None) -> int:
    """Return seconds until 16:00 ET today. 0 if already past close."""
    if now is None:
        current = datetime.now(MARKET_TIMEZONE)
    elif now.tzinfo is None:
        current = MARKET_TIMEZONE.localize(now)
    else:
        current = now.astimezone(MARKET_TIMEZONE)

    close_today = current.replace(
        hour=MARKET_CLOSE_TIME.hour,
        minute=MARKET_CLOSE_TIME.minute,
        second=0,
        microsecond=0,
    )
    delta = (close_today - current).total_seconds()
    return max(0, int(delta))


__all__ = [
    "MARKET_CLOSE_TIME",
    "MARKET_OPEN_TIME",
    "MARKET_TIMEZONE",
    "is_market_open",
    "seconds_until_market_close",
]
