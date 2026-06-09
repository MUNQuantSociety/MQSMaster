"""
End-sweep: 12 tickers with length >10 chars that were skipped by the main
2026 backfill (varchar(10) guard). Column has since been widened to
VARCHAR(20), and the guard has been removed from concurrent_backfill.py.

Range 2026-01-01 -> 2026-05-21, interval=1, on_conflict=ignore, threads=4.

Run:
    ./mqs/bin/python -u scripts/backfill_runs/run_backfill_long_tickers.py \
        > scripts/backfill_runs/backfill_long_tickers.log 2>&1
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.orchestrator.backfill.backfill_cli import _build_exchange_map  # noqa: E402
from src.orchestrator.backfill.concurrent_backfill import concurrent_backfill  # noqa: E402

LONG_TICKERS = [
    "AGENTFUNUSD",
    "BABYDOGEUSD",
    "BANANAS31USD",
    "CONSCIOUSUSD",
    "FARTCOINUSD",
    "FIGRHELOCUSD",
    "GOMININGUSD",
    "JELLYJELLYUSD",
    "PC0000023USD",
    "PC0000031USD",
    "PIEVERSEUSD",
    "WHITEWHALEUSD",
]
START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 5, 21)
INTERVAL = 1
THREADS = 4
ON_CONFLICT = "ignore"
PROGRESS_PATH = os.path.join(
    os.path.dirname(__file__), "backfill_long_tickers_progress.jsonl"
)


def _write_progress(event: dict) -> None:
    event["ts"] = datetime.utcnow().isoformat() + "Z"
    with open(PROGRESS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("backfill_long_tickers")
    log.info("Tickers: %s", LONG_TICKERS)
    exchange_map = _build_exchange_map(LONG_TICKERS)
    log.info("Exchange map: %s", exchange_map)
    _write_progress({"event": "start", "tickers": LONG_TICKERS})
    started = datetime.utcnow()
    try:
        concurrent_backfill(
            tickers=LONG_TICKERS,
            start_date=START_DATE,
            end_date=END_DATE,
            interval=INTERVAL,
            exchange="crypto",
            exchange_map=exchange_map,
            dry_run=False,
            on_conflict=ON_CONFLICT,
            threads=THREADS,
        )
        elapsed = datetime.utcnow() - started
        log.info("Long-ticker sweep complete in %s", elapsed)
        _write_progress({"event": "complete", "elapsed_seconds": elapsed.total_seconds()})
        return 0
    except Exception as exc:  # noqa: BLE001
        log.exception("Sweep failed: %s", exc)
        _write_progress({"event": "fatal", "error": repr(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
