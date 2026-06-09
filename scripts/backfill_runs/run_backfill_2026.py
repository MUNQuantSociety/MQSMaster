"""
Driver script: concurrent backfill of all tickers in tickers.json
from 2026-01-01 to 2026-05-21, interval=1, on_conflict=ignore, threads=6.

Run:
    ./mqs/bin/python -u scripts/backfill_runs/run_backfill_2026.py \
        > scripts/backfill_runs/backfill_2026.log 2>&1
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

START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 5, 21)
INTERVAL = 1
THREADS = 6
ON_CONFLICT = "ignore"
TICKERS_PATH = os.path.join(
    REPO_ROOT, "src", "orchestrator", "backfill", "tickers.json"
)
PROGRESS_PATH = os.path.join(
    os.path.dirname(__file__), "backfill_2026_progress.jsonl"
)


def _load_tickers() -> list[str]:
    with open(TICKERS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [str(t).strip().upper() for t in data if str(t).strip()]


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
    log = logging.getLogger("backfill_2026")

    tickers = _load_tickers()
    log.info("Loaded %d tickers from %s", len(tickers), TICKERS_PATH)
    log.info(
        "Range %s -> %s, interval=%dm, threads=%d, on_conflict=%s",
        START_DATE,
        END_DATE,
        INTERVAL,
        THREADS,
        ON_CONFLICT,
    )

    exchange_map = _build_exchange_map(tickers)
    summary = {k: 0 for k in ("crypto", "commodity", "sp500", "nasdaq")}
    for exch in exchange_map.values():
        summary[exch] = summary.get(exch, 0) + 1
    log.info("Exchange mix: %s", summary)

    _write_progress(
        {
            "event": "start",
            "tickers": len(tickers),
            "start_date": START_DATE.isoformat(),
            "end_date": END_DATE.isoformat(),
            "threads": THREADS,
            "interval": INTERVAL,
            "exchange_mix": summary,
        }
    )

    started = datetime.utcnow()
    try:
        concurrent_backfill(
            tickers=tickers,
            start_date=START_DATE,
            end_date=END_DATE,
            interval=INTERVAL,
            exchange="nasdaq",
            exchange_map=exchange_map,
            dry_run=False,
            on_conflict=ON_CONFLICT,
            threads=THREADS,
        )
        elapsed = datetime.utcnow() - started
        log.info("Backfill complete in %s", elapsed)
        _write_progress(
            {
                "event": "complete",
                "elapsed_seconds": elapsed.total_seconds(),
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        log.exception("Backfill failed: %s", exc)
        _write_progress({"event": "fatal", "error": repr(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
