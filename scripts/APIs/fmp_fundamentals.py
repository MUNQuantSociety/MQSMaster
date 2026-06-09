"""
Fetch latest annual fundamentals (ROE, gross_profit, total_assets) per ticker
from FMP and write them to a local CSV. No database changes.

Defaults:
  - tickers: read from src/portfolios/portfolio_6/universe.json
  - output: <repo_root>/fundamentals/fundamentals.csv

Usage:
  python -m scripts.APIs.fmp_fundamentals
  python -m scripts.APIs.fmp_fundamentals --tickers AAPL MSFT GOOG
  python -m scripts.APIs.fmp_fundamentals --output some/other/path.csv
  python -m scripts.APIs.fmp_fundamentals --limit 10
  python -m scripts.APIs.fmp_fundamentals --progress-every 5
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_APIS_DIR = REPO_ROOT / "scripts" / "APIs"
if str(SCRIPTS_APIS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_APIS_DIR))

from _fmp_helpers import FMPClient  # noqa: E402


DEFAULT_UNIVERSE_PATH = REPO_ROOT / "src" / "portfolios" / "portfolio_6" / "universe.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "fundamentals" / "fundamentals.csv"
FIELDNAMES = [
    "ticker",
    "roe",
    "gross_profit",
    "total_assets",
    "income_date",
    "balance_date",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("fmp_fundamentals")


def _coerce_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fetch_first(client: FMPClient, url: str, params: dict, label: str) -> Optional[dict]:
    data = client.get(url, params, label=label)
    if not data or not isinstance(data, list) or not data:
        return None
    head = data[0]
    return head if isinstance(head, dict) else None


def fetch_ttm_ratios(client: FMPClient, ticker: str) -> Optional[dict]:
    return _fetch_first(
        client,
        "https://financialmodelingprep.com/stable/ratios-ttm",
        {"symbol": ticker},
        label=f"ratios-ttm:{ticker}",
    )


def fetch_income_annual(client: FMPClient, ticker: str) -> Optional[dict]:
    return _fetch_first(
        client,
        "https://financialmodelingprep.com/stable/income-statement",
        {"symbol": ticker, "period": "annual", "limit": 1},
        label=f"income:{ticker}",
    )


def fetch_balance_annual(client: FMPClient, ticker: str) -> Optional[dict]:
    return _fetch_first(
        client,
        "https://financialmodelingprep.com/stable/balance-sheet-statement",
        {"symbol": ticker, "period": "annual", "limit": 1},
        label=f"balance:{ticker}",
    )


def build_row(client: FMPClient, ticker: str) -> dict:
    ratios = fetch_ttm_ratios(client, ticker) or {}
    income = fetch_income_annual(client, ticker) or {}
    balance = fetch_balance_annual(client, ticker) or {}

    # FMP /stable/ratios-ttm has no direct ROE. Try the field anyway (older v3
    # used to populate it), then fall back to netIncome / totalStockholdersEquity
    # using the latest annual statements.
    roe = ratios.get("returnOnEquityTTM") or ratios.get("returnOnEquity")
    if roe is None:
        net_income = _coerce_float(income.get("netIncome"))
        equity = _coerce_float(balance.get("totalStockholdersEquity"))
        if equity is None:
            equity = _coerce_float(balance.get("totalEquity"))
        if net_income is not None and equity not in (None, 0.0):
            roe = net_income / equity

    return {
        "ticker": ticker,
        "roe": _coerce_float(roe),
        "gross_profit": _coerce_float(income.get("grossProfit")),
        "total_assets": _coerce_float(balance.get("totalAssets")),
        "income_date": income.get("date"),
        "balance_date": balance.get("date"),
    }


def load_universe(path: Path) -> List[str]:
    logger.info("STEP: loading universe from %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    out = [t for t in data if isinstance(t, str) and t.strip()]
    logger.info("STEP: universe loaded -> %s tickers", len(out))
    return out


def write_csv(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("STEP: writing %s rows to %s", len(rows), output_path)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("STEP: CSV write OK")


def _log_progress(i: int, total: int, succ: int, fail: int, started_at: float, calls: int):
    elapsed = max(time.time() - started_at, 1e-6)
    rate = i / elapsed
    eta = (total - i) / rate if rate > 0 else float("inf")
    logger.info(
        "PROGRESS %s/%s | succ=%s fail=%s | elapsed=%.1fs | rate=%.2f tkr/s | eta=%.1fs | api_calls=%s",
        i,
        total,
        succ,
        fail,
        elapsed,
        rate,
        eta,
        calls,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch FMP fundamentals into a local CSV.")
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Explicit ticker list (defaults to portfolio_6 universe.json).",
    )
    parser.add_argument(
        "--universe-path",
        default=str(DEFAULT_UNIVERSE_PATH),
        help="Path to universe.json if --tickers is not provided.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Destination CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap how many tickers to process (useful for smoke tests).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Log a progress line every N tickers (default 10).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    started_at = time.time()
    logger.info("=== fmp_fundamentals START ===")
    logger.info(
        "ARGS tickers=%s universe_path=%s output=%s limit=%s progress_every=%s",
        "explicit-list" if args.tickers else "from-file",
        args.universe_path,
        args.output,
        args.limit,
        args.progress_every,
    )

    if args.tickers:
        tickers = [t.strip() for t in args.tickers if t.strip()]
        logger.info("STEP: explicit ticker list -> %s tickers", len(tickers))
    else:
        tickers = load_universe(Path(args.universe_path))

    if args.limit is not None and args.limit > 0:
        original = len(tickers)
        tickers = tickers[: args.limit]
        logger.info("STEP: applied --limit -> %s of %s tickers", len(tickers), original)

    if not tickers:
        logger.error("STEP: no tickers to fetch; aborting.")
        return 1

    try:
        client = FMPClient(logger=logger)
    except ValueError as e:
        logger.error("Cannot init FMP client: %s", e)
        return 1

    rows: List[dict] = []
    succ = 0
    fail = 0
    progress_every = max(1, args.progress_every)

    for i, ticker in enumerate(tickers, start=1):
        logger.info("TICKER %s/%s -> %s", i, len(tickers), ticker)
        try:
            row = build_row(client, ticker)
            rows.append(row)
            populated = any(
                row[k] is not None for k in ("roe", "gross_profit", "total_assets")
            )
            if populated:
                succ += 1
                logger.info(
                    "TICKER %s/%s %s OK roe=%s gross_profit=%s total_assets=%s",
                    i,
                    len(tickers),
                    ticker,
                    row["roe"],
                    row["gross_profit"],
                    row["total_assets"],
                )
            else:
                fail += 1
                logger.warning(
                    "TICKER %s/%s %s all fields empty (ROE/GP/TA all None)",
                    i,
                    len(tickers),
                    ticker,
                )
        except Exception as e:
            fail += 1
            logger.exception("TICKER %s/%s %s exception: %s", i, len(tickers), ticker, e)
            rows.append({k: None for k in FIELDNAMES} | {"ticker": ticker})

        if i % progress_every == 0 or i == len(tickers):
            _log_progress(i, len(tickers), succ, fail, started_at, client.get_call_count())

    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    write_csv(rows, output)

    elapsed = time.time() - started_at
    logger.info(
        "SUMMARY rows=%s succ=%s fail=%s elapsed=%.1fs api_calls=%s output=%s",
        len(rows),
        succ,
        fail,
        elapsed,
        client.get_call_count(),
        output,
    )
    logger.info("=== fmp_fundamentals DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
