"""
Build portfolio_6 candidate universe: S&P 500 + Nasdaq-100.

Writes the deduped, sorted ticker list to
  src/portfolios/portfolio_6/universe.json

Usage:
  python -m scripts.APIs.build_portfolio6_universe
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_APIS_DIR = REPO_ROOT / "scripts" / "APIs"
if str(SCRIPTS_APIS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_APIS_DIR))

from _fmp_helpers import FMPClient  # noqa: E402


OUTPUT_PATH = REPO_ROOT / "src" / "portfolios" / "portfolio_6" / "universe.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("build_portfolio6_universe")


def fetch_symbols(client: FMPClient, url: str, label: str) -> List[str]:
    logger.info("STEP: fetching %s constituents from %s", label, url)
    data = client.get(url, label=label)
    if not data or not isinstance(data, list):
        logger.error("STEP: %s returned no usable data.", label)
        return []
    symbols: List[str] = []
    for entry in data:
        if isinstance(entry, dict):
            sym = entry.get("symbol")
            if isinstance(sym, str) and sym.strip():
                symbols.append(sym.strip())
    logger.info("STEP: %s parsed -> %s tickers", label, len(symbols))
    if symbols:
        logger.info("STEP: %s sample first 5 -> %s", label, symbols[:5])
    return symbols


def main() -> int:
    start = time.time()
    logger.info("=== build_portfolio6_universe START ===")
    logger.info("REPO_ROOT=%s", REPO_ROOT)
    logger.info("OUTPUT_PATH=%s", OUTPUT_PATH)

    try:
        client = FMPClient(logger=logger)
    except ValueError as e:
        logger.error("Cannot init FMP client: %s", e)
        return 1

    sp500 = fetch_symbols(
        client,
        "https://financialmodelingprep.com/stable/sp500-constituent",
        "S&P 500",
    )
    nasdaq100 = fetch_symbols(
        client,
        "https://financialmodelingprep.com/stable/nasdaq-constituent",
        "Nasdaq-100",
    )

    logger.info("STEP: merging + deduplicating")
    universe = sorted({t for t in [*sp500, *nasdaq100] if t})
    logger.info(
        "STEP: merged -> %s unique (S&P 500=%s, Nasdaq-100=%s, overlap=%s)",
        len(universe),
        len(sp500),
        len(nasdaq100),
        len(sp500) + len(nasdaq100) - len(universe),
    )

    logger.info("STEP: writing %s tickers to %s", len(universe), OUTPUT_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(universe, f, indent=2)
    logger.info("STEP: write OK")

    elapsed = time.time() - start
    logger.info(
        "SUMMARY total_calls=%s | elapsed=%.2fs | tickers=%s",
        client.get_call_count(),
        elapsed,
        len(universe),
    )
    logger.info("=== build_portfolio6_universe DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
