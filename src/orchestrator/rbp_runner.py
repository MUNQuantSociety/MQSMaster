"""RBP forecast refresher daemon.

Refreshes the ``rbp_forecasts`` table every ``REFRESH_INTERVAL_SEC`` seconds
for the union of tickers across all enabled portfolios. Designed to be
launched from ``start.sh`` as a background PID, mirroring
``orchestrator/realTime/realtimeDataIngestor.py``.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import pytz

# --- Dual-path import shim, mirroring RBP/pipeline.py + portfolio_8/strategy.py.
try:
    from RBP.config import RBPConfig
    from RBP.service import RBPForecastService
except ImportError:  # pragma: no cover - fallback for ``src.``-prefixed layouts
    from src.RBP.config import RBPConfig  # type: ignore
    from src.RBP.service import RBPForecastService  # type: ignore

try:
    from common.database.MQSDBConnector import MQSDBConnector
except ImportError:  # pragma: no cover
    from src.common.database.MQSDBConnector import MQSDBConnector  # type: ignore


# --- Configuration ---
REFRESH_INTERVAL_SEC = 300  # 5 minutes
TIMEZONE = pytz.timezone("America/New_York")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PORTFOLIOS_DIR = PROJECT_ROOT / "src" / "portfolios"
PORTFOLIO_MANAGER_CONFIG = PORTFOLIOS_DIR / "portfolio_manager_config.json"
P6_UNIVERSE_PATH = PORTFOLIOS_DIR / "portfolio_6" / "universe.json"
UNIVERSE_BUILDER_PORTFOLIOS = {"6", "7", "8"}

logger = logging.getLogger(__name__)

# Module-level run flag, flipped by the signal handlers.
running = True


def _signal_handler(signum, _frame):
    """Set ``running=False`` so the main loop exits cleanly."""
    global running
    logger.info("Received signal %s; shutting down after current cycle.", signum)
    running = False


def collect_universe() -> List[str]:
    """Union of TICKERS across enabled portfolios (+ P6 universe.json if needed).

    A portfolio is "enabled" when its weight in ``portfolio_manager_config.json``
    is strictly greater than zero. P6/P7/P8 build their universe at runtime
    from ``portfolio_6/universe.json``, so that file is unioned in whenever
    any of those three is enabled.
    """
    try:
        with open(PORTFOLIO_MANAGER_CONFIG, "r") as f:
            pm_cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error("Could not load %s: %s", PORTFOLIO_MANAGER_CONFIG, exc)
        return []

    weights = pm_cfg.get("portfolio_weights", {}) or {}
    enabled_ids = [pid for pid, w in weights.items() if isinstance(w, (int, float)) and w > 0]
    logger.info("Enabled portfolios (weight>0): %s", enabled_ids)

    tickers: set = set()
    for pid in enabled_ids:
        cfg_path = PORTFOLIOS_DIR / f"portfolio_{pid}" / "config.json"
        try:
            with open(cfg_path, "r") as f:
                p_cfg = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("Skipping portfolio %s (cannot load %s): %s", pid, cfg_path, exc)
            continue
        portfolio_tickers = p_cfg.get("TICKERS", []) or []
        tickers.update(t for t in portfolio_tickers if isinstance(t, str))

    if any(pid in UNIVERSE_BUILDER_PORTFOLIOS for pid in enabled_ids):
        try:
            with open(P6_UNIVERSE_PATH, "r") as f:
                universe = json.load(f)
            tickers.update(t for t in universe if isinstance(t, str))
            logger.info("Unioned %d tickers from %s", len(universe), P6_UNIVERSE_PATH)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("Could not load P6 universe at %s: %s", P6_UNIVERSE_PATH, exc)

    return sorted(tickers)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("======= Starting RBP Forecast Runner =======")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Ensure rbp_forecasts table exists. Idempotent — uses IF NOT EXISTS.
    try:
        try:
            from src.common.database.schemaDefinitions import SchemaDefinitions
        except ImportError:
            from common.database.schemaDefinitions import SchemaDefinitions
        SchemaDefinitions().create_all_tables()
        logger.info("Schema bootstrap complete.")
    except Exception as exc:
        logger.exception("Schema bootstrap failed; continuing anyway (table may already exist): %s", exc)

    universe = collect_universe()
    if not universe:
        logger.warning(
            "Universe is empty (no enabled portfolios or all configs unreadable). "
            "Daemon will idle and re-check each cycle."
        )
    else:
        logger.info("Collected universe of %d tickers for RBP refresh.", len(universe))

    db = MQSDBConnector()
    config = RBPConfig(tickers=universe, lookback_days=365 * 5)
    service = RBPForecastService(config=config, db=db)

    try:
        while running:
            if not universe:
                # Idle: try to rebuild the universe in case configs were edited.
                logger.info("Empty universe; sleeping %ds before re-checking configs.", REFRESH_INTERVAL_SEC)
            else:
                t0 = time.time()
                try:
                    asof = datetime.now(TIMEZONE)
                    n = service.refresh(asof=asof)
                    logger.info("Refreshed %d forecasts in %.1fs", n, time.time() - t0)
                except Exception:
                    logger.exception("refresh() crashed; will retry next cycle")

            # Sleep in 1-second increments so SIGTERM/SIGINT is responsive.
            for _ in range(REFRESH_INTERVAL_SEC):
                if not running:
                    break
                time.sleep(1)

            # Refresh universe view between cycles so enabling/disabling a
            # portfolio in portfolio_manager_config.json takes effect without
            # a daemon restart.
            new_universe = collect_universe()
            if new_universe != universe:
                service.config.tickers = list(new_universe)
                logger.info("Universe changed: %d → %d tickers (preserved %d cached)",
                            len(universe), len(new_universe),
                            len(set(universe) & set(new_universe)))
                universe = new_universe
    finally:
        try:
            db.close_all_connections()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to close DB connections cleanly.")
        logger.info("======= RBP Forecast Runner Stopped =======")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
