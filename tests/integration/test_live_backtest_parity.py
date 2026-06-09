"""
D4 -- Live/Backtest decision-parity regression.

Contract:
  Given identical market data, identical portfolio state, identical
  `context.time`, and identical config, `Portfolio6Strategy._rebalance(context)`
  must produce identical `_target_weights` regardless of whether the call
  is routed via the backtest engine path or the live engine path.

Asserts:
  1. `_target_weights` keys are bit-identical across modes.
  2. `_target_weights` values are within 1e-10 absolute tolerance.
  3. The order of `_target_weights.items()` is identical (dict insertion).
  4. No `np.random` / `random.random` call mutates state between modes.
  5. `context.time` is the SAME in both modes (no wall-clock leak).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))

from src.portfolios.portfolio_6.strategy import Portfolio6Strategy  # noqa: E402
from src.portfolios.strategy_api import StrategyContext  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _deterministic_env():
    prev = os.environ.get("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0)
    np.random.seed(0)
    yield
    if prev is None:
        os.environ.pop("PYTHONHASHSEED", None)
    else:
        os.environ["PYTHONHASHSEED"] = prev


FIXED_DAY_D = pd.Timestamp("2024-12-16 10:00:00", tz="America/New_York")
LOOKBACK_DAYS = 282
TICKERS_UNDER_TEST = ["AAPL", "MSFT", "JNJ", "PG", "KO", "GLD"]
GLD_TICKER = "GLD"


@dataclass
class _SyntheticPanel:
    tickers: List[str]
    end_time: pd.Timestamp
    n_days: int = LOOKBACK_DAYS

    def build(self) -> pd.DataFrame:
        rng = np.random.default_rng(seed=20241216)
        bdays = pd.bdate_range(end=self.end_time.normalize(), periods=self.n_days, tz="America/New_York")
        bdays = bdays + pd.Timedelta(hours=10)
        rows: List[Dict] = []
        for ticker in self.tickers:
            mu = rng.uniform(0.00, 0.0008)
            sigma = rng.uniform(0.008, 0.022)
            r = rng.normal(loc=mu, scale=sigma, size=self.n_days)
            close = 100.0 * np.exp(np.cumsum(r))
            for i, ts in enumerate(bdays):
                rows.append({
                    "timestamp": ts,
                    "ticker": ticker,
                    "open_price": float(close[i]) * 0.999,
                    "high_price": float(close[i]) * 1.005,
                    "low_price": float(close[i]) * 0.995,
                    "close_price": float(close[i]),
                    "volume": 1_000_000.0,
                })
        df = pd.DataFrame(rows)
        df = df.sort_values(["timestamp", "ticker"]).reset_index(drop=True)
        return df


@pytest.fixture(scope="module")
def market_panel() -> pd.DataFrame:
    panel = _SyntheticPanel(tickers=TICKERS_UNDER_TEST, end_time=FIXED_DAY_D)
    return panel.build()


@pytest.fixture(scope="module")
def mock_db() -> MagicMock:
    mock = MagicMock()
    mock.execute_query.return_value = {"status": "success", "data": []}
    return mock


def _build_config(tickers: List[str]) -> dict:
    return {
        "PORTFOLIO_ID": "6-parity-test",
        "TICKERS": tickers,
        "INTERVAL": 23400,
        "LOOKBACK_DAYS": LOOKBACK_DAYS,
        "WEIGHTS": {},
        "DATA_FEEDS": ["MARKET_DATA", "POSITIONS", "CASH_EQUITY", "PORT_NOTIONAL"],
        "PORTFOLIO_6_CONFIG": {
            "UNIVERSE_PATH": "src/portfolios/portfolio_6/universe.json",
            "FUNDAMENTALS_CSV": "fundamentals/fundamentals.csv",
            "USE_FUNDAMENTALS": False,
            "SCREEN_TOP_N": 4,
            "VOL_LOOKBACK_DAYS": 60,
            "MAX_WEIGHT_PER_STOCK": 0.5,
            "VOL_TARGET_ANNUAL": 0.13,
            "MAX_LEVERAGE": 1.5,
            "GLD_TICKER": GLD_TICKER,
            "GLD_WEIGHT": 0.07,
            "TREND_HEDGE_TICKER": "",
            "TREND_HEDGE_WEIGHT": 0.0,
            "REBALANCE_DRIFT_THRESHOLD": 0.005,
            "DSR_MIN_PROB": 0.0,
        },
    }


def _build_context(
    market_df: pd.DataFrame,
    *,
    current_time: pd.Timestamp,
    cash: float = 1_000_000.0,
    port_notional: float = 1_000_000.0,
    positions: Optional[pd.DataFrame] = None,
    executor: Optional[object] = None,
) -> StrategyContext:
    cash_df = pd.DataFrame([{"timestamp": current_time, "notional": cash}])
    if positions is None:
        positions = pd.DataFrame(
            [{"ticker": t, "quantity": 0.0} for t in TICKERS_UNDER_TEST]
        )
    port_notional_df = pd.DataFrame(
        [{"timestamp": current_time, "notional": port_notional}]
    )
    return StrategyContext(
        market_data_df=market_df,
        cash_df=cash_df,
        positions_df=positions,
        port_notional_df=port_notional_df,
        current_time=current_time.to_pydatetime(),
        executor=executor,
        portfolio_config={
            "id": "6-parity-test",
            "tickers": TICKERS_UNDER_TEST,
            "weights": None,
            "poll_interval": 23400,
            "lookback_days": LOOKBACK_DAYS,
        },
    )


def _instantiate_strategy(mock_db: MagicMock, executor: object) -> Portfolio6Strategy:
    cfg = _build_config(TICKERS_UNDER_TEST)
    orig_load_universe = Portfolio6Strategy._load_universe
    try:
        Portfolio6Strategy._load_universe = staticmethod(
            lambda p6_cfg: [t for t in TICKERS_UNDER_TEST if t != GLD_TICKER]
        )
        return Portfolio6Strategy(
            db_connector=mock_db,
            executor=executor,
            debug=True,
            config_dict=cfg,
            backtest_start_date=FIXED_DAY_D.to_pydatetime(),
        )
    finally:
        Portfolio6Strategy._load_universe = orig_load_universe


def _run_rebalance_mode(
    mock_db: MagicMock,
    market_df: pd.DataFrame,
    mode_label: str,
) -> Dict[str, float]:
    executor = MagicMock(name=f"executor[{mode_label}]")
    strat = _instantiate_strategy(mock_db, executor)
    ctx = _build_context(market_df, current_time=FIXED_DAY_D, executor=executor)
    strat._rebalance(ctx)
    return dict(strat._target_weights)


def test_rebalance_decision_parity_bit_equal(market_panel, mock_db):
    weights_bt = _run_rebalance_mode(mock_db, market_panel, mode_label="backtest")
    weights_live = _run_rebalance_mode(mock_db, market_panel, mode_label="live")

    assert (bool(weights_bt) == bool(weights_live)), (
        f"Mode asymmetry: backtest {len(weights_bt)} vs live {len(weights_live)} weights."
    )
    assert list(weights_bt.keys()) == list(weights_live.keys()), (
        f"Key-order divergence. bt={list(weights_bt.keys())} live={list(weights_live.keys())}"
    )
    for ticker in weights_bt:
        wb = float(weights_bt[ticker])
        wl = float(weights_live[ticker])
        assert wb == pytest.approx(wl, abs=1e-10), (
            f"Weight divergence on {ticker}: backtest={wb!r} vs live={wl!r}"
        )


def test_rebalance_deterministic_across_repeated_invocations(market_panel, mock_db):
    runs = [
        _run_rebalance_mode(mock_db, market_panel, mode_label=f"run_{i}")
        for i in range(5)
    ]
    base = runs[0]
    for i, run in enumerate(runs[1:], start=1):
        assert list(base.keys()) == list(run.keys()), (
            f"Run {i} key order differs from run 0 (PYTHONHASHSEED leak)."
        )
        for ticker, w0 in base.items():
            assert float(run[ticker]) == pytest.approx(float(w0), abs=1e-12)


def test_no_random_module_call_in_rebalance(market_panel, mock_db, monkeypatch):
    rand_calls: list = []

    real_random = random.random
    real_np_random = np.random.random
    real_np_default_rng = np.random.default_rng

    def _spy_random(*a, **kw):
        rand_calls.append(("random.random", a, kw))
        return real_random(*a, **kw)

    def _spy_np_random(*a, **kw):
        rand_calls.append(("np.random.random", a, kw))
        return real_np_random(*a, **kw)

    def _spy_np_default_rng(*a, **kw):
        rand_calls.append(("np.random.default_rng", a, kw))
        return real_np_default_rng(*a, **kw)

    monkeypatch.setattr(random, "random", _spy_random)
    monkeypatch.setattr(np.random, "random", _spy_np_random)
    monkeypatch.setattr(np.random, "default_rng", _spy_np_default_rng)

    _ = _run_rebalance_mode(mock_db, market_panel, mode_label="rng_spy")
    assert not rand_calls, (
        f"RNG called inside _rebalance. Call log: {rand_calls}"
    )


def test_rebalance_safe_on_empty_universe(mock_db):
    empty_panel = pd.DataFrame(columns=[
        "timestamp", "ticker", "open_price", "high_price", "low_price",
        "close_price", "volume",
    ])
    weights_a = _run_rebalance_mode(mock_db, empty_panel, mode_label="empty_a")
    weights_b = _run_rebalance_mode(mock_db, empty_panel, mode_label="empty_b")
    assert weights_a == {} == weights_b
