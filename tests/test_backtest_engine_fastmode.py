import builtins
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.backtest.backtest_engine as backtest_engine_module
from src.backtest.backtest_engine import BacktestEngine
from src.backtest.vector_strategy_adapters import ADAPTERS_BY_CLASSNAME


class _DummyPortfolio:
    def __init__(self):
        self.portfolio_id = "test_portfolio"
        self.tickers = ["AAPL"]


class _FakeVectorBacktester:
    def __init__(
        self,
        data,
        commission=0.0,
        slippage=0.0,
        initial_capital=100000.0,
        store_intermediates=False,
        **kwargs,
    ):
        self.data = data
        self.metrics = {"total_return": 0.02, "sharpe": 1.1}
        self.monte_carlo_calls = 0

    def run(self, quick=True):
        idx = pd.date_range("2025-01-01", periods=5, freq="D")
        return pd.DataFrame(
            {
                "close": np.array([100, 101, 102, 103, 104], dtype=float),
                "cum_strategy": np.array([1.0, 1.01, 1.015, 1.018, 1.02], dtype=float),
                "cum_market": np.array([1.0, 1.005, 1.01, 1.012, 1.013], dtype=float),
            },
            index=idx,
        )

    def run_from_returns(
        self, strategy_returns, market_returns=None, close_series=None
    ):
        idx = pd.date_range("2025-01-01", periods=5, freq="D")
        self.metrics = {"total_return": 0.02, "sharpe": 1.1}
        return pd.DataFrame(
            {
                "close": np.array([100, 101, 102, 103, 104], dtype=float),
                "cum_strategy": np.array([1.0, 1.01, 1.015, 1.018, 1.02], dtype=float),
                "cum_market": np.array([1.0, 1.005, 1.01, 1.012, 1.013], dtype=float),
            },
            index=idx,
        )

    def run_and_save_same_window_previous_years(self, **kwargs):
        return {
            "metrics_path": "ignored",
            "overlay_path": "ignored",
        }

    def monte_carlo(self, n_sims=10000, method="bootstrap", block_size=1, seed=None):
        self.monte_carlo_calls += 1
        sims = np.full((n_sims, 5), 1.01, dtype=float)
        stats = {
            "n_sims": n_sims,
            "method": method,
            "block_size": block_size,
            "mean_return": 0.01,
        }
        return sims, stats


def _patch_tracking_backtester(monkeypatch):
    created_instances = []

    class _TrackingFakeVectorBacktester(_FakeVectorBacktester):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created_instances.append(self)

    monkeypatch.setattr(
        backtest_engine_module,
        "VectorBacktester",
        _TrackingFakeVectorBacktester,
    )
    return created_instances


def _build_daily_close_df():
    idx = pd.date_range("2025-01-01", periods=30, freq="D")
    return pd.DataFrame(
        {
            "ticker": ["AAPL"] * len(idx),
            "trade_date": idx,
            "close_price": np.linspace(100, 120, len(idx)),
        }
    )


def test_fastmode_does_not_run_monte_carlo_when_disabled(monkeypatch, tmp_path: Path):
    created_instances = _patch_tracking_backtester(monkeypatch)
    monkeypatch.setattr(
        backtest_engine_module,
        "get_vector_adapter_for_portfolio",
        lambda portfolio: (
            lambda close_df: type(
                "SignalResult",
                (),
                {
                    "target_weights": pd.DataFrame(
                        1.0, index=close_df.index, columns=close_df.columns
                    )
                },
            )()
        ),
    )

    engine = BacktestEngine(db_connector=object(), backtest_executor=None)
    engine.setup(
        portfolio_classes=[],
        start_date="2025-01-01",
        end_date="2025-01-31",
        initial_capital=1000000.0,
        slippage=0.0,
        backtest_mode="fast",
        fast_config={"mc_enabled": False, "years_back": 2, "benchmark_label": "SPY"},
    )

    monkeypatch.setattr(
        engine,
        "_fetch_fast_daily_close_data",
        lambda **kwargs: _build_daily_close_df(),
    )
    monkeypatch.setattr(
        engine, "_build_fast_output_dir", lambda portfolio_id: str(tmp_path)
    )

    engine._run_fast_vectorized(_DummyPortfolio())

    assert sum(i.monte_carlo_calls for i in created_instances) == 0
    assert not (tmp_path / "monte_carlo_summary_metrics.csv").exists()
    assert not (tmp_path / "monte_carlo_percentile_paths.csv").exists()


def test_fastmode_runs_monte_carlo_when_enabled(monkeypatch, tmp_path: Path):
    created_instances = _patch_tracking_backtester(monkeypatch)
    monkeypatch.setattr(
        backtest_engine_module,
        "get_vector_adapter_for_portfolio",
        lambda portfolio: (
            lambda close_df: type(
                "SignalResult",
                (),
                {
                    "target_weights": pd.DataFrame(
                        1.0, index=close_df.index, columns=close_df.columns
                    )
                },
            )()
        ),
    )

    engine = BacktestEngine(db_connector=object(), backtest_executor=None)
    engine.setup(
        portfolio_classes=[],
        start_date="2025-01-01",
        end_date="2025-01-31",
        initial_capital=1000000.0,
        slippage=0.0,
        backtest_mode="fast",
        fast_config={
            "mc_enabled": True,
            "mc_n_sims": 25,
            "mc_method": "bootstrap",
            "mc_block_size": 3,
            "mc_seed": 7,
            "mc_plot_percentiles": [10, 50, 90],
            "years_back": 2,
            "benchmark_label": "SPY",
        },
    )

    monkeypatch.setattr(
        engine,
        "_fetch_fast_daily_close_data",
        lambda **kwargs: _build_daily_close_df(),
    )
    monkeypatch.setattr(
        engine, "_build_fast_output_dir", lambda portfolio_id: str(tmp_path)
    )

    engine._run_fast_vectorized(_DummyPortfolio())

    assert sum(i.monte_carlo_calls for i in created_instances) == 1
    assert (tmp_path / "monte_carlo_summary_metrics.csv").is_file()
    assert (tmp_path / "monte_carlo_percentile_paths.csv").is_file()


def test_fastmode_skips_without_adapter(monkeypatch, tmp_path: Path):
    created_instances = _patch_tracking_backtester(monkeypatch)
    monkeypatch.setattr(
        backtest_engine_module,
        "get_vector_adapter_for_portfolio",
        lambda portfolio: None,
    )

    engine = BacktestEngine(db_connector=object(), backtest_executor=None)
    engine.setup(
        portfolio_classes=[],
        start_date="2025-01-01",
        end_date="2025-01-31",
        initial_capital=1000000.0,
        slippage=0.0,
        backtest_mode="fast",
        fast_config={"mc_enabled": True},
    )

    monkeypatch.setattr(
        engine,
        "_fetch_fast_daily_close_data",
        lambda **kwargs: _build_daily_close_df(),
    )
    monkeypatch.setattr(
        engine, "_build_fast_output_dir", lambda portfolio_id: str(tmp_path)
    )

    engine._run_fast_vectorized(_DummyPortfolio())

    assert sum(i.monte_carlo_calls for i in created_instances) == 0
    assert not (tmp_path / "performance_timeseries_absolute.csv").exists()


def test_adapter_registry_contains_expected_strategies():
    assert "MomentumStrategy" in ADAPTERS_BY_CLASSNAME
    assert "VolMomentum" in ADAPTERS_BY_CLASSNAME
    assert "RegimeAdaptiveStrategy" in ADAPTERS_BY_CLASSNAME
    assert "TrendRotateStrategy" in ADAPTERS_BY_CLASSNAME


def test_run_fastmode_skips_strategy_init_indicator_registration(
    monkeypatch, tmp_path: Path
):
    class MomentumStrategy:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Strategy __init__ should not run in fast mode")

    fake_class_file = tmp_path / "portfolio_x" / "strategy.py"
    fake_config_path = tmp_path / "portfolio_x" / "config.json"
    fake_class_file.parent.mkdir(parents=True, exist_ok=True)
    fake_config = {
        "PORTFOLIO_ID": "2",
        "TICKERS": ["AAPL", "MSFT"],
    }

    monkeypatch.setattr(
        backtest_engine_module.inspect,
        "getfile",
        lambda _cls: str(fake_class_file),
    )
    monkeypatch.setattr(
        backtest_engine_module.os.path,
        "exists",
        lambda path: path == str(fake_config_path),
    )

    original_open = builtins.open

    def _fake_open(path, mode="r", *args, **kwargs):
        if path == str(fake_config_path) and "r" in mode:
            return io.StringIO(json.dumps(fake_config))
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _fake_open)

    captured = {}

    def _capture_fast_run(self, portfolio_instance):
        captured["portfolio_id"] = portfolio_instance.portfolio_id
        captured["tickers"] = portfolio_instance.tickers
        captured["class_name"] = portfolio_instance.__class__.__name__

    monkeypatch.setattr(BacktestEngine, "_run_fast_vectorized", _capture_fast_run)

    engine = BacktestEngine(db_connector=object(), backtest_executor=None)
    engine.setup(
        portfolio_classes=[MomentumStrategy],
        start_date="2025-01-01",
        end_date="2025-01-31",
        initial_capital=1000000.0,
        backtest_mode="fast",
    )

    engine.run()

    assert captured["portfolio_id"] == 2
    assert captured["tickers"] == ["AAPL", "MSFT"]
    assert captured["class_name"] == "MomentumStrategy"
