# [Incomplete: going to implement V2] module for vectorized backtesting. This class provides a way to run fast backtests on OHLCV data using vectorized operations with pandas and numpy. It includes methods to generate signals based on a simple SMA crossover strategy with a z-score filter, calculate returns with transaction costs, compute performance metrics, and run Monte Carlo simulations. Additionally, it has functionality to run the same calendar window across multiple years for seasonal analysis, and to save results for downstream visualization. The design emphasizes efficiency and flexibility while maintaining clarity in the backtesting process.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class SeasonalRunResult:
    per_year_metrics: pd.DataFrame
    overlay_data: pd.DataFrame


class VectorBacktester:
    def __init__(
        self,
        data,
        commission=0.001,
        slippage=0.0005,
        initial_capital=100000,
        annualization_factor=252,
        *,
        copy_data=True,
        dtype="float64",
        store_intermediates=True,
    ):
        """
        Vectorized backtester for OHLCV data.

        Parameters:
        - data: pd.DataFrame with OHLCV columns, datetime index
        - commission: per-trade commission as a decimal fraction (e.g., 0.001 = 0.1% = 10 bps)
        - slippage: per-trade slippage as a decimal fraction (e.g., 0.0005 = 0.05% = 5 bps)
        - initial_capital: starting capital
        """
        self.data = data.copy() if copy_data else data
        self.data.columns = [str(col).lower() for col in self.data.columns]
        if "close" not in self.data.columns:
            raise ValueError("Input data must contain a 'close' column.")
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        self.data = self.data.sort_index()
        self.commission = commission
        self.slippage = slippage
        self.initial_capital = initial_capital
        self.annualization_factor = annualization_factor
        self.dtype = np.float32 if str(dtype).lower() == "float32" else np.float64
        self.store_intermediates = store_intermediates
        self.results = None
        self.metrics = None
        self.mc_stats = None
        self.mc_paths = None

    def _compute_signals_array(
        self,
        fast_window=20,
        slow_window=50,
        z_entry=2.0,
        z_exit=0.5,
        *,
        store_intermediates=True,
    ):
        close = self.data["close"].astype(self.dtype)
        fast_series = close.rolling(fast_window).mean()
        slow_series = close.rolling(slow_window).mean()

        rolling_20 = close.rolling(20)
        zscore = (close - rolling_20.mean()) / rolling_20.std()

        sma_cross_up = (fast_series > slow_series) & (
            fast_series.shift(1) <= slow_series.shift(1)
        )
        sma_cross_down = (fast_series < slow_series) & (
            fast_series.shift(1) >= slow_series.shift(1)
        )
        zscore_exit = np.abs(zscore) < z_exit

        signal = np.full(len(self.data), np.nan, dtype=self.dtype)
        entry_mask = (sma_cross_up & (zscore < -z_entry)).to_numpy(
            dtype=bool, copy=False
        )
        exit_mask = (sma_cross_down | zscore_exit).to_numpy(dtype=bool, copy=False)
        signal[entry_mask] = 1.0
        signal[exit_mask] = 0.0

        position = (
            pd.Series(signal, index=self.data.index)
            .ffill()
            .fillna(0.0)
            .shift(1)
            .fillna(0.0)
            .to_numpy(dtype=self.dtype, copy=False)
        )

        if store_intermediates:
            self.data["sma_fast"] = fast_series
            self.data["sma_slow"] = slow_series
            self.data["zscore"] = zscore

        return position

    def _compute_returns_array(self, position: np.ndarray):
        close = self.data["close"].to_numpy(dtype=self.dtype, copy=False)
        market_returns = np.zeros(len(close), dtype=self.dtype)

        if len(close) > 1:
            prev = close[:-1]
            curr = close[1:]
            valid = np.isfinite(prev) & np.isfinite(curr) & (prev != 0)
            diff = np.zeros(len(curr), dtype=self.dtype)
            diff[valid] = (curr[valid] - prev[valid]) / prev[valid]
            market_returns[1:] = diff

        gross_returns = position * market_returns
        position_diff = np.abs(np.diff(position, prepend=position[0]))
        txn_costs = position_diff * (self.commission + self.slippage)
        strategy_returns = gross_returns - txn_costs

        market_clean = np.nan_to_num(market_returns, nan=0.0, posinf=0.0, neginf=0.0)
        strategy_clean = np.nan_to_num(
            strategy_returns, nan=0.0, posinf=0.0, neginf=0.0
        )
        cum_market = np.cumprod((1.0 + market_clean).astype(self.dtype))
        cum_strategy = np.cumprod((1.0 + strategy_clean).astype(self.dtype))

        return {
            "market_returns": market_returns,
            "gross_returns": gross_returns,
            "strategy_returns": strategy_returns,
            "cum_market": cum_market,
            "cum_strategy": cum_strategy,
        }

    def _compute_metrics_from_arrays(
        self, strategy_returns: np.ndarray, cum_strategy: np.ndarray
    ):
        valid = np.isfinite(strategy_returns) & np.isfinite(cum_strategy)
        returns = strategy_returns[valid]
        cumulative = cum_strategy[valid]

        if returns.size == 0 or cumulative.size == 0:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "annual_vol": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
            }

        total_return = float(cumulative[-1] - 1.0)
        periods = returns.size
        annual_return = (
            float(cumulative[-1] ** (self.annualization_factor / periods) - 1.0)
            if periods > 0 and cumulative[-1] > 0
            else 0.0
        )

        ret_std = float(returns.std())
        annual_vol = float(ret_std * np.sqrt(self.annualization_factor))
        sharpe = (
            float(returns.mean() / ret_std * np.sqrt(self.annualization_factor))
            if ret_std > 0
            else 0.0
        )

        running_max = np.maximum.accumulate(cumulative)
        max_drawdown = float(np.min((cumulative / running_max) - 1.0))

        wins_mask = returns > 0
        losses_mask = returns < 0
        win_rate = float(wins_mask.mean())
        avg_win = float(returns[wins_mask].mean()) if wins_mask.any() else 0.0
        avg_loss = float(returns[losses_mask].mean()) if losses_mask.any() else 0.0
        wins_sum = float(returns[wins_mask].sum()) if wins_mask.any() else 0.0
        losses_sum = float(returns[losses_mask].sum()) if losses_mask.any() else 0.0
        profit_factor = abs(wins_sum / losses_sum) if losses_sum != 0 else None

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
        }

    def _build_results_frame(self):
        columns = [
            "close",
            "position",
            "market_returns",
            "strategy_returns",
            "cum_market",
            "cum_strategy",
        ]
        return self.data[columns].dropna()

    @staticmethod
    def _coerce_benchmark_prices(df: pd.DataFrame, benchmark=None) -> pd.Series:
        if benchmark is None:
            return df["close"]

        if isinstance(benchmark, str):
            benchmark_col = benchmark.lower()
            if benchmark_col not in df.columns:
                raise ValueError(f"Benchmark column '{benchmark}' not found in data.")
            return df[benchmark_col]

        if isinstance(benchmark, pd.Series):
            series = benchmark.copy()
            if not isinstance(series.index, pd.DatetimeIndex):
                series.index = pd.to_datetime(series.index)
            return series.reindex(df.index).ffill().bfill()

        raise TypeError(
            "benchmark must be None, a column name (str), or a pandas Series of prices."
        )

    @staticmethod
    def _normalize_path_to_returns(path: pd.Series) -> pd.Series:
        series = pd.to_numeric(path, errors="coerce").ffill().bfill()
        if series.empty:
            return series

        base = float(series.iloc[0])
        if not np.isfinite(base) or abs(base) < 1e-12:
            return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
        return (series / base) - 1.0

    @staticmethod
    def _safe_year_replace(ts: pd.Timestamp, year: int) -> pd.Timestamp:
        try:
            return ts.replace(year=year)
        except ValueError:
            if ts.month == 2 and ts.day == 29:
                return ts.replace(year=year, day=28)
            raise

    @classmethod
    def _window_for_year(
        cls, anchor_start: pd.Timestamp, anchor_end: pd.Timestamp, year: int
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        start = cls._safe_year_replace(anchor_start, year)
        end = cls._safe_year_replace(anchor_end, year)
        if end < start:
            end = cls._safe_year_replace(anchor_end, year + 1)
        return start, end

    @staticmethod
    def _align_to_index_tz(ts: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
        """Align a timestamp's timezone-awareness to match a target DatetimeIndex."""
        index_tz = index.tz
        if index_tz is None:
            if ts.tzinfo is not None:
                return ts.tz_convert("UTC").tz_localize(None)
            return ts

        if ts.tzinfo is None:
            return ts.tz_localize(index_tz)
        return ts.tz_convert(index_tz)

    def generate_signals(self, fast_window=20, slow_window=50, z_entry=2.0, z_exit=0.5):
        """Generate position signals using SMA crossover + z-score filter."""
        position = self._compute_signals_array(
            fast_window=fast_window,
            slow_window=slow_window,
            z_entry=z_entry,
            z_exit=z_exit,
            store_intermediates=self.store_intermediates,
        )
        self.data["position"] = position
        return self

    def calculate_returns(self):
        """Compute strategy returns with costs."""
        if "position" not in self.data.columns:
            raise ValueError("Position not found. Run generate_signals first.")

        position = self.data["position"].to_numpy(dtype=self.dtype, copy=False)
        out = self._compute_returns_array(position)
        self.data["market_returns"] = out["market_returns"]
        self.data["gross_returns"] = out["gross_returns"]
        self.data["strategy_returns"] = out["strategy_returns"]
        self.data["cum_market"] = out["cum_market"]
        self.data["cum_strategy"] = out["cum_strategy"]

        return self

    def compute_metrics(self):
        """Calculate key performance metrics."""
        if (
            "strategy_returns" not in self.data.columns
            or "cum_strategy" not in self.data.columns
        ):
            raise ValueError("Returns not found. Run calculate_returns first.")

        strategy_returns = self.data["strategy_returns"].to_numpy(
            dtype=self.dtype, copy=False
        )
        cum_strategy = self.data["cum_strategy"].to_numpy(dtype=self.dtype, copy=False)
        self.metrics = self._compute_metrics_from_arrays(strategy_returns, cum_strategy)
        return self.metrics

    def run(
        self,
        fast_window=20,
        slow_window=50,
        z_entry=2.0,
        z_exit=0.5,
        *,
        quick=False,
        store_intermediates=None,
    ):
        """Full backtest pipeline."""
        use_intermediates = (
            self.store_intermediates
            if store_intermediates is None
            else bool(store_intermediates)
        )
        if quick:
            use_intermediates = False

        position = self._compute_signals_array(
            fast_window=fast_window,
            slow_window=slow_window,
            z_entry=z_entry,
            z_exit=z_exit,
            store_intermediates=use_intermediates,
        )
        out = self._compute_returns_array(position)

        self.data["position"] = position
        self.data["market_returns"] = out["market_returns"]
        self.data["gross_returns"] = out["gross_returns"]
        self.data["strategy_returns"] = out["strategy_returns"]
        self.data["cum_market"] = out["cum_market"]
        self.data["cum_strategy"] = out["cum_strategy"]

        self.metrics = self._compute_metrics_from_arrays(
            out["strategy_returns"],
            out["cum_strategy"],
        )
        self.results = self._build_results_frame()
        return self.results

    def run_from_returns(
        self,
        strategy_returns,
        *,
        market_returns=None,
        close_series=None,
    ):
        """Populate backtest outputs from externally computed return streams."""
        strategy = pd.Series(strategy_returns, index=self.data.index, dtype=self.dtype)
        strategy = strategy.fillna(0.0)

        if market_returns is None:
            if close_series is None:
                close_series = self.data["close"]
            market = (
                pd.Series(close_series, index=self.data.index).pct_change().fillna(0.0)
            )
        else:
            market = pd.Series(
                market_returns, index=self.data.index, dtype=self.dtype
            ).fillna(0.0)

        strategy_np = strategy.to_numpy(dtype=self.dtype, copy=False)
        market_np = market.to_numpy(dtype=self.dtype, copy=False)
        cum_strategy = np.cumprod((1.0 + strategy_np).astype(self.dtype, copy=False))
        cum_market = np.cumprod((1.0 + market_np).astype(self.dtype, copy=False))

        if close_series is None:
            close_series = self.data["close"]
        close_series = pd.Series(close_series, index=self.data.index).ffill().bfill()

        self.data["close"] = close_series.to_numpy(dtype=self.dtype, copy=False)
        self.data["position"] = 0.0
        self.data["market_returns"] = market_np
        self.data["gross_returns"] = strategy_np
        self.data["strategy_returns"] = strategy_np
        self.data["cum_market"] = cum_market
        self.data["cum_strategy"] = cum_strategy

        self.metrics = self._compute_metrics_from_arrays(strategy_np, cum_strategy)
        self.results = self._build_results_frame()
        return self.results

    def monte_carlo(
        self,
        n_sims=10000,
        method="bootstrap",
        block_size=1,
        seed=None,
    ):
        """Run Monte Carlo simulations on realized strategy returns.

        Parameters:
        - n_sims: number of simulated paths
        - method: 'bootstrap' or 'parametric'
        - block_size: contiguous block size for bootstrap; set >1 for block bootstrap
        - seed: optional RNG seed for deterministic simulations
        """
        if "strategy_returns" not in self.data.columns:
            raise ValueError("Strategy returns not found. Run run() first.")

        returns = (
            self.data["strategy_returns"]
            .dropna()
            .to_numpy(dtype=self.dtype, copy=False)
        )
        if returns.size == 0:
            raise ValueError(
                "Strategy returns are empty. Run run() with sufficient data."
            )

        n_sims = max(1, int(n_sims))
        method = str(method).lower().strip()
        block_size = max(1, int(block_size))

        rng = np.random.default_rng(seed)
        periods = returns.size

        if method == "bootstrap":
            if block_size > 1:
                if block_size > periods:
                    block_size = periods

                n_blocks = int(np.ceil(periods / block_size))
                max_start = periods - block_size + 1
                block_starts = rng.integers(0, max_start, size=(n_sims, n_blocks))
                offsets = np.arange(block_size)
                sample_idx = block_starts[..., None] + offsets
                sample_idx = sample_idx.reshape(n_sims, -1)[:, :periods]
                sim_returns = returns[sample_idx]
            else:
                sample_idx = rng.integers(0, periods, size=(n_sims, periods))
                sim_returns = returns[sample_idx]
        elif method == "parametric":
            mu = float(np.mean(returns))
            sigma = float(np.std(returns))
            sim_returns = rng.normal(mu, sigma, size=(n_sims, periods)).astype(
                self.dtype,
                copy=False,
            )
        else:
            raise ValueError("method must be either 'bootstrap' or 'parametric'.")

        sim_cum = np.cumprod((1.0 + sim_returns).astype(self.dtype, copy=False), axis=1)
        final_returns = sim_cum[:, -1] - 1.0
        running_max = np.maximum.accumulate(sim_cum, axis=1)
        drawdowns = (sim_cum / running_max) - 1.0
        max_drawdowns = np.min(drawdowns, axis=1)

        self.mc_stats = {
            "n_sims": int(n_sims),
            "periods": int(periods),
            "method": method,
            "block_size": int(block_size),
            "mean_return": float(np.mean(final_returns)),
            "median_return": float(np.median(final_returns)),
            "p10_return": float(np.percentile(final_returns, 10)),
            "p90_return": float(np.percentile(final_returns, 90)),
            "max_drawdown_p5": float(np.percentile(max_drawdowns, 5)),
            "max_drawdown_p50": float(np.percentile(max_drawdowns, 50)),
        }
        self.mc_paths = sim_cum

        return sim_cum, self.mc_stats

    def run_same_window_previous_years(
        self,
        start_date,
        end_date,
        years_back=5,
        *,
        benchmark=None,
        highlight_years: Iterable[int] | None = None,
        fast_window=20,
        slow_window=50,
        z_entry=2.0,
        z_exit=0.5,
        quick=True,
        benchmark_label="Benchmark",
    ) -> SeasonalRunResult:
        """
        Run fast vectorized backtests over the same calendar window for previous years.

        Returns:
        - per_year_metrics: one row per window year
        - overlay_data: normalized return paths (strategy and benchmark) by relative day
        """
        anchor_start = self._align_to_index_tz(
            pd.Timestamp(start_date), self.data.index
        )
        anchor_end = self._align_to_index_tz(pd.Timestamp(end_date), self.data.index)
        end_year = anchor_start.year
        years = [end_year - offset for offset in range(years_back + 1)]

        metrics_rows = []
        overlay_frames = []

        for year in sorted(years):
            window_start, window_end = self._window_for_year(
                anchor_start, anchor_end, year
            )
            window_data = self.data.loc[
                (self.data.index >= window_start) & (self.data.index <= window_end)
            ].copy()

            if (
                window_data.empty
                or len(window_data) < max(fast_window, slow_window) + 5
            ):
                continue

            has_external_returns = (
                "strategy_returns" in window_data.columns
                and "market_returns" in window_data.columns
            )

            if has_external_returns:
                strategy_returns = (
                    pd.to_numeric(window_data["strategy_returns"], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=self.dtype, copy=False)
                )
                cum_strategy = np.cumprod(
                    (1.0 + strategy_returns).astype(self.dtype, copy=False)
                )
                year_metrics = dict(
                    self._compute_metrics_from_arrays(strategy_returns, cum_strategy)
                )
                strategy_path = pd.Series(cum_strategy, index=window_data.index)
            else:
                bt = VectorBacktester(
                    data=window_data,
                    commission=self.commission,
                    slippage=self.slippage,
                    initial_capital=self.initial_capital,
                    annualization_factor=self.annualization_factor,
                    copy_data=False,
                    store_intermediates=not quick,
                )
                results = bt.run(
                    fast_window=fast_window,
                    slow_window=slow_window,
                    z_entry=z_entry,
                    z_exit=z_exit,
                    quick=quick,
                )
                year_metrics = dict(bt.metrics or {})
                strategy_path = (
                    results["cum_strategy"].reindex(window_data.index).ffill().bfill()
                )

            year_metrics["year"] = year
            year_metrics["window_start"] = window_start
            year_metrics["window_end"] = window_end
            metrics_rows.append(year_metrics)

            if benchmark is None and "market_returns" in window_data.columns:
                benchmark_returns = pd.to_numeric(
                    window_data["market_returns"], errors="coerce"
                ).fillna(0.0)
            else:
                benchmark_prices = self._coerce_benchmark_prices(window_data, benchmark)
                benchmark_returns = benchmark_prices.pct_change().fillna(0.0)
            bench_path = (1 + benchmark_returns).cumprod()

            strategy_norm = self._normalize_path_to_returns(strategy_path)
            benchmark_norm = self._normalize_path_to_returns(bench_path)

            frame = pd.DataFrame(
                {
                    "date": window_data.index,
                    "day": np.arange(len(window_data), dtype=int),
                    "year": year,
                    "strategy_return": strategy_norm.to_numpy(),
                    "benchmark_return": benchmark_norm.to_numpy(),
                    "benchmark_label": benchmark_label,
                }
            )
            overlay_frames.append(frame)

        per_year_metrics = pd.DataFrame(metrics_rows)
        overlay_data = (
            pd.concat(overlay_frames, ignore_index=True)
            if overlay_frames
            else pd.DataFrame(
                columns=[
                    "date",
                    "day",
                    "year",
                    "strategy_return",
                    "benchmark_return",
                    "benchmark_label",
                ]
            )
        )

        if highlight_years is not None and not overlay_data.empty:
            highlight_set = {int(y) for y in highlight_years}
            overlay_data["is_highlight"] = overlay_data["year"].isin(highlight_set)

        return SeasonalRunResult(
            per_year_metrics=per_year_metrics,
            overlay_data=overlay_data,
        )

    @staticmethod
    def save_seasonal_results(
        seasonal_result: SeasonalRunResult,
        output_dir,
        *,
        metrics_filename="seasonal_metrics.csv",
        overlay_filename="overlay_return_paths.csv",
    ):
        """Persist seasonal metrics and overlay paths for downstream visualization."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        metrics_path = output_path / metrics_filename
        overlay_path = output_path / overlay_filename

        seasonal_result.per_year_metrics.to_csv(metrics_path, index=False)
        seasonal_result.overlay_data.to_csv(overlay_path, index=False)

        return {
            "metrics_path": str(metrics_path),
            "overlay_path": str(overlay_path),
        }

    def run_and_save_same_window_previous_years(
        self,
        output_dir=None,
        *,
        start_date,
        end_date,
        years_back=5,
        benchmark=None,
        benchmark_label="Benchmark",
        highlight_years: Iterable[int] | None = None,
        fast_window=20,
        slow_window=50,
        z_entry=2.0,
        z_exit=0.5,
        quick=True,
        metrics_filename="seasonal_metrics.csv",
        overlay_filename="overlay_return_paths.csv",
    ):
        """Run seasonal vectorized backtest and persist overlay artifacts in one call."""
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent / "data"

        seasonal_result = self.run_same_window_previous_years(
            start_date=start_date,
            end_date=end_date,
            years_back=years_back,
            benchmark=benchmark,
            highlight_years=highlight_years,
            fast_window=fast_window,
            slow_window=slow_window,
            z_entry=z_entry,
            z_exit=z_exit,
            quick=quick,
            benchmark_label=benchmark_label,
        )

        saved_paths = self.save_seasonal_results(
            seasonal_result=seasonal_result,
            output_dir=output_dir,
            metrics_filename=metrics_filename,
            overlay_filename=overlay_filename,
        )

        return {
            "seasonal_result": seasonal_result,
            **saved_paths,
        }

    def plot_overlay(
        self,
        overlay_data: pd.DataFrame,
        *,
        highlight_years: Iterable[int] | None = None,
        benchmark_label="Benchmark",
    ):
        """Overlay same-window yearly strategy and benchmark return paths."""
        if overlay_data.empty:
            raise ValueError(
                "overlay_data is empty. Run run_same_window_previous_years first."
            )

        fig, ax = plt.subplots(figsize=(12, 7))
        highlight_set = {int(y) for y in highlight_years} if highlight_years else set()

        for year, group in overlay_data.groupby("year"):
            is_highlight = year in highlight_set if highlight_set else True
            alpha = 0.95 if is_highlight else 0.22
            lw = 2.1 if is_highlight else 1.0

            ax.plot(
                group["day"],
                group["strategy_return"],
                linewidth=lw,
                alpha=alpha,
                label=f"{year} Strategy",
            )
            ax.plot(
                group["day"],
                group["benchmark_return"],
                linestyle="--",
                linewidth=max(1.0, lw - 0.3),
                alpha=max(0.2, alpha - 0.15),
                label=f"{year} {benchmark_label}",
            )

        ax.axhline(0, color="grey", linestyle=":", linewidth=1)
        ax.set_title("Same-Window Return Paths Across Years")
        ax.set_xlabel("Day in Window")
        ax.set_ylabel("Normalized Return")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.show()

    def plot(self):
        """Plot equity curves and drawdown."""
        if self.results is None or self.results.empty:
            raise ValueError("No results to plot. Run the backtest first.")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Equity curves
        ax1.plot(
            self.results.index,
            self.results["cum_strategy"],
            label="Strategy",
            linewidth=2,
        )
        ax1.plot(
            self.results.index,
            self.results["cum_market"],
            label="Buy & Hold",
            alpha=0.7,
        )
        ax1.set_ylabel("Cumulative Returns")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        strategy_dd = (
            self.results["cum_strategy"] / self.results["cum_strategy"].cummax()
        ) - 1
        market_dd = (
            self.results["cum_market"] / self.results["cum_market"].cummax()
        ) - 1
        ax2.fill_between(
            self.results.index,
            strategy_dd,
            0,
            alpha=0.3,
            color="green",
            label="Strategy DD",
        )
        ax2.fill_between(
            self.results.index, market_dd, 0, alpha=0.3, color="red", label="Market DD"
        )
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Date")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def summary(self):
        """Print formatted results."""
        if self.metrics is None:
            raise ValueError("No metrics available. Run the backtest first.")

        print("=== BACKTEST SUMMARY ===")
        print(f"Total Return:   {self.metrics['total_return']:.1%}")
        print(f"Annual Return:  {self.metrics['annual_return']:.1%}")
        print(f"Annual Vol:     {self.metrics['annual_vol']:.1%}")
        print(f"Sharpe Ratio:   {self.metrics['sharpe']:.2f}")
        print(f"Max Drawdown:   {self.metrics['max_drawdown']:.1%}")
        print(f"Win Rate:       {self.metrics['win_rate']:.1%}")
        print(f"Profit Factor:  {self.metrics['profit_factor']:.2f}")
