"""
File for the BacktestEngine class.
"""
# src/backtest/backtest_engine.py

import os
import inspect
import json
import logging
from logging import Logger
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.core.tools.datetimes import DatetimeScalar

from src.backtest.cost_model import CostModel
from src.common.database.MQSDBConnector import MQSDBConnector
from src.portfolios.portfolio_BASE.strategy import BasePortfolio

from .reporting import (
    _calculate_portfolio_risk_components,
    _calculate_rolling_portfolio_risk,
)
from .runner import BacktestRunner
from .vector_strategy_adapters import get_vector_adapter_for_portfolio
from .vectorized_backtest import VectorBacktester


class BacktestEngine:
    """
    The BacktestEngine orchestrates backtesting runs.
    It is updated to load portfolio configurations dynamically.
    """

    def __init__(
        self,
        db_connector: "MQSDBConnector",
        backtest_executor = None,
        backtest_output_root: str | None = None,
    ):
        self.db_connector: MQSDBConnector = db_connector
        self.backtest_executor = backtest_executor
        self.logger: Logger = logging.getLogger(self.__class__.__name__)
        self.backtest_output_root: str | None = backtest_output_root
        self.portfolio_classes: list[type[BasePortfolio]] = []
        self.start_date: str = ""
        self.end_date: str = ""
        self.initial_capital: float = 0.0
        self.slippage: float = 0.0
        self.cost_model: CostModel | None = None
        self.backtest_mode: str = "event"
        self.fast_config: dict[str, int | str | bool | list[int] | None] = self._default_fast_config()

    @staticmethod
    def _default_fast_config() -> dict[str, int | str | bool | list[int] | None]:
        return {
            "years_back": 3,
            "benchmark_label": "Benchmark",
            "quick": True,
            "mc_enabled": False,
            "mc_n_sims": 10000,
            "mc_method": "bootstrap",
            "mc_block_size": 1,
            "mc_seed": None,
            "mc_plot_percentiles": [10, 50, 90],
        }

    @classmethod
    def _normalize_fast_config(
        cls,
        fast_config: dict[str, Any] | None,
        *,
        fast_years_back: int | None,
        fast_benchmark_label: str | None,
    ) -> dict[str, Any]:
        """
        Normalize fast-mode config where explicit scalar params win over fast_config keys.
        """
        cfg: dict[str, bool | str | int | Any] = deepcopy(cls._default_fast_config())
        if fast_config:
            cfg.update(dict(fast_config))

        if fast_years_back is not None:
            cfg["years_back"] = int(fast_years_back)
        if fast_benchmark_label is not None:
            cfg["benchmark_label"] = str(fast_benchmark_label)

        cfg["years_back"] = max(0, int(cfg.get("years_back", 3)))
        cfg["benchmark_label"] = str(cfg.get("benchmark_label", "Benchmark"))
        cfg["quick"] = bool(cfg.get("quick", True))
        cfg["mc_enabled"] = bool(cfg.get("mc_enabled", False))
        cfg["mc_n_sims"] = max(1, int(cfg.get("mc_n_sims", 10000)))
        cfg["mc_method"] = str(cfg.get("mc_method", "bootstrap")).lower().strip()
        cfg["mc_block_size"] = max(1, int(cfg.get("mc_block_size", 1)))
        cfg["mc_seed"] = cfg.get("mc_seed", "")

        raw_percentiles = cfg.get("mc_plot_percentiles", [10, 50, 90])
        if not isinstance(raw_percentiles, (list, tuple)):
            raw_percentiles = [10, 50, 90]
        clean_percentiles = sorted(
            {
                int(p)
                for p in raw_percentiles
                if isinstance(p, (int, float)) and 0 <= float(p) <= 100
            }
        )
        cfg["mc_plot_percentiles"] = clean_percentiles or [10, 50, 90]

        return cfg

    def setup(
        self,
        portfolio_classes: list[type[BasePortfolio]],
        start_date: str,
        end_date: str,
        initial_capital: float,
        slippage: float = 0.0,
        cost_model: CostModel | None = None,
        backtest_mode: str = "event",
        fast_config: dict[str, Any] | None = None,
        fast_years_back: int | None = None,
        fast_benchmark_label: str | None = None,
    ):
        """
        Configures the backtest with the necessary parameters.
        """
        self.portfolio_classes = portfolio_classes
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.slippage = slippage
        self.cost_model = cost_model
        self.backtest_mode = str(backtest_mode).lower().strip()
        self.fast_config = self._normalize_fast_config(
            fast_config,
            fast_years_back=fast_years_back,
            fast_benchmark_label=fast_benchmark_label,
        )
        self.logger.info("Backtest engine setup complete.")

    def _build_fast_output_dir(self, portfolio_id: str) -> str:
        run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_root_raw = self.backtest_output_root or os.environ.get(
            "BACKTEST_OUTPUT_DIR"
        )
        if output_root_raw:
            output_root = Path(output_root_raw).expanduser()
        else:
            output_root = Path(__file__).resolve().parent / "data"

        try:
            output_root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(
                f"Backtest output root is not writable: {output_root} ({e})"
            ) from e
        if not os.access(output_root, os.W_OK):
            raise ValueError(f"Backtest output root is not writable: {output_root}")

        out_dir = output_root / f"{run_ts}_backtest_{portfolio_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir)

    @staticmethod
    def _to_utc_timestamp(value: pd.Timestamp | DatetimeScalar) -> pd.Timestamp:
        if isinstance(value, pd.Timestamp):
            ts: pd.Timestamp | DatetimeScalar = value
            if ts.tz is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts

        ts: pd.Timestamp = pd.to_datetime(value, utc=True)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts

    def _fetch_fast_daily_close_data(
        self,
        portfolio_instance: BasePortfolio,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        tickers: list[str] | None = None,
    ) -> pd.DataFrame:
        tickers = (
            tickers
            if tickers is not None
            else getattr(portfolio_instance, "tickers", [])
        )
        if not tickers:
            self.logger.warning("Fast mode skipped: portfolio has no tickers.")
            return pd.DataFrame()

        placeholders = ", ".join(["%s"] * len(tickers))
        sql = f"""
            SELECT DISTINCT ON (ticker, (timestamp AT TIME ZONE 'America/New_York')::date)
                   ticker,
                   timestamp,
                   close_price
            FROM market_data
            WHERE ticker IN ({placeholders})
              AND timestamp BETWEEN %s AND %s
            ORDER BY ticker,
                     (timestamp AT TIME ZONE 'America/New_York')::date,
                     timestamp DESC
        """
        start_param = self._to_utc_timestamp(start_date).to_pydatetime()
        end_param = self._to_utc_timestamp(end_date).to_pydatetime()
        params = tickers + [start_param, end_param]
        result = portfolio_instance.db.execute_query(sql, params, fetch=True)

        if result.get("status") != "success":
            self.logger.warning(
                "Fast mode daily query failed: %s",
                result.get("message", "<no message>"),
            )
            return pd.DataFrame()

        df = pd.DataFrame(result.get("data", []))
        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True, errors="coerce"
        ).dt.tz_convert("America/New_York")
        df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")
        df.dropna(subset=["timestamp", "ticker", "close_price"], inplace=True)
        if df.empty:
            return df

        df.sort_values(["ticker", "timestamp"], inplace=True)
        df["trade_date"] = df["timestamp"].dt.normalize().dt.tz_localize(None)
        return df

    def _run_fast_vectorized(self, portfolio_instance: BasePortfolio):
        fast_cfg = self.fast_config
        self.logger.info(
            "Running fast vectorized backtest for portfolio_id=%s",
            portfolio_instance.portfolio_id,
        )

        ts = [pd.to_datetime(self.start_date), pd.to_datetime(self.end_date)]
        query_start = ts[0] - pd.DateOffset(years=int(fast_cfg["years_back"]))
        query_start_utc = self._to_utc_timestamp(query_start)
        end_ts_utc = self._to_utc_timestamp(ts[1])

        self.logger.info(
            "Fast mode data window: %s to %s (years_back=%s)",
            query_start,
            ts[1],
            fast_cfg["years_back"],
        )

        adapter = get_vector_adapter_for_portfolio(portfolio_instance)
        if adapter is None:
            self.logger.warning(
                "Fast mode skipped: no external vector adapter registered for %s.",
                portfolio_instance.__class__.__name__,
            )
            return

        historical_daily = self._fetch_fast_daily_close_data(
            portfolio_instance=portfolio_instance,
            start_date=query_start_utc,
            end_date=end_ts_utc,
            tickers=getattr(portfolio_instance, "tickers", None),
        )

        if historical_daily.empty:
            self.logger.warning("Fast mode skipped: no historical data returned.")
            return

        self.logger.info(
            "Fast mode fetched daily close bars: %s rows",
            len(historical_daily),
        )

        close_matrix = (
            historical_daily.pivot_table(
                index="trade_date",
                columns="ticker",
                values="close_price",
                aggfunc="last",
            )
            .sort_index()
            .ffill()
            .dropna(how="all")
        )
        if close_matrix.empty:
            self.logger.warning("Fast mode skipped: unable to build ticker close matrix.")
            return

        selected_tickers = [
            ticker for ticker in getattr(portfolio_instance, "tickers", [])
            if ticker in close_matrix.columns
        ]
        if not selected_tickers:
            self.logger.warning("Fast mode skipped: no overlapping ticker prices found.")
            return

        close_matrix = close_matrix[selected_tickers].dropna(how="all")
        if close_matrix.empty:
            self.logger.warning("Fast mode skipped: selected ticker matrix is empty.")
            return

        signal_result = adapter(close_matrix)
        weights = signal_result.target_weights.reindex(
            index=close_matrix.index,
            columns=close_matrix.columns,
        ).fillna(0.0)
        returns_matrix = (
            close_matrix.pct_change()
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
        )

        weights_full = weights.copy(deep=True)
        returns_matrix_full = returns_matrix.copy(deep=True)
        close_matrix_full = close_matrix.copy(deep=True)
        lagged_weights_full = weights_full.shift(1).fillna(0.0)

        gross_returns_full = (lagged_weights_full * returns_matrix_full).sum(axis=1)
        turnover_full = (
            weights_full.diff().abs().sum(axis=1).fillna(weights_full.abs().sum(axis=1))
        )
        transaction_costs_full = turnover_full * float(self.slippage)
        strategy_returns_full = gross_returns_full - transaction_costs_full
        benchmark_returns_full = returns_matrix_full.mean(axis=1)
        benchmark_close_full = close_matrix_full.mean(axis=1)

        weights = weights_full.loc[ts[0]:ts[1]]
        lagged_weights = lagged_weights_full.loc[ts[0]:ts[1]]
        returns_matrix = returns_matrix_full.loc[ts[0]:ts[1]]
        close_matrix = close_matrix_full.loc[ts[0]:ts[1]]
        if (
            weights.empty
            or lagged_weights.empty
            or returns_matrix.empty
            or close_matrix.empty
        ):
            self.logger.warning(
                "Fast mode skipped: no data remains in requested backtest window %s to %s.",
                ts[0], ts[1]
            )
            return

        gross_returns = (lagged_weights * returns_matrix).sum(axis=1)
        turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
        transaction_costs = turnover * float(self.slippage)
        strategy_returns = gross_returns - transaction_costs

        benchmark_returns = returns_matrix.mean(axis=1)
        benchmark_close = close_matrix.mean(axis=1)

        price_data = pd.DataFrame({"close": benchmark_close}, index=close_matrix.index)
        price_data = (
            price_data[~price_data.index.duplicated(keep="last")].ffill().dropna()
        )
        if price_data.empty:
            self.logger.warning(
                "Fast mode skipped: benchmark reference series is empty."
            )
            return

        bt = VectorBacktester(
            data=price_data,
            commission=0.0,
            slippage=0.0,
            initial_capital=int(self.initial_capital),
            store_intermediates=False,
        )

        # Save standard quick outputs plus overlay artifacts in the same run folder.
        out_dir = self._build_fast_output_dir(str(portfolio_instance.portfolio_id))
        quick_results = bt.run_from_returns(
            strategy_returns=strategy_returns,
            market_returns=benchmark_returns,
            close_series=benchmark_close,
        )
        required_cols = {"close", "cum_strategy", "cum_market"}
        missing_cols = required_cols.difference(quick_results.columns)
        if missing_cols:
            raise ValueError(
                "VectorBacktester.run_from_returns returned unexpected columns; "
                f"missing: {sorted(missing_cols)}"
            )

        perf_df = pd.DataFrame(
            {
                "timestamp": quick_results.index,
                "reference_close": quick_results["close"].to_numpy(),
                "portfolio_value": quick_results["cum_strategy"].to_numpy()
                * self.initial_capital,
                "pnl_pct": quick_results["cum_strategy"].to_numpy() - 1.0,
                "turnover": turnover.reindex(quick_results.index).to_numpy(),
            }
        )
        perf_df.to_csv(
            os.path.join(out_dir, "performance_timeseries_absolute.csv"), index=False
        )

        benchmark_df = pd.DataFrame(
            {
                "timestamp": quick_results.index,
                "buy_and_hold_value": quick_results["cum_market"].to_numpy()
                * self.initial_capital,
                "buy_and_hold_return": quick_results["cum_market"].to_numpy() - 1.0,
            }
        )
        benchmark_df.to_csv(
            os.path.join(out_dir, "benchmark_buy_and_hold_performance.csv"),
            index=False,
        )

        pd.DataFrame([bt.metrics]).to_csv(
            os.path.join(out_dir, "summary_metrics.csv"), index=False
        )

        seasonal_price_data = pd.DataFrame(
            {"close": benchmark_close_full}, index=close_matrix_full.index
        )
        seasonal_price_data = (
            seasonal_price_data[~seasonal_price_data.index.duplicated(keep="last")]
            .ffill()
            .dropna()
        )
        if seasonal_price_data.empty:
            self.logger.warning(
                "Fast mode seasonal overlay skipped: full historical benchmark series is empty."
            )
        else:
            seasonal_bt = VectorBacktester(
                data=seasonal_price_data,
                commission=0.0,
                slippage=0.0,
                initial_capital=int(self.initial_capital),
                store_intermediates=False,
            )
            seasonal_bt.run_from_returns(
                strategy_returns=strategy_returns_full,
                market_returns=benchmark_returns_full,
                close_series=benchmark_close_full,
            )

            seasonal_bt.run_and_save_same_window_previous_years(
                output_dir=out_dir,
                start_date=self.start_date,
                end_date=self.end_date,
                years_back=int(fast_cfg["years_back"]),
                quick=True,
                benchmark=benchmark_close_full,
                benchmark_label=str(fast_cfg["benchmark_label"]),
            )

        if bool(fast_cfg["mc_enabled"]):
            sim_cum, sim_stats = bt.monte_carlo(
                n_sims=int(fast_cfg["mc_n_sims"]),
                method=str(fast_cfg["mc_method"]),
                block_size=int(fast_cfg["mc_block_size"]),
                seed=fast_cfg["mc_seed"],
            )

            pd.DataFrame([sim_stats]).to_csv(
                os.path.join(out_dir, "monte_carlo_summary_metrics.csv"), index=False
            )

            percentiles = list(fast_cfg["mc_plot_percentiles"])
            percentile_paths = np.percentile(sim_cum, percentiles, axis=0) - 1.0
            mc_percentile_df = pd.DataFrame({"day": np.arange(sim_cum.shape[1])})
            for idx, pct in enumerate(percentiles):
                mc_percentile_df[f"p{pct}"] = percentile_paths[idx]
            mc_percentile_df.to_csv(
                os.path.join(out_dir, "monte_carlo_percentile_paths.csv"),
                index=False,
            )

            self.logger.info(
                "Fast mode Monte Carlo artifacts saved to %s (n_sims=%s, method=%s)",
                out_dir,
                fast_cfg["mc_n_sims"],
                fast_cfg["mc_method"],
            )

        # --- Section: Risk Analytics (parity with event-mode reporting) ---
        try:
            # Derive portfolio weights from the last row of the signal weights matrix.
            last_weights = weights.iloc[-1]
            portfolio_weights = last_weights[last_weights != 0].to_dict()

            if portfolio_weights:
                # Reshape historical_daily to the format expected by reporting helpers:
                # DataFrame with columns: timestamp, ticker, close_price
                risk_data = historical_daily[
                    ["timestamp", "ticker", "close_price"]
                ].copy()

                corr_matrix, indiv_vols, weights_df = (
                    _calculate_portfolio_risk_components(risk_data, portfolio_weights)
                )
                if not corr_matrix.empty:
                    aligned_weights_df = weights_df[
                        weights_df["ticker"].isin(corr_matrix.columns)
                    ]
                    risk_components_summary = pd.concat(
                        [
                            aligned_weights_df.set_index("ticker"),
                            indiv_vols.rename("annualized_volatility"),
                        ],
                        axis=1,
                    )
                    risk_components_summary.to_csv(
                        os.path.join(out_dir, "portfolio_risk_components.csv")
                    )
                    corr_matrix.to_csv(
                        os.path.join(out_dir, "annualized_correlation_matrix.csv")
                    )
                    self.logger.info(
                        "Saved portfolio_risk_components.csv and annualized_correlation_matrix.csv"
                    )

                rolling_risk_df = _calculate_rolling_portfolio_risk(
                    risk_data, portfolio_weights
                )
                if not rolling_risk_df.empty:
                    rolling_risk_df.to_csv(
                        os.path.join(out_dir, "rolling_portfolio_risk.csv"), index=False
                    )
                    self.logger.info("Saved rolling_portfolio_risk.csv")
            else:
                self.logger.warning(
                    "Fast mode risk analytics skipped: all portfolio weights are zero."
                )
        except Exception as e:
            self.logger.error("Error in fast mode risk analytics: %s", e, exc_info=True)

        # --- Section: Portfolio Composition Timeseries (daily approximation) ---
        # Fast mode has no trade log, so we approximate the composition from the
        # daily weight allocations and cumulative portfolio value.
        try:
            portfolio_value_series = (
                quick_results["cum_strategy"] * self.initial_capital
            )
            composition_df = pd.DataFrame({"timestamp": weights.index})
            for ticker in selected_tickers:
                ticker_weight = (
                    lagged_weights[ticker].reindex(weights.index).fillna(0.0)
                )
                composition_df[f"{ticker}_value"] = (
                    ticker_weight.values
                    * portfolio_value_series.reindex(weights.index).values
                )
            total_allocated = composition_df[
                [c for c in composition_df.columns if c.endswith("_value")]
            ].sum(axis=1)
            composition_df["cash_value"] = (
                portfolio_value_series.reindex(weights.index).values
                - total_allocated.values
            )
            composition_df["portfolio_value"] = portfolio_value_series.reindex(
                weights.index
            ).values
            composition_df.to_csv(
                os.path.join(out_dir, "portfolio_composition_daily.csv"),
                index=False,
            )
            self.logger.info("Saved portfolio_composition_daily.csv")
        except Exception as e:
            self.logger.error(
                "Error generating composition timeseries: %s", e, exc_info=True
            )

        self.logger.info("Fast vectorized artifacts saved to %s", out_dir)

    def _build_fast_portfolio_stub(self, portfolio_class, config_data: dict[str, Any]):
        """Create a lightweight portfolio-like object for fast mode.

        This avoids strategy __init__ side effects (indicator registration/warmup)
        while preserving the fields required by the fast vectorized path.
        """
        raw_portfolio_id = config_data.get("PORTFOLIO_ID")
        portfolio_id = (
            str(raw_portfolio_id).strip() if raw_portfolio_id is not None else ""
        )
        if not portfolio_id:
            raise ValueError(
                f"Invalid PORTFOLIO_ID for {portfolio_class.__name__}: {raw_portfolio_id!r}"
            )

        raw_tickers = config_data.get("TICKERS")
        if not isinstance(raw_tickers, list) or not raw_tickers:
            raise ValueError(
                f"Invalid TICKERS for {portfolio_class.__name__}: expected non-empty list."
            )
        tickers = [str(t).strip() for t in raw_tickers if str(t).strip()]
        if not tickers:
            raise ValueError(
                f"Invalid TICKERS for {portfolio_class.__name__}: all values are empty."
            )

        stub_cls = type(
            str(portfolio_class.__name__),
            (BasePortfolio,),
            {"OnData": lambda self, context: None},
        )
        stub = object.__new__(stub_cls)
        stub.logger = logging.getLogger(
            f"{portfolio_class.__name__}_{portfolio_id}_fast_stub"
        )
        stub.executor = None
        stub.running = True
        stub.debug = False
        stub.backtest_start_date = None
        stub.db = self.db_connector
        stub.portfolio_id = int(portfolio_id)
        stub.tickers = tickers
        stub.portfolio_config_dict = {
            "id": portfolio_id,
            "tickers": tickers,
            "weights": None,
            "poll_interval": None,
            "lookback_days": None,
        }
        return stub

    def run(self) -> list[Any]:
        """
        Initializes and runs the backtest for each portfolio.
        """
        trade_logs = []
        if not self.portfolio_classes:
            self.logger.error("No portfolio classes provided to run backtests.")
            return trade_logs

        for portfolio_class in self.portfolio_classes:
            try:
                # --- Dynamically load the config for the portfolio ---
                # Get the file path of the portfolio's strategy class
                class_file_path = inspect.getfile(portfolio_class)
                portfolio_dir = os.path.dirname(class_file_path)
                config_path = os.path.join(portfolio_dir, "config.json")

                if not os.path.exists(config_path):
                    self.logger.error(
                        "Configuration file not found for %s at %s",
                        portfolio_class.__name__,
                        config_path
                    )
                    continue

                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)

                if self.backtest_mode == "fast":
                    portfolio_instance = self._build_fast_portfolio_stub(
                        portfolio_class,
                        config_data,
                    )
                    self.logger.info(
                        "--- Running backtest for portfolio: %s ---",
                        portfolio_instance.portfolio_id
                    )
                    self._run_fast_vectorized(portfolio_instance)
                else:
                    # --- Instantiate with the loaded config_dict ---
                    portfolio_instance = portfolio_class(
                        db_connector=self.db_connector,
                        executor=None,  # The runner will set the executor later
                        config_dict=config_data,
                        backtest_start_date=pd.to_datetime(self.start_date),
                    )
                    self.logger.info(
                        "\n--- Running backtest for portfolio: %s ---",
                        portfolio_instance.portfolio_id
                    )
                    runner = BacktestRunner(
                        portfolio=portfolio_instance,
                        start_date=self.start_date,
                        end_date=self.end_date,
                        initial_capital=self.initial_capital,
                        slippage=self.slippage,
                        cost_model=self.cost_model,
                    )
                    trade_log = runner.run()
                    trade_logs.append(trade_log)
                self.logger.info(
                    "\n--- Backtest for portfolio: %s finished ---",
                    portfolio_instance.portfolio_id
                )
            except Exception as e:
                self.logger.exception(
                    "Error running backtest for %s: %s",
                    portfolio_class.__name__, e,
                    exc_info=True,
                )

        return trade_logs
