"""
Portfolio 8 = Portfolio 6 + RBP-rank overlay.

Pipeline:
  1. Inherit Portfolio_6's universe load, returns-matrix collection, DSR
     check, inverse-vol weights, vol-target scaling, and hedge sleeves.
  2. Before P6's top-N selection, refresh RBP 21-day forward-return
     forecasts for the candidate set via RBP.pipeline.RBPPipeline.
  3. Add an extra rank term (rbp_rank descending) to the composite
     score so high-conviction RBP names move up the ranking.

Production cadence: monthly, at the same tick as P6's _rebalance.
Failure of the RBP refresh degrades gracefully to a pure P6 screen.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

try:
    from portfolios.portfolio_6.strategy import Portfolio6Strategy
    from portfolios.portfolio_6.screener import (
        inverse_vol_weights,
        score_universe,
        select_top_n,
        vol_target_scale,
        deflated_sharpe_ratio,
    )
    from portfolios.strategy_api import StrategyContext
except ImportError:
    from src.portfolios.portfolio_6.strategy import Portfolio6Strategy
    from src.portfolios.portfolio_6.screener import (
        inverse_vol_weights,
        score_universe,
        select_top_n,
        vol_target_scale,
        deflated_sharpe_ratio,
    )
    from src.portfolios.strategy_api import StrategyContext


class Portfolio8Strategy(Portfolio6Strategy):
    """Portfolio 6 with an additional RBP-rank signal in the composite score."""

    def __init__(
        self,
        db_connector,
        executor,
        debug=False,
        config_dict=None,
        backtest_start_date=None,
    ):
        if config_dict is None:
            raise ValueError("config_dict is required for Portfolio8Strategy.")

        super().__init__(
            db_connector=db_connector,
            executor=executor,
            debug=debug,
            config_dict=config_dict,
            backtest_start_date=backtest_start_date,
        )
        self.logger = logging.getLogger(
            f"{self.__class__.__name__}_{self.portfolio_id}"
        )

        rbp_cfg = dict(config_dict.get("RBP_BLEND", {}))
        self.rbp_enabled: bool = bool(rbp_cfg.get("ENABLED", True))
        self.rbp_weight: float = float(rbp_cfg.get("WEIGHT", 1.0))
        self.score_weights_p8: Dict[str, float] = {
            "vol": float(rbp_cfg.get("W_VOL", 1.0)),
            "max": float(rbp_cfg.get("W_MAX", 1.0)),
            "q":   float(rbp_cfg.get("W_QUALITY", 1.0)),
            "rbp": self.rbp_weight,
        }
        self.rbp_lookback_days: int = int(rbp_cfg.get("LOOKBACK_DAYS", 365 * 5))
        self.rbp_split_date: str = str(rbp_cfg.get("SPLIT_DATE", "2023-01-01"))
        self.rbp_max_combo: int = int(rbp_cfg.get("MAX_COMBINATION_SIZE", 1))
        self.rbp_censoring: list = list(
            rbp_cfg.get("CENSORING_QUANTILES", [0.0, 0.2, 0.5, 0.8])
        )
        self.rbp_n_jobs: int = int(rbp_cfg.get("N_JOBS", -1))
        self.rbp_max_universe: int = int(rbp_cfg.get("MAX_UNIVERSE", 150))
        self._last_rbp_forecasts: Dict[str, float] = {}

        self.logger.info(
            "Portfolio8Strategy init: rbp_enabled=%s, rbp_weight=%.2f, "
            "score_weights=%s, max_universe=%d",
            self.rbp_enabled,
            self.rbp_weight,
            self.score_weights_p8,
            self.rbp_max_universe,
        )

    def _refresh_rbp_forecasts(self, candidate_tickers: list) -> Dict[str, float]:
        if not self.rbp_enabled or not candidate_tickers:
            return {}

        try:
            try:
                from RBP.config import RBPConfig
                from RBP.pipeline import RBPPipeline
            except ImportError:
                from src.RBP.config import RBPConfig  # type: ignore
                from src.RBP.pipeline import RBPPipeline  # type: ignore
        except Exception as exc:
            self.logger.warning("RBP package not importable; skipping RBP step (%s)", exc)
            return {}

        capped_tickers = list(candidate_tickers)[: self.rbp_max_universe]
        config = RBPConfig(
            tickers=capped_tickers,
            lookback_days=self.rbp_lookback_days,
            train_test_split_date=self.rbp_split_date,
            max_combination_size=self.rbp_max_combo,
            censoring_quantiles=self.rbp_censoring,
            n_jobs=self.rbp_n_jobs,
            max_test_tasks=None,
        )

        try:
            pipeline = RBPPipeline(config=config)
            predictions_df, rbi_df = pipeline.run()
        except Exception as exc:
            self.logger.exception("RBP pipeline failed; degrading to pure P6: %s", exc)
            return {}

        if predictions_df is None or predictions_df.empty:
            self.logger.warning("RBP returned no predictions; degrading to pure P6.")
            return {}

        if "ticker" in predictions_df.columns:
            latest_per_ticker = (
                predictions_df.dropna(subset=["y_pred_rbp"])
                .groupby("ticker")["y_pred_rbp"]
                .mean()
            )
            forecasts = latest_per_ticker.to_dict()
        else:
            self.logger.warning(
                "RBPPipeline.run() output lacks 'ticker' column; using "
                "task-averaged predictions across the full test window."
            )
            forecasts = {
                t: float(predictions_df["y_pred_rbp"].mean())
                for t in capped_tickers
            }

        if rbi_df is not None and not rbi_df.empty:
            try:
                rbi_mean = rbi_df.mean(numeric_only=True).sort_values(ascending=False)
                self.logger.info(
                    "[P8] RBI top-5 features (mean adj_fit gap): %s",
                    rbi_mean.head(5).to_dict(),
                )
            except Exception as exc:
                self.logger.debug("RBI summary failed (non-fatal): %s", exc)

        forecast_values = [v for v in forecasts.values() if pd.notna(v)]
        if forecast_values:
            self.logger.info(
                "[P8] RBP refreshed: %d tickers (median=%.4f, p10=%.4f, p90=%.4f).",
                len(forecasts),
                float(np.nanmedian(forecast_values)),
                float(np.nanpercentile(forecast_values, 10)),
                float(np.nanpercentile(forecast_values, 90)),
            )
        return {str(k): float(v) for k, v in forecasts.items() if pd.notna(v)}

    def _compose_score(
        self,
        returns_matrix: Dict[str, pd.Series],
        rbp_forecasts: Dict[str, float],
    ) -> pd.Series:
        """P6's score + an RBP-rank term. Lower score = better."""
        base = score_universe(
            returns_matrix,
            self.fundamentals_df,
            method=self.score_method,
            weights=self.score_weights,
            winsor_sigma=self.score_winsor_sigma,
            exclusions_cfg=self.exclusions_cfg,
            use_fundamentals=self.use_fundamentals,
            use_momentum=self.use_momentum_12_2,
            use_op_profitability=self.use_op_profitability,
            use_asset_growth=self.use_asset_growth,
            momentum_lookback_days=self.momentum_lookback_days,
            momentum_skip_days=self.momentum_skip_days,
            logger=self.logger,
        )
        if base.empty:
            return base

        if not rbp_forecasts:
            self.logger.info("[P8] No RBP forecasts available; using pure P6 score.")
            return base

        rbp_series = pd.Series(
            {t: rbp_forecasts.get(t, np.nan) for t in base.index}
        ).replace([np.inf, -np.inf], np.nan)
        valid = rbp_series.dropna()
        if valid.empty:
            self.logger.warning("[P8] RBP forecasts all NaN after alignment; using P6 score.")
            return base

        rbp_rank_desc = valid.rank(ascending=False)
        rbp_rank_full = pd.Series(
            data=base.median(), index=base.index, dtype=float
        )
        rbp_rank_full.loc[rbp_rank_desc.index] = rbp_rank_desc.values

        composite = (
            self.score_weights_p8["vol"] * base
            + self.score_weights_p8["rbp"] * rbp_rank_full
        )
        return composite.sort_values(ascending=True)

    def _rebalance(self, context: StrategyContext):
        returns_matrix = self._collect_returns(context)
        if not returns_matrix:
            self.logger.warning("[P8] No tickers have sufficient history; skipping rebalance.")
            return

        rbp_forecasts = self._refresh_rbp_forecasts(list(returns_matrix.keys()))
        self._last_rbp_forecasts = rbp_forecasts

        scores = self._compose_score(returns_matrix, rbp_forecasts)
        top = select_top_n(scores, n=self.screen_top_n)
        if not top:
            self.logger.warning("[P8] Top-N selection empty; skipping rebalance.")
            return

        # Use parent's weighting dispatcher (B1 routing).
        try:
            from src.portfolios.portfolio_6.strategy import _WEIGHTING_DISPATCH
        except ImportError:
            from portfolios.portfolio_6.strategy import _WEIGHTING_DISPATCH

        weighter = _WEIGHTING_DISPATCH[self.weighting_method]
        weight_kwargs = {"max_weight": self.max_weight}
        if self.weighting_method == "HRP":
            weight_kwargs["linkage_method"] = self.hrp_linkage_method
        sigma = getattr(self, "_shrinkage_cov", None)
        if sigma is not None and self.weighting_method in ("HRP", "ERC"):
            weight_kwargs["cov"] = sigma
        try:
            weights = weighter(
                {t: returns_matrix[t] for t in top},
                **weight_kwargs,
            )
        except Exception as e:
            self.logger.exception(
                "[P8] %s weighting failed (%s); falling back to inverse_vol.",
                self.weighting_method, e,
            )
            weights = inverse_vol_weights({t: returns_matrix[t] for t in top}, max_weight=self.max_weight)

        if not weights:
            self.logger.warning("[P8] Weighting empty; skipping rebalance.")
            return

        returns_df = pd.DataFrame({t: returns_matrix[t] for t in top}).dropna(how="all")
        weight_series = pd.Series(weights).reindex(returns_df.columns).fillna(0.0)
        sleeve_returns = (returns_df * weight_series).sum(axis=1)

        try:
            dsr = deflated_sharpe_ratio(sleeve_returns, n_trials=len(returns_matrix))
            self.logger.info(
                "[P8] DSR=%.3f (n_trials=%s, top_n=%s)",
                dsr, len(returns_matrix), len(top),
            )
            if dsr < self.dsr_min_prob:
                self.logger.warning(
                    "[P8] DSR=%.3f below threshold %.2f; selection may be noise.",
                    dsr, self.dsr_min_prob,
                )
        except Exception as exc:
            self.logger.exception("[P8] DSR computation failed: %s", exc)

        vol_scale = vol_target_scale(
            sleeve_returns,
            target_annual_vol=self.vol_target_annual,
            max_scale=self.max_leverage,
        )
        weights = {t: w * vol_scale for t, w in weights.items()}

        target_weights = dict(weights)
        if self.gld_ticker and self.gld_ticker in self.tickers:
            gld_asset = context.Market[self.gld_ticker]
            if gld_asset.Exists:
                target_weights[self.gld_ticker] = self.gld_weight
        if self.trend_ticker and self.trend_ticker in self.tickers:
            trend_asset = context.Market[self.trend_ticker]
            if trend_asset.Exists:
                target_weights[self.trend_ticker] = self.trend_weight

        total = sum(target_weights.values())
        if total > self.max_leverage > 0:
            scale = self.max_leverage / total
            target_weights = {t: w * scale for t, w in target_weights.items()}
            self.logger.info(
                "[P8] Leverage cap applied: total %.3f scaled to %.3f.",
                total, self.max_leverage,
            )

        self._target_weights = target_weights
        ranked = sorted(target_weights.items(), key=lambda kv: -kv[1])
        self.logger.info(
            "[P8] New targets: positions=%s, total_weight=%.3f, top5=%s",
            len(target_weights), sum(target_weights.values()), ranked[:5],
        )
