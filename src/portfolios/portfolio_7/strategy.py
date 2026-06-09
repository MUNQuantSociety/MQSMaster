"""
Portfolio 7 = Portfolio 6 + cross-sectional sentiment z-score tilt.

Inherits the full P6 screen + inverse-vol + vol-target + hedge pipeline and
overlays a post-screen exponential tilt on the stock sleeve only:

    w_tilt_i = w_p6_i * exp(LAMBDA * z_i)

where z_i is the cross-sectional standardized 21d EWM-mean sentiment of
ticker i computed strictly from articles with published_at < context.time.

Strict ex-ante (no look-ahead): every SQL filter uses '<' on published_at
against context.time, never '<=', and never the current bar's date.

Configurable knobs (PORTFOLIO_7_CONFIG):
  SENTIMENT_TILT_LAMBDA             float, default 0.25
  SENTIMENT_WINDOW_DAYS             int,   default 21 (trading days)
  SENTIMENT_EWM_HALFLIFE_DAYS       float, default 5.0
  MIN_ARTICLES_PER_TICKER           int,   default 3
  SENTIMENT_AGG_METHOD              str,   "ewm" | "mean", default "ewm"
  SENTIMENT_Z_CLIP                  float, default 3.0
  SENTIMENT_FALLBACK_TO_MARKET_DATA bool,  default False (per D3 audit)
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from portfolios.portfolio_6.strategy import Portfolio6Strategy
    from portfolios.strategy_api import StrategyContext
except ImportError:
    from src.portfolios.portfolio_6.strategy import Portfolio6Strategy
    from src.portfolios.strategy_api import StrategyContext


class Portfolio7Strategy(Portfolio6Strategy):
    """Portfolio 6 with a cross-sectional sentiment z-score tilt overlay."""

    _SENTIMENT_QUERY = """
        SELECT ticker, published_at, sentiment_score
        FROM news_sentiment
        WHERE ticker = ANY(%s::text[])
          AND published_at >= %s
          AND published_at <  %s
          AND sentiment_score IS NOT NULL
    """

    _MARKET_DATA_SENTIMENT_QUERY = """
        SELECT ticker, date,
               COALESCE(avg_sentiment, sentiment_score)::float8 AS s
        FROM market_data
        WHERE ticker = ANY(%s::text[])
          AND date >= %s
          AND date <  %s
          AND COALESCE(avg_sentiment, sentiment_score) IS NOT NULL
    """

    def __init__(
        self,
        db_connector,
        executor,
        debug=False,
        config_dict=None,
        backtest_start_date=None,
    ):
        if config_dict is None:
            raise ValueError("config_dict is required for Portfolio7Strategy.")

        cfg = dict(config_dict)
        p7_cfg = dict(cfg.get("PORTFOLIO_7_CONFIG", {}))

        super().__init__(
            db_connector=db_connector,
            executor=executor,
            debug=debug,
            config_dict=cfg,
            backtest_start_date=backtest_start_date,
        )

        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")

        self.tilt_lambda: float = float(p7_cfg.get("SENTIMENT_TILT_LAMBDA", 0.25))
        self.sent_window_days: int = int(p7_cfg.get("SENTIMENT_WINDOW_DAYS", 21))
        self.sent_ewm_halflife: float = float(
            p7_cfg.get("SENTIMENT_EWM_HALFLIFE_DAYS", 5.0)
        )
        self.min_articles: int = int(p7_cfg.get("MIN_ARTICLES_PER_TICKER", 3))
        self.sent_agg_method: str = str(
            p7_cfg.get("SENTIMENT_AGG_METHOD", "ewm")
        ).strip().lower()
        if self.sent_agg_method not in ("ewm", "mean"):
            self.logger.warning(
                "[P7] Unknown SENTIMENT_AGG_METHOD=%s; falling back to 'ewm'.",
                self.sent_agg_method,
            )
            self.sent_agg_method = "ewm"
        self.z_clip: float = float(p7_cfg.get("SENTIMENT_Z_CLIP", 3.0))
        self.fallback_to_md: bool = bool(
            p7_cfg.get("SENTIMENT_FALLBACK_TO_MARKET_DATA", False)
        )

        self.logger.info(
            "[P7] sentiment tilt cfg: lambda=%.3f, window=%dd, halflife=%.1fd, "
            "min_articles=%d, agg=%s, z_clip=%.2f, fallback_md=%s",
            self.tilt_lambda,
            self.sent_window_days,
            self.sent_ewm_halflife,
            self.min_articles,
            self.sent_agg_method,
            self.z_clip,
            self.fallback_to_md,
        )

    def _fetch_sentiment_ex_ante(
        self,
        tickers: Iterable[str],
        cutoff_ts: pd.Timestamp,
    ) -> Dict[str, pd.Series]:
        tickers = [t for t in tickers if isinstance(t, str) and t]
        if not tickers or cutoff_ts is None:
            return {}

        start_ts = cutoff_ts - pd.Timedelta(days=max(self.sent_window_days * 2, 31))

        result = self.db.execute_query(
            self._SENTIMENT_QUERY,
            (list(tickers), start_ts.to_pydatetime(), cutoff_ts.to_pydatetime()),
            fetch=True,
        )

        out: Dict[str, pd.Series] = {}
        if result.get("status") == "success" and result.get("data"):
            df = pd.DataFrame(result["data"])
            df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
            df["sentiment_score"] = pd.to_numeric(
                df["sentiment_score"], errors="coerce"
            ).clip(-1.0, 1.0)
            df = df.dropna(subset=["published_at", "sentiment_score", "ticker"])
            for ticker, group in df.groupby("ticker"):
                series = group.set_index("published_at")["sentiment_score"].sort_index()
                if not series.empty:
                    out[str(ticker)] = series

        if self.fallback_to_md:
            short = [
                t for t in tickers
                if t not in out or len(out[t]) < self.min_articles
            ]
            if short:
                md = self._fetch_market_data_sentiment(short, cutoff_ts)
                for t, s in md.items():
                    if t not in out or len(s) >= self.min_articles:
                        out[t] = s
        return out

    def _fetch_market_data_sentiment(
        self, tickers: List[str], cutoff_ts: pd.Timestamp
    ) -> Dict[str, pd.Series]:
        if not tickers:
            return {}
        cutoff_date = cutoff_ts.tz_convert("UTC").date() if cutoff_ts.tzinfo else cutoff_ts.date()
        start_date = cutoff_date - timedelta(days=max(self.sent_window_days * 2, 31))

        result = self.db.execute_query(
            self._MARKET_DATA_SENTIMENT_QUERY,
            (list(tickers), start_date, cutoff_date),
            fetch=True,
        )
        out: Dict[str, pd.Series] = {}
        if result.get("status") == "success" and result.get("data"):
            df = pd.DataFrame(result["data"])
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["s"] = pd.to_numeric(df["s"], errors="coerce").clip(-1.0, 1.0)
            df = df.dropna(subset=["date", "s", "ticker"])
            for ticker, group in df.groupby("ticker"):
                series = group.set_index("date")["s"].sort_index()
                if not series.empty:
                    out[str(ticker)] = series
        return out

    def _aggregate_sentiment(
        self, series_by_ticker: Dict[str, pd.Series]
    ) -> Tuple[pd.Series, pd.Series]:
        agg_vals: Dict[str, float] = {}
        n_articles: Dict[str, int] = {}
        for ticker, series in series_by_ticker.items():
            n = len(series)
            n_articles[ticker] = n
            if n < self.min_articles:
                agg_vals[ticker] = float(series.mean()) if n > 0 else np.nan
                continue
            if self.sent_agg_method == "ewm":
                ewm = series.ewm(
                    halflife=pd.Timedelta(days=self.sent_ewm_halflife),
                    times=series.index,
                    adjust=True,
                ).mean()
                agg_vals[ticker] = float(ewm.iloc[-1])
            else:
                agg_vals[ticker] = float(series.mean())
        return pd.Series(agg_vals, dtype=float), pd.Series(n_articles, dtype=int)

    def _cross_sectional_z(
        self,
        sentiment: pd.Series,
        n_articles: pd.Series,
    ) -> pd.Series:
        eligible_mask = n_articles >= self.min_articles
        eligible = sentiment[eligible_mask].dropna()
        if eligible.empty or eligible.std(ddof=0) <= 1e-12:
            return pd.Series(0.0, index=sentiment.index, dtype=float)

        mu = float(eligible.mean())
        sd = float(eligible.std(ddof=0))
        z = (sentiment - mu) / sd
        z = z.where(eligible_mask, 0.0)
        z = z.clip(lower=-self.z_clip, upper=self.z_clip)
        return z.fillna(0.0)

    def _apply_tilt(
        self,
        stock_weights: Dict[str, float],
        z_scores: pd.Series,
    ) -> Dict[str, float]:
        if not stock_weights:
            return {}
        if self.tilt_lambda == 0.0:
            return dict(stock_weights)

        original_gross = float(sum(stock_weights.values()))
        if original_gross <= 0.0:
            return dict(stock_weights)

        tilted: Dict[str, float] = {}
        for ticker, w in stock_weights.items():
            z_val = z_scores.get(ticker, 0.0)
            try:
                z = float(z_val) if pd.notna(z_val) else 0.0
            except (TypeError, ValueError):
                z = 0.0
            tilted[ticker] = max(float(w), 0.0) * float(np.exp(self.tilt_lambda * z))

        tilted_gross = sum(tilted.values())
        if tilted_gross <= 0.0:
            return dict(stock_weights)

        scale = original_gross / tilted_gross
        tilted = {t: w * scale for t, w in tilted.items()}

        s = pd.Series(tilted, dtype=float)
        for _ in range(20):
            over = s > self.max_weight
            if not over.any():
                break
            slack = float((s[over] - self.max_weight).sum())
            s[over] = self.max_weight
            under = ~over
            under_sum = float(s[under].sum())
            if under_sum <= 0:
                break
            s[under] = s[under] + slack * (s[under] / under_sum)
        return s.to_dict()

    def _rebalance(self, context: StrategyContext):
        try:
            from portfolios.portfolio_6.screener import (
                deflated_sharpe_ratio,
                score_universe,
                select_top_n,
                vol_target_scale,
            )
        except ImportError:
            from src.portfolios.portfolio_6.screener import (
                deflated_sharpe_ratio,
                score_universe,
                select_top_n,
                vol_target_scale,
            )

        returns_matrix = self._collect_returns(context)
        if not returns_matrix:
            self.logger.warning(
                "[P7] No tickers have sufficient history; skipping rebalance."
            )
            return

        scores = score_universe(
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
        top = select_top_n(scores, n=self.screen_top_n)
        if not top:
            self.logger.warning("[P7] Top-N selection empty; skipping rebalance.")
            return

        # Use the parent's weighting dispatcher so HRP/INV_VOL/ERC choice is inherited.
        try:
            from src.portfolios.portfolio_6.strategy import _WEIGHTING_DISPATCH
            from src.portfolios.portfolio_6.screener import inverse_vol_weights
        except ImportError:
            from portfolios.portfolio_6.strategy import _WEIGHTING_DISPATCH
            from portfolios.portfolio_6.screener import inverse_vol_weights

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
                "[P7] %s weighting failed (%s); falling back to inverse_vol.",
                self.weighting_method, e,
            )
            weights = inverse_vol_weights({t: returns_matrix[t] for t in top}, max_weight=self.max_weight)

        if not weights:
            self.logger.warning("[P7] Weighting empty; skipping rebalance.")
            return

        # Sentiment tilt (the P7-specific step)
        cutoff_ts = pd.Timestamp(context.time)
        if cutoff_ts.tzinfo is None:
            cutoff_ts = cutoff_ts.tz_localize("UTC")
        else:
            cutoff_ts = cutoff_ts.tz_convert("UTC")

        sentiment_by_ticker = self._fetch_sentiment_ex_ante(
            tickers=list(weights.keys()),
            cutoff_ts=cutoff_ts,
        )
        sent_scalar, n_articles = self._aggregate_sentiment(sentiment_by_ticker)
        z = self._cross_sectional_z(sent_scalar, n_articles)

        pre_tilt_top5 = sorted(weights.items(), key=lambda kv: -kv[1])[:5]
        weights = self._apply_tilt(weights, z)
        post_tilt_top5 = sorted(weights.items(), key=lambda kv: -kv[1])[:5]
        try:
            n_supported = int((n_articles >= self.min_articles).sum())
            self.logger.info(
                "[P7] sentiment tilt applied: lambda=%.2f, supported=%d/%d, "
                "z_min=%.2f z_max=%.2f, pre_top5=%s post_top5=%s",
                self.tilt_lambda,
                n_supported,
                len(weights),
                float(z.min()) if len(z) else 0.0,
                float(z.max()) if len(z) else 0.0,
                pre_tilt_top5,
                post_tilt_top5,
            )
        except Exception:
            pass

        returns_df = pd.DataFrame({t: returns_matrix[t] for t in top}).dropna(how="all")
        weight_series = pd.Series(weights).reindex(returns_df.columns).fillna(0.0)
        sleeve_returns = (returns_df * weight_series).sum(axis=1)

        try:
            dsr = deflated_sharpe_ratio(
                sleeve_returns,
                n_trials=len(returns_matrix) + 1,
            )
            self.logger.info(
                "[P7] Deflated Sharpe probability=%.3f (n_trials=%s, top_n=%s)",
                dsr,
                len(returns_matrix) + 1,
                len(top),
            )
            if dsr < self.dsr_min_prob:
                self.logger.warning(
                    "[P7] DSR=%.3f below threshold %.2f; selection+tilt may be noise.",
                    dsr,
                    self.dsr_min_prob,
                )
        except Exception as e:
            self.logger.exception("[P7] DSR computation failed: %s", e)

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
                "[P7] Leverage cap applied: total %.3f scaled to %.3f.",
                total,
                self.max_leverage,
            )

        self._target_weights = target_weights
        ranked = sorted(target_weights.items(), key=lambda kv: -kv[1])
        self.logger.info(
            "[P7] New targets: positions=%s, total_weight=%.3f, top5=%s",
            len(target_weights),
            sum(target_weights.values()),
            ranked[:5],
        )
