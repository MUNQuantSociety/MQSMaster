"""
Portfolio 6: Boring + Not-Lottery screen with inverse-volatility weighting,
GLD + optional trend-following hedge, and portfolio-level volatility targeting.

v1 baseline. Pipeline:
  1. Universe = S&P 500 + Nasdaq-100 (from src/portfolios/portfolio_6/universe.json).
  2. Monthly screen ranks candidates by (low realized vol) + (low max 1-day return)
     and optional (high gross_profit / total_assets) from local fundamentals CSV.
  3. Top SCREEN_TOP_N stocks weighted by 1/vol with per-stock cap MAX_WEIGHT_PER_STOCK.
     Optional WEIGHTING_METHOD={INV_VOL,HRP,ERC} dispatch via Team B B1.
  4. Stock-sleeve weights scaled so realized portfolio vol ~= VOL_TARGET_ANNUAL,
     bounded by MAX_LEVERAGE.
  5. Hedge sleeve adds GLD_WEIGHT in GLD_TICKER and TREND_HEDGE_WEIGHT in
     TREND_HEDGE_TICKER (config; empty disables that sleeve).
  6. Deflated Sharpe Ratio is logged as a diagnostic of multiple-testing bias.
  7. OnData runs the screen on the first tick of each trading day and then
     issues rebalance orders toward the stored target weights. The active
     ticker+weight set is written to current_holdings.json beside config.json
     so the live shortlist is visible without reading process memory.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

try:
    from portfolios.portfolio_BASE.strategy import BasePortfolio
    from portfolios.strategy_api import StrategyContext
except ImportError:
    from src.portfolios.portfolio_BASE.strategy import BasePortfolio
    from src.portfolios.strategy_api import StrategyContext

try:
    from src.portfolios.portfolio_6.screener import (
        deflated_sharpe_ratio,
        inverse_vol_weights,
        score_universe,
        select_top_n,
        vol_target_scale,
    )
except ImportError:
    from portfolios.portfolio_6.screener import (
        deflated_sharpe_ratio,
        inverse_vol_weights,
        score_universe,
        select_top_n,
        vol_target_scale,
    )

# Team B / B1 -- HRP and ERC opt-in weighting methods.
try:
    from src.portfolios.portfolio_6.hrp_weights import erc_weights, hrp_weights
except ImportError:
    from portfolios.portfolio_6.hrp_weights import erc_weights, hrp_weights

_WEIGHTING_DISPATCH = {
    "INV_VOL": inverse_vol_weights,
    "HRP":     hrp_weights,
    "ERC":     erc_weights,
}


REPO_ROOT = Path(__file__).resolve().parents[3]


class Portfolio6Strategy(BasePortfolio):
    def __init__(
        self,
        db_connector,
        executor,
        debug=False,
        config_dict=None,
        backtest_start_date=None,
    ):
        if config_dict is None:
            raise ValueError("config_dict is required for Portfolio6Strategy.")

        cfg = dict(config_dict)
        p6_cfg = dict(cfg.get("PORTFOLIO_6_CONFIG", {}))

        universe = self._load_universe(p6_cfg)
        hedge_tickers = [
            t for t in (
                str(p6_cfg.get("GLD_TICKER", "GLD")).strip(),
                str(p6_cfg.get("TREND_HEDGE_TICKER", "")).strip(),
            ) if t
        ]
        merged_tickers = sorted({*universe, *hedge_tickers})
        if merged_tickers:
            cfg["TICKERS"] = merged_tickers

        super().__init__(db_connector, executor, debug, cfg, backtest_start_date)

        self.logger = logging.getLogger(
            f"{self.__class__.__name__}_{self.portfolio_id}"
        )

        self.screen_top_n: int = int(p6_cfg.get("SCREEN_TOP_N", 50))
        self.vol_lookback_days: int = int(p6_cfg.get("VOL_LOOKBACK_DAYS", 252))
        self.max_weight: float = float(p6_cfg.get("MAX_WEIGHT_PER_STOCK", 0.05))
        self.vol_target_annual: float = float(p6_cfg.get("VOL_TARGET_ANNUAL", 0.13))
        self.max_leverage: float = float(p6_cfg.get("MAX_LEVERAGE", 1.5))
        self.gld_ticker: str = str(p6_cfg.get("GLD_TICKER", "GLD")).strip()
        self.gld_weight: float = float(p6_cfg.get("GLD_WEIGHT", 0.07))
        self.trend_ticker: str = str(p6_cfg.get("TREND_HEDGE_TICKER", "")).strip()
        self.trend_weight: float = float(p6_cfg.get("TREND_HEDGE_WEIGHT", 0.10))
        self.rebalance_drift: float = float(
            p6_cfg.get("REBALANCE_DRIFT_THRESHOLD", 0.005)
        )
        self.use_fundamentals: bool = bool(p6_cfg.get("USE_FUNDAMENTALS", True))
        self.dsr_min_prob: float = float(p6_cfg.get("DSR_MIN_PROB", 0.5))
        self.fundamentals_csv_rel: str = str(
            p6_cfg.get("FUNDAMENTALS_CSV", "fundamentals/fundamentals.csv")
        )

        # Team A SYNTHESIS §7 -- scoring + factor flags + exclusions
        self.score_method: str = str(p6_cfg.get("SCORE_METHOD", "rank_sum")).strip().lower()
        sw = p6_cfg.get("SCORE_WEIGHTS")
        self.score_weights = (
            {str(k): float(v) for k, v in sw.items()} if isinstance(sw, dict) else None
        )
        self.score_winsor_sigma: float = float(p6_cfg.get("SCORE_WINSOR_SIGMA", 3.0))
        self.use_momentum_12_2: bool = bool(p6_cfg.get("USE_MOMENTUM_12_2", False))
        self.use_op_profitability: bool = bool(p6_cfg.get("USE_OPERATING_PROFITABILITY", False))
        self.use_asset_growth: bool = bool(p6_cfg.get("USE_ASSET_GROWTH", False))
        self.momentum_lookback_days: int = int(p6_cfg.get("MOMENTUM_LOOKBACK_DAYS", 252))
        self.momentum_skip_days: int = int(p6_cfg.get("MOMENTUM_SKIP_DAYS", 21))
        self.exclusions_cfg: dict = dict(p6_cfg.get("EXCLUSIONS", {}))

        # Team B / B1 -- weighting method dispatch.
        wm = str(p6_cfg.get("WEIGHTING_METHOD", "INV_VOL")).strip().upper()
        if wm not in _WEIGHTING_DISPATCH:
            self.logger.warning("[P6] Unknown WEIGHTING_METHOD=%r; falling back to INV_VOL.", wm)
            wm = "INV_VOL"
        self.weighting_method: str = wm
        self.hrp_linkage_method: str = str(p6_cfg.get("HRP_LINKAGE_METHOD", "single")).strip().lower()

        self.fundamentals_df: Optional[pd.DataFrame] = self._load_fundamentals()
        self._target_weights: Dict[str, float] = {}
        self._last_rebalance_date: Optional[Tuple[int, int, int]] = None
        self._holdings_state_path = Path(__file__).resolve().parent / "current_holdings.json"

        self.logger.info(
            "Portfolio6Strategy init: candidates=%s, top_n=%s, vol_target=%.2f, "
            "max_leverage=%.2f, gld=%s(%.2f), trend=%s(%.2f), fundamentals=%s, "
            "score_method=%s, weighting=%s",
            len(self.tickers),
            self.screen_top_n,
            self.vol_target_annual,
            self.max_leverage,
            self.gld_ticker or "(none)",
            self.gld_weight,
            self.trend_ticker or "(none)",
            self.trend_weight,
            "loaded" if self.fundamentals_df is not None else "missing",
            self.score_method,
            self.weighting_method,
        )

    @staticmethod
    def _load_universe(p6_cfg: dict):
        rel_path = str(p6_cfg.get("UNIVERSE_PATH", "src/portfolios/portfolio_6/universe.json"))
        full_path = REPO_ROOT / rel_path
        if not full_path.exists():
            return []
        try:
            with open(full_path, "r") as f:
                data = json.load(f)
            return [str(t).strip() for t in data if isinstance(t, str) and str(t).strip()]
        except (OSError, json.JSONDecodeError):
            return []

    def _load_fundamentals(self) -> Optional[pd.DataFrame]:
        if not self.use_fundamentals:
            return None
        full_path = REPO_ROOT / self.fundamentals_csv_rel
        if not full_path.exists():
            self.logger.warning(
                "Fundamentals CSV missing at %s; profitable-screen disabled.",
                full_path,
            )
            return None
        try:
            df = pd.read_csv(full_path)
        except Exception as e:
            self.logger.exception("Failed to load fundamentals CSV: %s", e)
            return None
        if "ticker" not in df.columns:
            self.logger.warning(
                "Fundamentals CSV missing 'ticker' column; profitable-screen disabled."
            )
            return None
        return df.set_index("ticker")

    def OnData(self, context: StrategyContext):
        if context is None or context.time is None:
            return

        date_key = (context.time.year, context.time.month, context.time.day)
        if self._last_rebalance_date != date_key:
            self.logger.info(
                "[P6] Running daily screen+rebalance at %s", context.time
            )
            self._rebalance(context)
            self._last_rebalance_date = date_key

        if self._target_weights:
            self._execute_orders(context)

    def _collect_returns(self, context: StrategyContext) -> Dict[str, pd.Series]:
        lookback_str = f"{self.vol_lookback_days + 30}d"
        out: Dict[str, pd.Series] = {}
        candidates = [
            t for t in self.tickers if t not in (self.gld_ticker, self.trend_ticker)
        ]
        min_required = max(self.vol_lookback_days // 2, 30)

        for ticker in candidates:
            asset = context.Market[ticker]
            if not asset.Exists:
                continue
            hist = asset.History(lookback_str)
            if hist is None or hist.empty or "close_price" not in hist.columns:
                continue
            close = pd.to_numeric(hist["close_price"], errors="coerce").dropna()
            if len(close) < min_required + 1:
                continue
            returns = close.pct_change().dropna()
            if len(returns) < min_required:
                continue
            out[ticker] = returns.iloc[-self.vol_lookback_days:]
        return out

    def _rebalance(self, context: StrategyContext):
        returns_matrix = self._collect_returns(context)
        if not returns_matrix:
            self.logger.warning(
                "[P6] No tickers have sufficient history; skipping rebalance."
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
            self.logger.warning("[P6] Top-N selection empty; skipping rebalance.")
            return

        # Team B / B1 -- dispatch on weighting method. INV_VOL = bit-exact.
        weighter = _WEIGHTING_DISPATCH[self.weighting_method]
        weight_kwargs = {"max_weight": self.max_weight}
        if self.weighting_method == "HRP":
            weight_kwargs["linkage_method"] = self.hrp_linkage_method
        # B2 hook: when a shrinkage Sigma provider is wired up on the strategy
        # instance (self._shrinkage_cov), HRP/ERC pick it up automatically.
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
                "[P6] %s weighting failed (%s); falling back to inverse_vol.",
                self.weighting_method, e,
            )
            weights = inverse_vol_weights({t: returns_matrix[t] for t in top}, max_weight=self.max_weight)

        if not weights:
            self.logger.warning("[P6] %s weighting empty; skipping rebalance.", self.weighting_method)
            return

        returns_df = pd.DataFrame({t: returns_matrix[t] for t in top}).dropna(how="all")
        weight_series = pd.Series(weights).reindex(returns_df.columns).fillna(0.0)
        sleeve_returns = (returns_df * weight_series).sum(axis=1)

        try:
            dsr = deflated_sharpe_ratio(sleeve_returns, n_trials=len(returns_matrix))
            self.logger.info(
                "[P6] Deflated Sharpe probability=%.3f (n_trials=%s, top_n=%s)",
                dsr,
                len(returns_matrix),
                len(top),
            )
            if dsr < self.dsr_min_prob:
                self.logger.warning(
                    "[P6] DSR=%.3f below threshold %.2f; selection may be noise.",
                    dsr,
                    self.dsr_min_prob,
                )
        except Exception as e:
            self.logger.exception("[P6] DSR computation failed: %s", e)

        # Sleeve-level vol-target (single authoritative layer per B3).
        # vol_scale = min(VOL_TARGET_ANNUAL / sigma_realized, MAX_LEVERAGE).
        # Applied only to the stock sleeve. GLD + optional trend hedge added
        # below at FIXED config weights (vol-naive). See
        # .claude/agents-output/teamB/B3_vol_target_audit.md.
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
            else:
                self.logger.warning(
                    "[P6] GLD ticker '%s' has no market data; sleeve dropped.",
                    self.gld_ticker,
                )

        if self.trend_ticker and self.trend_ticker in self.tickers:
            trend_asset = context.Market[self.trend_ticker]
            if trend_asset.Exists:
                target_weights[self.trend_ticker] = self.trend_weight
            else:
                self.logger.warning(
                    "[P6] Trend hedge ticker '%s' has no market data; sleeve dropped.",
                    self.trend_ticker,
                )
        elif self.trend_weight > 0 and not self.trend_ticker:
            self.logger.info(
                "[P6] TREND_HEDGE_TICKER unset; trend sleeve weight %.2f dropped.",
                self.trend_weight,
            )

        # Gross-notional cap (NOT a second vol-target). Proportional rescale
        # when total > MAX_LEVERAGE. Flagged in B3 §4.3 for follow-up.
        total = sum(target_weights.values())
        if total > self.max_leverage > 0:
            scale = self.max_leverage / total
            target_weights = {t: w * scale for t, w in target_weights.items()}
            self.logger.info(
                "[P6] Leverage cap applied: total %.3f scaled to %.3f.",
                total,
                self.max_leverage,
            )

        self._target_weights = target_weights
        ranked = sorted(target_weights.items(), key=lambda kv: -kv[1])
        self.logger.info(
            "[P6] New targets: positions=%s, total_weight=%.3f, top5=%s",
            len(target_weights),
            sum(target_weights.values()),
            ranked[:5],
        )
        self._persist_holdings(context, ranked)

    def _persist_holdings(self, context: StrategyContext, ranked):
        """Write today's selected tickers + weights to current_holdings.json so
        the active 50-name list is visible/auditable outside the process."""
        try:
            payload = {
                "as_of": context.time.isoformat() if context.time else None,
                "portfolio_id": self.portfolio_id,
                "weighting_method": self.weighting_method,
                "score_method": self.score_method,
                "n_positions": len(ranked),
                "total_weight": float(sum(w for _, w in ranked)),
                "holdings": [
                    {"ticker": t, "weight": round(float(w), 6)} for t, w in ranked
                ],
            }
            with open(self._holdings_state_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            self.logger.warning(
                "[P6] Failed to persist current_holdings.json: %s", e
            )

    def _execute_orders(self, context: StrategyContext):
        for ticker, qty in list(context.Portfolio.positions.items()):
            try:
                qty_f = float(qty)
            except (TypeError, ValueError):
                qty_f = 0.0
            if qty_f == 0 or ticker in self._target_weights:
                continue
            self._issue_target(context, ticker, 0.0)

        for ticker, target_weight in self._target_weights.items():
            asset = context.Market[ticker]
            if not asset.Exists or asset.Close is None or asset.Close <= 0:
                continue
            current_weight = context.Portfolio.get_asset_weight(ticker, asset.Close)
            if abs(target_weight - current_weight) < self.rebalance_drift:
                continue
            self._issue_target(context, ticker, target_weight)

    def _issue_target(self, context: StrategyContext, ticker: str, target_weight: float):
        asset = context.Market[ticker]
        if not asset.Exists or asset.Close is None or asset.Close <= 0:
            return
        if self.executor is None:
            self.logger.warning("[P6] No executor available; cannot trade %s.", ticker)
            return
        safe_weight = max(float(target_weight), 1e-9)
        self.executor.execute_trade(
            portfolio_id=self.portfolio_id,
            ticker=ticker,
            signal_type="BUY",
            confidence=1.0,
            arrival_price=asset.Close,
            cash=context.Portfolio.cash,
            positions=None,
            port_notional=context.Portfolio.total_value,
            ticker_weight=safe_weight,
            timestamp=context.time,
        )
