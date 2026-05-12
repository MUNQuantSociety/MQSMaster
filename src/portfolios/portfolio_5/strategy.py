"""
Portfolio 5: Relevance-Based Prediction (RBP) Strategy.

Uses the RBP model to generate forward return predictions per ticker,
then issues BUY/SELL signals based on configurable thresholds.
"""

import logging

import pandas as pd

try:
    from portfolios.portfolio_BASE.strategy import BasePortfolio
    from portfolios.strategy_api import StrategyContext
except ImportError:
    from src.portfolios.portfolio_BASE.strategy import BasePortfolio
    from src.portfolios.strategy_api import StrategyContext

from src.portfolios.portfolio_5.rbp_model import RBPModel


class RBPStrategy(BasePortfolio):
    """
    Strategy that uses Relevance-Based Prediction to forecast 21-day
    returns and trade accordingly.
    """

    def __init__(
        self,
        db_connector,
        executor,
        debug=False,
        config_dict=None,
        backtest_start_date=None,
    ):
        super().__init__(
            db_connector, executor, debug, config_dict, backtest_start_date
        )
        self.logger = logging.getLogger(
            f"{self.__class__.__name__}_{self.portfolio_id}"
        )

        rbp_cfg = config_dict.get("RBP_CONFIG", {})
        self.feature_cols = rbp_cfg.get(
            "FEATURE_COLS",
            [
                "past_return_21d",
                "past_vol_21d",
                "past_return_63d",
                "past_vol_63d",
                "past_return_252d",
            ],
        )
        self.target_col = rbp_cfg.get("TARGET_COL", "target_return_21d")
        self.split_date = rbp_cfg.get("SPLIT_DATE", "2020-01-01")
        self.buy_threshold = rbp_cfg.get("BUY_THRESHOLD", 0.005)
        self.sell_threshold = rbp_cfg.get("SELL_THRESHOLD", -0.005)
        thresholds = rbp_cfg.get("RELEVANCE_THRESHOLDS", [0.0, 0.2, 0.5, 0.8])

        self.rbp_model = RBPModel(
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            relevance_thresholds=thresholds,
        )
        self._model_fitted = False

    def _ensure_model_fitted(self, context: StrategyContext):
        """Lazily fits the RBP model on first OnData call using available market data."""
        if self._model_fitted:
            return

        market_df = None
        for ticker in self.tickers:
            asset = context.Market[ticker]
            hist = asset.History(f"{self.lookback_days}d")
            if hist.empty:
                continue
            hist = hist.copy()
            hist["ticker"] = ticker
            if "timestamp" in hist.columns:
                hist = hist.drop(columns=["timestamp"])
            market_df = hist if market_df is None else pd.concat([market_df, hist])

        if market_df is None or market_df.empty:
            self.logger.warning("Not enough market data to fit RBP model yet.")
            return

        processed = RBPModel.engineer_features(market_df)
        if processed.empty:
            self.logger.warning("Feature engineering produced no rows.")
            return

        self.rbp_model.fit(processed, self.split_date)
        self._model_fitted = True
        self.logger.info("RBP model fitted successfully.")

    def OnData(self, context: StrategyContext):
        """
        On each time step:
        1. Ensure the RBP model is fitted.
        2. For each ticker, build a feature vector from current market data.
        3. Get the RBP composite prediction.
        4. Issue BUY/SELL based on predicted return vs thresholds.
        """
        import pandas as pd

        self._ensure_model_fitted(context)
        if not self._model_fitted:
            return

        portfolio = context.Portfolio
        is_risk_off = portfolio.cash < (float(portfolio.total_value) * 0.10)

        for ticker in self.tickers:
            asset = context.Market[ticker]
            if not asset.Exists:
                continue

            # Build current feature vector from recent history
            hist = asset.History("300d")
            if hist.empty or len(hist) < 252:
                continue

            close = hist["close_price"]
            daily_ret = close.pct_change()

            x_t = pd.Series(
                {
                    "past_return_21d": close.pct_change(21).iloc[-1],
                    "past_vol_21d": daily_ret.rolling(21).std().iloc[-1],
                    "past_return_63d": close.pct_change(63).iloc[-1],
                    "past_vol_63d": daily_ret.rolling(63).std().iloc[-1],
                    "past_return_252d": close.pct_change(252).iloc[-1],
                }
            )

            if x_t.isna().any():
                continue

            try:
                prediction, rbi_scores = self.rbp_model.predict(x_t)
            except Exception as e:
                self.logger.error("RBP prediction failed for %s: %s", ticker, e)
                continue

            position = portfolio.positions.get(ticker, 0)

            self.logger.debug(
                "[%s] RBP prediction=%.6f, top RBI=%s",
                ticker,
                prediction,
                rbi_scores.sort_values(ascending=False).head(2).to_dict(),
            )

            if prediction > self.buy_threshold and not is_risk_off:
                context.buy(ticker, confidence=min(abs(prediction) * 10, 1.0))
            elif prediction < self.sell_threshold and position > 0:
                context.sell(ticker, confidence=min(abs(prediction) * 10, 1.0))
