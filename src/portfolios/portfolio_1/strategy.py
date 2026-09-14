import logging
import math

# Try relative imports first; on failure, log and attempt absolute imports.
try:
    from portfolios.order_interface import StrategyContext
    from portfolios.portfolio_BASE.strategy import BasePortfolio
except ImportError as rel_err:
    logging.warning(
        "Base Portfolio and order_interface relative import failed; using absolute import. Details: %s",
        rel_err,
    )
    try:
        from src.portfolios.order_interface import StrategyContext
        from src.portfolios.portfolio_BASE.strategy import BasePortfolio
    except ImportError as abs_err:
        logging.error(
            "Failed to import BasePortfolio and StrategyContext from both relative and absolute paths. Details: %s",
            abs_err,
        )
        raise


class VolMomentum(BasePortfolio):
    def __init__(
        self,
        db_connector,
        executor,
        debug=False,
        config_dict=None,
        backtest_start_date=None,
        order_manager=None,
    ):
        super().__init__(
            db_connector,
            executor,
            debug,
            config_dict,
            backtest_start_date,
            order_manager,
        )
        self.logger = logging.getLogger(
            f"{self.__class__.__name__}_{self.portfolio_id}"
        )
        # Format: "indicator_variable_name": ("IndicatorName", {params})
        self.PERIOD = 20
        self.TARGET_WEIGHT = 0.2
        indicator_definitions = {
            "roc": ("RateOfChange", {"period": self.PERIOD}),
        }
        self.RegisterIndicatorSet(indicator_definitions)

    def OnData(self, context: StrategyContext):
        """Generates BUY, SELL, and HOLD signals based on momentum and volatility, updates cash available for trade, and then calls the trade execution logic for each signal."""
        portfolio = context.Portfolio
        is_risk_off = portfolio.cash < (float(portfolio.total_value) * 0.10)
        if is_risk_off:
            self.logger.info(
                "Risk-Off Mode: Cash is low. No new long positions will be opened."
            )

        # ? A loop to iterate through each ticker and generate signals based on momentum and volatility.
        for ticker in self.tickers:
            asset = context.Market[ticker]
            roc = self.roc[ticker]
            vol_multiplier = 1.5  # This can be adjusted or made configurable

            if not all([asset.Exists, roc.IsReady]):
                continue

            return_history = asset.History("60d")
            returns = (
                return_history["close_price"]
                .pct_change(fill_method=None)
                .dropna()
            )
            if len(returns) < self.PERIOD:
                continue
            # ROC is a 20-bar percentage return. Compare it with volatility
            # over the same horizon and keep both values in percent units.
            volatility = float(returns.std() * (self.PERIOD**0.5) * 100.0)

            momentum = roc.Current
            if (
                momentum is None
                or not math.isfinite(momentum)
                or not math.isfinite(volatility)
            ):
                raise ValueError(
                    f"VolMomentum: non-finite momentum or volatility for {ticker}."
                    )
            threshold = volatility * vol_multiplier
            position = portfolio.positions.get(ticker, 0)

            bullish = momentum > threshold
            bearish = momentum < -threshold

            weight = self.TARGET_WEIGHT if bullish else 0.0
            asset_weight = 0.0
            if asset.Exists:
                asset_weight = portfolio.get_asset_weight(ticker, asset.Close)
            if asset_weight <= weight:
                target_weight = True
            elif asset_weight > weight:
                target_weight = False

            if bullish and target_weight and not is_risk_off:  # Max 25% weight
                self.logger.debug(
                    f"[{ticker}] BUY signal: momentum ({momentum:.4f}) > threshold ({threshold:.4f}), position={position}"
                )
                context.buy(ticker, confidence=1.0)

            elif position > 0 and (bearish or target_weight is False or is_risk_off):
                self.logger.debug(
                    f"[{ticker}] SELL signal: momentum ({momentum:.4f}) < threshold ({threshold:.4f}), position={position}"
                )
                context.sell(ticker, confidence=1.0)
