"""
src/backtest/cost_model.py

Realistic transaction-cost model for the MQS backtest stack.

Literature:
  - Almgren, Thum, Hauptmann, Li (2005), "Direct Estimation of Equity Market Impact".
  - Bouchaud et al. "Trades, Quotes and Prices" (CUP 2018). Square-root law:
        I(Q) = Y * sigma_daily * sqrt(Q / ADV),  with Y ~ O(1).
  - Kissell & Glantz (2003), "Optimal Trading Strategies". I-Star MI_bp.
  - Frazzini, Israel, Moskowitz (2018), "Trading Costs of Asset Pricing Anomalies",
    AQR / SSRN 2294498.

Components (basis points):
    fixed_bps        : commission + exchange fees + ticket cost
    spread_bps       : full bid-ask spread; we charge half on each side
    impact_bps       : alpha * sigma_daily * sqrt(trade_notional / adv_notional) * 1e4

Total per-trade cost in bps:
    total_bps = fixed_bps + 0.5 * spread_bps + impact_bps
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping


@dataclass(frozen=True)
class CostModelParams:
    """Parameters for the transaction-cost model.

    Defaults calibrated to S&P 500 / NDX universe at $1M-$100M AUM.
    For Russell 2000 / micro-cap universes use ``CostModel.for_small_cap``.
    """

    fixed_bps: float = 0.5
    spread_bps_default: float = 5.0
    alpha_impact: float = 1.0
    min_impact_bps: float = 0.0
    max_impact_bps: float = 500.0
    enable_impact: bool = True
    enable_spread: bool = True
    enable_fixed: bool = True

    def with_overrides(self, **overrides) -> "CostModelParams":
        kwargs = {**self.__dict__, **overrides}
        return CostModelParams(**kwargs)


class CostModel:
    """Per-trade cost model returning cost in basis points and as fractional cost."""

    def __init__(
        self,
        params: CostModelParams | None = None,
        spread_overrides: Mapping[str, float] | None = None,
    ):
        self.params: CostModelParams = params or CostModelParams()
        self.spread_overrides: Mapping[str, float] = dict(spread_overrides or {})

    def _spread_component_bps(self, ticker: str | None) -> float:
        if not self.params.enable_spread:
            return 0.0
        if ticker is not None and ticker in self.spread_overrides:
            return 0.5 * float(self.spread_overrides[ticker])
        return 0.5 * float(self.params.spread_bps_default)

    def _impact_component_bps(
        self,
        trade_notional: float,
        adv_notional: float,
        sigma_daily: float,
    ) -> float:
        if not self.params.enable_impact:
            return 0.0
        if trade_notional <= 0 or adv_notional <= 0 or sigma_daily <= 0:
            return 0.0
        ratio = float(trade_notional) / float(adv_notional)
        if ratio <= 0:
            return 0.0
        raw_bps = self.params.alpha_impact * float(sigma_daily) * math.sqrt(ratio) * 1e4
        return max(self.params.min_impact_bps, min(raw_bps, self.params.max_impact_bps))

    def _fixed_component_bps(self) -> float:
        if not self.params.enable_fixed:
            return 0.0
        return float(self.params.fixed_bps)

    def cost_bps(
        self,
        trade_notional: float,
        adv_notional: float,
        sigma_daily: float,
        ticker: str | None = None,
    ) -> float:
        return (
            self._fixed_component_bps()
            + self._spread_component_bps(ticker)
            + self._impact_component_bps(trade_notional, adv_notional, sigma_daily)
        )

    def cost_fraction(
        self,
        trade_notional: float,
        adv_notional: float,
        sigma_daily: float,
        ticker: str | None = None,
    ) -> float:
        return self.cost_bps(trade_notional, adv_notional, sigma_daily, ticker) * 1e-4

    def apply_to_price(
        self,
        mid_price: float,
        side: str,
        trade_notional: float,
        adv_notional: float,
        sigma_daily: float,
        ticker: str | None = None,
    ) -> float:
        if mid_price <= 0:
            return mid_price
        frac = self.cost_fraction(trade_notional, adv_notional, sigma_daily, ticker)
        side_u = side.upper()
        if side_u == "BUY":
            return mid_price * (1.0 + frac)
        if side_u == "SELL":
            return mid_price * (1.0 - frac)
        return mid_price

    @classmethod
    def for_large_cap(cls) -> "CostModel":
        return cls(CostModelParams(
            fixed_bps=0.5,
            spread_bps_default=4.0,
            alpha_impact=1.0,
        ))

    @classmethod
    def for_small_cap(cls) -> "CostModel":
        return cls(CostModelParams(
            fixed_bps=1.0,
            spread_bps_default=40.0,
            alpha_impact=1.5,
        ))

    @classmethod
    def disabled(cls) -> "CostModel":
        return cls(CostModelParams(
            enable_fixed=False,
            enable_spread=False,
            enable_impact=False,
        ))
