# src/portfolios/indicators/relative_momentum_index.py

from __future__ import annotations

from datetime import datetime, timezone
import random
from typing import Optional
import numpy as np
import pandas as pd
from src.portfolios.indicators.base import Indicator

class RelativeMomentumIndex(Indicator):
    """Relative Momentum Index (RMI)

    The RMI extends the RSI concept by replacing simple one-period momentum (price change)
    with a d-period momentum: price_t - price_{t-d}. The RSI-style smoothing is then applied
    to the positive and negative momentum values over the lookback ``period``.

    Formula (conceptual):
        M_t = price_t - price_{t - momentum_period}
        Up_t = max(M_t, 0)
        Down_t = max(-M_t, 0)
        AvgGain_t, AvgLoss_t smoothed via Wilder (or EMA equivalent)
        RMI_t = 100 - 100 / (1 + AvgGain_t / AvgLoss_t)

    Readiness: becomes ready after collecting at least ``period + momentum_period`` raw prices
    (enough to compute the first momentum_period-diff series and then the initial averages).
    """

    def __init__(self, ticker: str, **kwargs):
        super().__init__(ticker=ticker, **kwargs)
        self.period: int = int(kwargs.get("period", 14))
        self.price_col: str = kwargs.get("price_col", "close_price")
        self.momentum_period: int = int(kwargs.get("momentum_period", 5))

        if self.period <= 0:
            raise ValueError("'period' must be positive")
        if self.momentum_period <= 0:
            raise ValueError("'momentum_period' must be positive")

        # Internal state
        self._prices: list[float] = []
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        self._last_updated: Optional[datetime] = None

    def _compute_momentum(self) -> float:
        return self._prices[-1] - self._prices[-1 - self.momentum_period]

    def Update(self, timestamp: datetime, data_point: float) -> Optional[float]:  # type: ignore[override]
        """Update the indicator with a new price.

        Returns the current RMI value if ready, else None.
        """
        self._prices.append(float(data_point))
        self._last_updated = timestamp

        # Need at least momentum_period + 1 prices to compute the first momentum value
        if len(self._prices) <= self.momentum_period:
            return None

        momentum = self._compute_momentum()
        gain = max(momentum, 0.0)
        loss = max(-momentum, 0.0)

        if self._avg_gain is None:
            # Build initial window of size 'period'
            # We collect Up/Down values until we have 'period' of them, then seed averages.
            if not hasattr(self, "_seed_gains"):
                self._seed_gains: list[float] = []
                self._seed_losses: list[float] = []
            self._seed_gains.append(gain)
            self._seed_losses.append(loss)
            if len(self._seed_gains) == self.period:
                self._avg_gain = pd.Series(self._seed_gains).ewm(span=self.period, adjust=False).mean().iloc[-1]
                self._avg_loss = pd.Series(self._seed_losses).ewm(span=self.period, adjust=False).mean().iloc[-1]
                del self._seed_gains, self._seed_losses
            else:
                # Not enough momentum observations yet
                self._avg_gain = 0.0
                self._avg_loss = 0.0
                self._current_value = None
                self._is_ready = False
                return None
        elif self._avg_gain is not None and self._avg_loss is not None:
            # Wilder smoothing (both avg values are always initialized together)
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period

        if self._avg_loss == 0:
            rmi_value = 100.0 if self._avg_gain is not None and self._avg_gain > 0 else 0.0
        elif self._avg_gain is not None and self._avg_loss is not None:
            rs = self._avg_gain / self._avg_loss
            rmi_value = 100 - (100 / (1 + rs))
        else:
            rmi_value = 0.0

        # Readiness after we have seeded averages => first computable RMI
        self._current_value = rmi_value
        self._is_ready = self._avg_gain is not None and self._avg_loss is not None
        return self._current_value if self._is_ready else 0.0

    def Reset(self) -> None:
        """Reset internal state to initial (not ready) condition."""
        self._prices.clear()
        self._avg_gain = None
        self._avg_loss = None
        self._current_value = None
        self._is_ready = False
        self._last_updated = None

    # --- Properties / metadata ---

    @property
    def CurrentValue(self) -> Optional[float]:
        """Alias for latest RMI value (may be None if not ready)."""
        return self._current_value

    @property
    def IsReady(self) -> bool:  # type: ignore[override]
        return self._is_ready

    @property
    def LastUpdated(self) -> Optional[datetime]:
        return self._last_updated

    @property
    def Name(self) -> str:
        return f"RMI_{self.ticker}_{self.period}_{self.momentum_period}"

    @property
    def Ticker(self) -> str:
        return self.ticker

    @property
    def PriceCol(self) -> str:
        return self.price_col

    @property
    def Period(self) -> int:
        return self.period

    @property
    def MomentumPeriod(self) -> int:
        return self.momentum_period

    def __str__(self) -> str:  # pragma: no cover - repr convenience
        return (
            f"RMI(ticker={self.ticker}, period={self.period}, momentum_period={self.momentum_period}, "
            f"current_value={self._current_value}, is_ready={self._is_ready})"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return self.__str__()

if __name__ == "__main__":  # simple smoke demo
    rmi = RelativeMomentumIndex(ticker="AAPL", period=3, momentum_period=10)
    prices = [random.uniform(150, 250) for _ in range(1000)]
    return_val = 0

    for i, price in enumerate(prices, start=1):
        val = rmi.Update(datetime.now(timezone.utc), price)
        if val is not None:
            #print(f"Update {i:02d}: price={price:0.2f} rmi={val:0.2f} ready={rmi.IsReady}")
            if val < 20:
                return_val += 2
            if val > 80:
                return_val -= 1
        else:
            print(f"Update {i:02d}: price={price:0.2f} rmi={val} ready={rmi.IsReady}")
    print(f"Return value: {return_val}")
    rmi.Reset()
    print("After reset:", rmi)
        