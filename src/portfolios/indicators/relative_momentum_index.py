# src/portfolios/indicators/relative_momentum_index.py

from datetime import datetime
import numpy as np
import pandas as pd
from .base import Indicator

class RelativeMomentumIndex(Indicator):
    """
    A stateful Relative Momentum Index (RMI) indicator.
    """
    def __init__(self, ticker: str, **kwargs):
        super().__init__(ticker=ticker, **kwargs)

        self.period = int(kwargs.get('period', 14))
        self.price_col = kwargs.get('price_col', 'close_price')
        self.momentum_period = int(kwargs.get('momentum_period', 5))

        self._previous_price = None
        self._momentum_up = []
        self._momentum_down = []
        self._avg_gain = None
        self._avg_loss = None

    def Update(self, timestamp: datetime, data_point: float):
        """
        Updates the RMI with a new data point.
        """
        if self._previous_price is None:
            self._previous_price = data_point
            return
        change = 0.0
        change = data_point - self._previous_price
        self._previous_price = data_point

        if change > 0:
            self._momentum_up.append(change)
            self._momentum_down.append(0)
        else:
            self._momentum_up.append(0)
            self._momentum_down.append(abs(change))

        mom_up = pd.Series(self._momentum_up)
        mom_down = pd.Series(self._momentum_down)

        self._avg_gain = mom_up.ewm(span=self.momentum_period, adjust=False).mean().iloc[-1]
        self._avg_loss = mom_down.ewm(span=self.momentum_period, adjust=False).mean().iloc[-1]

        rm = self._avg_gain / self._avg_loss if self._avg_loss != 0 else 0
        self._current_value = 100 - (100 / (1 + rm))
        self._is_ready = len(self._momentum_up) >= self.period