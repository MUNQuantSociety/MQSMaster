# src/portfolios/indicators/relative_strength_index.py

from datetime import datetime
import numpy as np
from .base import Indicator

class RelativeStrengthIndex(Indicator):
    """
    A stateful Relative Strength Index (RSI) indicator.
    """
    def __init__(self, ticker: str, **kwargs):
        super().__init__(ticker=ticker, **kwargs)
        
        self.period = int(kwargs.get('period'))
        self.price_col = kwargs.get('price_col', 'close_price')
        
        if not self.period:
            raise ValueError("RelativeStrengthIndex requires a 'period' keyword argument.")

        self._previous_price = None
        self._gains = []
        self._losses = []
        self._avg_gain = None
        self._avg_loss = None

    def Update(self, timestamp: datetime, data_point: float):
        """
        Updates the RSI with a new data point.
        """
        if self._previous_price is None:
            self._previous_price = data_point
            return

        change = data_point - self._previous_price
        self._previous_price = data_point

        if change > 0:
            self._gains.append(change)
            self._losses.append(0)
        else:
            self._gains.append(0)
            self._losses.append(abs(change))

        # Once the initial period is filled, we calculate the first average
        if len(self._gains) == self.period:
            if self._avg_gain is None: # First-time calculation
                self._avg_gain = np.mean(self._gains)
                self._avg_loss = np.mean(self._losses)
            else: # Subsequent updates use smoothing
                self._avg_gain = (self._avg_gain * (self.period - 1) + self._gains[-1]) / self.period
                self._avg_loss = (self._avg_loss * (self.period - 1) + self._losses[-1]) / self.period

            # Once averages are calculated, we can remove the oldest data point
            self._gains.pop(0)
            self._losses.pop(0)
            
            if self._avg_loss == 0:
                # If there are no losses, RSI is 100
                self._current_value = 100.0
            else:
                rs = self._avg_gain / self._avg_loss
                self._current_value = 100 - (100 / (1 + rs))
            
            self._is_ready = True