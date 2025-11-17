from collections import deque
from datetime import datetime
import numpy as np
from .base import Indicator

class RelativeMomentumIndex(Indicator):
    """
    A corrected, stateful Relative Momentum Index (RMI) indicator.
    """
    def __init__(self, ticker: str, **kwargs):
        super().__init__(ticker=ticker, **kwargs)
        
        self.period = int(kwargs.get('period', 14))
        self.momentum_period = int(kwargs.get('momentum_period', 3))
        self.price_col = kwargs.get('price_col', 'close_price')

        if self.period <= 0: raise ValueError("'period' must be positive")
        if self.momentum_period <= 0: raise ValueError("'momentum_period' must be positive")

        # --- @ Lodo, I'm now using a deque to prevent memory leaks ---
        # We only need to store momentum_period + 1 prices to calculate momentum
        self._prices = deque(maxlen=self.momentum_period + 1)
        
        self._avg_gain: float | None = None
        self._avg_loss: float | None = None
        
        # --- Using deques for seeding to simplify logic ---
        self._seed_gains = deque(maxlen=self.period)
        self._seed_losses = deque(maxlen=self.period)

    def Update(self, timestamp: datetime, data_point: float):
        """
        Update the indicator with a new price.
        """
        self._prices.append(data_point)

        # Not enough data to calculate even one momentum value yet
        if len(self._prices) < self.momentum_period + 1:
            return

        momentum = self._prices[-1] - self._prices[0]
        gain = max(momentum, 0.0)
        loss = max(-momentum, 0.0)

        # If the indicator is not yet "ready", we are in the initial seeding phase
        if not self.IsReady:
            self._seed_gains.append(gain)
            self._seed_losses.append(loss)
            
            # --- Use a simple average for the first calculation ---
            # Once we have 'period' number of gains/losses, we can seed the averages
            if len(self._seed_gains) == self.period:
                self._avg_gain = np.mean(self._seed_gains)
                self._avg_loss = np.mean(self._seed_losses)
                self._is_ready = True # The indicator is now ready
        
        # If ready, perform the standard Wilder's smoothing update
        else:
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period

        # --- Compute RMI ---
        # This only runs if the indicator has become ready in this step or was already ready
        if self.IsReady:
            if self._avg_loss == 0:
                self._current_value = 100.0
            else:
                rs = self._avg_gain / self._avg_loss
                self._current_value = 100.0 - (100.0 / (1.0 + rs))
            return self._current_value