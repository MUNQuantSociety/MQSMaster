# src/portfolios/indicators/simple_moving_average.py

import logging
from collections import deque
from datetime import datetime

try:
    from .base import Indicator
except ImportError as rel_err:
    logging.warning(
        "Indicator relative import failed; using absolute import. Details: %s",
        rel_err,
    )
    try:
        from src.portfolios.indicators.base import Indicator
    except ImportError:
        logging.error(
            "Failed to import Indicator from both relative and absolute paths."
        )
        raise

class SimpleMovingAverage(Indicator):
    """
    A stateful Simple Moving Average (SMA) indicator.
    """
    def __init__(self, ticker: str, **kwargs):
        super().__init__(ticker=ticker, **kwargs)
        
        # Extract specific parameters from kwargs
        self.period = int(kwargs.get('period'))
        self.price_col = kwargs.get('price_col', 'close_price')
        
        if not self.period:
            raise ValueError("SimpleMovingAverage requires a 'period' keyword argument.")

        self._window = deque(maxlen=self.period)
        self._sum = 0.0

    def Update(self, timestamp: datetime, data_point: float):
        """
        Efficiently updates the SMA with a new data point in O(1) time.
        """
        if len(self._window) == self.period:
            # Subtract the oldest value as it slides out of the window
            self._sum -= self._window[0]
        
        self._window.append(data_point)
        self._sum += data_point

        if len(self._window) == self.period:
            self._current_value = self._sum / self.period
            self._is_ready = True