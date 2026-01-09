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
    from portfolios.indicators.base import Indicator

class DisplacedMovingAverage(Indicator):
    """
    A stateful Displaced Moving Average (DMA) indicator.
    This indicator calculates a Simple Moving Average and shifts it in time.
    A positive displacement shifts the MA into the future (lags).
    """
    def __init__(self, ticker: str, **kwargs):
        super().__init__(ticker=ticker, **kwargs)
        
        self.period = int(kwargs.get('period', 14))
        # Displacement is in number of bars/periods
        self.displacement = int(kwargs.get('displacement', 5)) 
        self.price_col = kwargs.get('price_col', 'close_price')
        
        if self.period <= 0:
            raise ValueError("DisplacedMovingAverage requires a 'period' > 0.")

        # We need to store enough prices to cover the period and the displacement
        buffer_size = self.period + self.displacement
        self._price_buffer = deque(maxlen=buffer_size)
        
    def Update(self, timestamp: datetime, data_point: float):
        """
        Updates the indicator with a new data point and calculates the displaced average.
        """
        self._price_buffer.append(data_point)

        # We are ready once the buffer is full enough to get a displaced window
        if len(self._price_buffer) == self._price_buffer.maxlen:
            # The DMA value for the CURRENT time is the SMA of a PAST window.
            # The window ends 'displacement' periods ago.
            end_index = len(self._price_buffer) - self.displacement
            start_index = end_index - self.period
            
            # Slice the deque to get the correct historical window
            window = [self._price_buffer[i] for i in range(start_index, end_index)]
            
            self._current_value = sum(window) / self.period
            self._is_ready = True