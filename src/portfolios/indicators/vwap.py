from collections import deque
from datetime import datetime
from .base import Indicator

class VWAP(Indicator):
    """
    A stateful, rolling Volume Weighted Average Price (VWAP) indicator.
    Calculates the VWAP over a rolling window of 'period' bars.
    VWAP = sum(Price * Volume) / sum(Volume)
    """
    def __init__(self, ticker: str, **kwargs):
        """
        Initializes the VWAP indicator.
        
        Args:
            ticker (str): The ticker symbol this indicator is for.
            **kwargs:
                period (int): The lookback window (e.g., 20).
                price_col (str): The name of the price column (default: 'close_price').
                vol_col (str): The name of the volume column (default: 'volume').
        """
        super().__init__(ticker=ticker, **kwargs)
        
        self.period = int(kwargs.get('period', 20))
        self.price_col = kwargs.get('price_col', 'close_price')
        self.vol_col = kwargs.get('vol_col', 'volume')
        
        if self.period <= 0:
            raise ValueError("VWAP indicator requires a 'period' > 0.")

        # Buffers to track (price * volume) and volume separately
        self._pv_buffer = deque(maxlen=self.period)
        self._vol_buffer = deque(maxlen=self.period)
        self._sum_pv = 0.0
        self._sum_vol = 0.0

    def Update(self, timestamp: datetime, data_point: float, **kwargs):
        """
        Updates the indicator with a new price and optional volume.
        
        Args:
            timestamp (datetime): The timestamp of the new data.
            data_point (float): The price value.
            **kwargs: Optional 'volume' parameter (defaults to 1.0).
        """
        # Get volume from kwargs, default to 1.0 if not provided
        volume = float(kwargs.get('volume', 1.0))
        
        pv = data_point * volume
        
        # If buffer is full, subtract the oldest values
        if len(self._pv_buffer) == self.period:
            old_pv = self._pv_buffer[0]
            old_vol = self._vol_buffer[0]
            self._sum_pv -= old_pv
            self._sum_vol -= old_vol

        # Add new values to buffers and sums
        self._pv_buffer.append(pv)
        self._vol_buffer.append(volume)
        self._sum_pv += pv
        self._sum_vol += volume

        # Check if indicator is ready
        if len(self._pv_buffer) == self.period:
            self._is_ready = True
            if self._sum_vol > 0:
                self._current_value = self._sum_pv / self._sum_vol
            else:
                # Fallback to simple price average if total volume is 0
                self._current_value = data_point
        
        return self._current_value

