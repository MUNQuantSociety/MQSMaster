import logging
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
        self._logger = logging.getLogger(f"{self.__class__.__name__}_{self.ticker}")
        
        if self.period <= 0:
            raise ValueError("VWAP indicator requires a 'period' > 0.")

        # This buffer will store (price * volume, volume) tuples
        self._buffer = deque(maxlen=self.period)
        self._sum_price_x_volume = 0.0
        self._sum_volume = 0.0

    def Update(self, timestamp: datetime, data_row):
        """
        Updates the indicator with a new data row.
        
        Args:
            timestamp (datetime): The timestamp of the new data.
            data_row (dict or pd.Series): The new data bar, containing at least
                                        the price_col and vol_col.
        """
        try:
            price = float(data_row[self.price_col])
            volume = float(data_row[self.vol_col])
        except KeyError as e:
            # Handle missing data in the row
            self._logger.warning(f"Update failed. Data row missing key: {e}")
            self._is_ready = False
            self._current_value = None
            return
        except (TypeError, ValueError):
            # Handle None or non-numeric data
            self._is_ready = False
            self._current_value = None
            return

        new_pv = price * volume
        new_vol = volume

        # If buffer is full, subtract the oldest value before adding the new one
        if len(self._buffer) == self.period:
            old_pv, old_vol = self._buffer[0] # Get the item that's about to be popped
            self._sum_price_x_volume -= old_pv
            self._sum_volume -= old_vol

        # Add new values to buffer and rolling sums
        self._buffer.append((new_pv, new_vol))
        self._sum_price_x_volume += new_pv
        self._sum_volume += new_vol

        # Check if the indicator is ready and set current value
        if len(self._buffer) == self.period:
            self._is_ready = True
            if self._sum_volume > 0:
                self._current_value = self._sum_price_x_volume / self._sum_volume
            else:
                # Fallback to current price if total volume in window is 0
                self._current_value = price

