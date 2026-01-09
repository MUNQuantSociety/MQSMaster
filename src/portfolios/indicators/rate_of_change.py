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
    except ImportError as abs_err:
        logging.error(
            "Failed to import Indicator from both relative and absolute paths. Details: %s",
            abs_err,
        )
        raise

class RateOfChange(Indicator):
    """
    A stateful, rolling Rate of Change (ROC) indicator.
    
    Calculates the percentage change in price over a 'period'.
    ROC = ((Current Price - Price 'period' bars ago) / Price 'period' bars ago)
    """
    def __init__(self, ticker: str, **kwargs):
        """
        Initializes the RateOfChange indicator.
        
        Args:
            ticker (str): The ticker symbol this indicator is for.
            **kwargs:
                period (int): The lookback window (e.g., 5).
                price_col (str): The name of the price column (default: 'close_price').
        """
        super().__init__(ticker=ticker, **kwargs)
        
        self.period = int(kwargs.get('period', 5))
        self.price_col = kwargs.get('price_col', 'close_price')
        
        if self.period <= 0:
            raise ValueError("RateOfChange indicator requires a 'period' > 0.")

        # We need to store 'period' + 1 items:
        # The current price and the price 'period' bars ago.
        self._price_buffer = deque(maxlen=self.period + 1)
        

    def Update(self, timestamp: datetime, data_point: float):
        """
        Updates the indicator with a new price.
        
        Args:
            timestamp (datetime): The timestamp of the new data.
            data_point (float): The price value.
        """
        # Add the new price to our buffer
        self._price_buffer.append(data_point)

        # Check if the buffer is full
        if len(self._price_buffer) == self.period + 1:
            self._is_ready = True
            
            old_price = self._price_buffer[0] # The price 'period' bars ago
            new_price = self._price_buffer[-1] # The current price
            
            if old_price is not None and old_price != 0:
                self._current_value = ((new_price - old_price) / old_price) * 100.0
            elif old_price == 0:
                # Avoid division by zero; can't calculate ROC
                self._current_value = 0.0 
            else:
                # Should not happen if data is clean, but good to check
                self._current_value = None
                self._is_ready = False
        return self._current_value

