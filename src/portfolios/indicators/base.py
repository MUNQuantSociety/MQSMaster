# src/portfolios/indicators/base.py

from abc import ABC, abstractmethod
from datetime import datetime

class Indicator(ABC):
    """
    Abstract base class for all stateful technical indicators.
    """
    def __init__(self, ticker: str, **kwargs):
        self.ticker = ticker
        self.kwargs = kwargs  # Store all parameters
        self._is_ready = False
        self._current_value = None

    @property
    def IsReady(self) -> bool:
        """Returns True if the indicator has enough data to produce a value."""
        return self._is_ready

    @property
    def Current(self):
        """Returns the latest value of the indicator."""
        return self._current_value

    @abstractmethod
    def Update(self, timestamp: datetime, data_row):
        """
        Updates the indicator with a new data row (dict or pd.Series).
        """
        raise NotImplementedError("Each indicator must implement the 'Update' method.")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.ticker}, {self.period}) -> Value: {self.Current if self.IsReady else 'NotReady'}"