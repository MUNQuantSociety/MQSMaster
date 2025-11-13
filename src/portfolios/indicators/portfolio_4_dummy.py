from collections import deque
from datetime import datetime
from .base import Indicator

class Portfolio4Dummy(Indicator):
    # did not code this THIS WAS TABBED JUST TO TEST
    def __init__(self, period: int = 14):
        super().__init__()
        self.period = period
        self.values = deque(maxlen=period)
        self.Current = None

    def Update(self, value: float, timestamp: datetime) -> bool:
        self.values.append(value)
        if len(self.values) == self.period:
            self.Current = sum(self.values) / self.period  # Simple average as dummy logic
            return True
        return False