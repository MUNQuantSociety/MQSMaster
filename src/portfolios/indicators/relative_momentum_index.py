# src/portfolios/indicators/relative_momentum_index.py

from datetime import datetime
import numpy as np
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
