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

class AverageTrueRange(Indicator):
    """
    A stateful, rolling Average True Range (ATR) indicator.
    ATR = Simple Moving Average of TR
    """
    def __init__(self, ticker: str, **kwargs):
        super().__init__(ticker=ticker, **kwargs)
        
        self.period = int(kwargs.get('period', 14))
        self.high_col = kwargs.get('high_col', 'high_price')
        self.low_col = kwargs.get('low_col', 'low_price')
        self.close_col = kwargs.get('close_col', 'close_price')
        self._logger = logging.getLogger(f"{self.__class__.__name__}_{self.ticker}")
        
        if self.period <= 0:
            raise ValueError("ATR indicator requires a 'period' > 0.")

        self._tr_buffer = deque(maxlen=self.period)
        self._prev_close = None
        self._sum_tr = 0.0

    def Update(self, timestamp: datetime, data_point: float, **kwargs):
        """
        Update ATR with new data.
        data_point: close price (required)
        **kwargs: 'high' and 'low' prices (optional, default to close price)
        """
        close = data_point
        high = float(kwargs.get(self.high_col, close))
        low = float(kwargs.get(self.low_col, close))

        if self._prev_close is None:
            self._prev_close = close
            return

        tr_1 = high - low
        tr_2 = abs(high - self._prev_close)
        tr_3 = abs(low - self._prev_close)
        true_range = max(tr_1, tr_2, tr_3)

        if len(self._tr_buffer) == self.period:
            old_tr = self._tr_buffer[0]
            self._sum_tr -= old_tr

        self._tr_buffer.append(true_range)
        self._sum_tr += true_range
        self._prev_close = close

        if len(self._tr_buffer) == self.period:
            self._is_ready = True
            if self.period > 0:
                self._current_value = self._sum_tr / self.period
            else:
                self._current_value = 0.0
