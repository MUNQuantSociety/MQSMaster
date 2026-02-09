import logging
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


class ExponentialMovingAverage(Indicator):
    """
    A stateful Exponential Moving Average (EMA) indicator.

    EMA gives more weight to recent prices than SMA.
    Formula: EMA = (Price * multiplier) + (Previous EMA * (1 - multiplier))
    where multiplier = (smoothing) / (period + 1)
        A higher smoothing value places more weight on the new value (more responsive)
        Lower places more weight on previous value (less responsive).
    """

    def __init__(self, ticker: str, **kwargs):
        super().__init__(ticker=ticker, **kwargs)

        self.period = int(kwargs.get('period'))
        self.price_col = kwargs.get('price_col', 'close_price')

        if not self.period or self.period <= 0:
            raise ValueError("ExponentialMovingAverage requires a 'period' > 0.")

        # EMA multiplier: (smoothing) / (period + 1)
        self.multiplier = 2.0 / (self.period + 1)

        # Internal state
        self._ema = None
        self._count = 0
        self._sum = 0.0  # Used for initial SMA calculation

    def Update(self, timestamp: datetime, data_point: float, **kwargs):
        """
        Updates the EMA with a new data point.

        For the first 'period' data points, we calculate SMA as the seed.
        After that, we use the EMA formula.
        """
        self._count += 1 #increment time

        # EMA is initially none. Period size (declared in strategy, see config.json) is used for avg
        # Wait until period length reached, then calculate EMA
        if self._ema is None:
            # Accumulate values for initial SMA seed
            self._sum += data_point

            if self._count == self.period:
                # Use SMA as the initial EMA value
                self._ema = self._sum / self.period
                self._current_value = self._ema
                self._is_ready = True # moving average initialized, is ready.
        else:
            # Apply EMA formula: EMA = (Price * k) + (Previous EMA * (1 - k))
            self._ema = (data_point * self.multiplier) + (self._ema * (1 - self.multiplier))
            self._current_value = self._ema
