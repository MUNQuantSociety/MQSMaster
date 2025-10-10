# src/portfolios/toolkit.py

import pandas as pd
from scipy.ndimage import gaussian_filter1d
from typing import List

@pd.api.extensions.register_dataframe_accessor("toolkit")
@pd.api.extensions.register_series_accessor("toolkit")
class QuantToolkitAccessor:
    """
    A custom pandas accessor to provide quantitative finance-specific tools
    directly on Series and DataFrame objects.

    Usage:
        # On a Series
        smoothed_prices = history['close_price'].toolkit.gaussian_smooth(sigma=2)

        # On a DataFrame
        winsorized_df = history.toolkit.winsorize(limits=[0.05, 0.05])
    """
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def gaussian_smooth(self, sigma: float = 2.0) -> pd.Series:
        """
        Applies a Gaussian smoothing filter to a pandas Series.

        Args:
            sigma (float): The standard deviation for the Gaussian kernel.

        Returns:
            pd.Series: The smoothed data.
        """
        if not isinstance(self._obj, pd.Series):
            raise TypeError("gaussian_smooth can only be called on a pandas Series.")
        
        return pd.Series(
            gaussian_filter1d(self._obj.values, sigma=sigma),
            index=self._obj.index
        )

    def winsorize(self, limits: List[float] = [0.05, 0.05]):
        """
        Applies winsorization to a Series or each column of a DataFrame, capping
        extreme values to reduce the effect of outliers.

        Args:
            limits (List[float]): The lower and upper quantile limits. 
                                  E.g., [0.05, 0.05] caps the bottom 5% and top 5%.

        Returns:
            Union[pd.Series, pd.DataFrame]: The winsorized data.
        """
        if isinstance(self._obj, pd.DataFrame):
            return self._obj.apply(self._winsorize_series, limits=limits)
        else:
            return self._winsorize_series(self._obj, limits=limits)

    @staticmethod
    def _winsorize_series(series: pd.Series, limits: List[float]) -> pd.Series:
        """Helper to winsorize a single Series."""
        series = series.copy()
        quantiles = series.quantile(limits)
        if limits[0] is not None:
            series[series < quantiles.iloc[0]] = quantiles.iloc[0]
        if limits[1] is not None:
            series[series > quantiles.iloc[1]] = quantiles.iloc[1]
        return series