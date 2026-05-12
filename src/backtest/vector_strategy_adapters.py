# [Incomplete: going to implement V2] module for vectorized strategy adapters. These functions provide a way to compute target weights and signal strength for various strategies in a vectorized manner, suitable for backtesting. Each adapter corresponds to a specific strategy and implements its logic using pandas DataFrames. The get_vector_adapter_for_portfolio function allows retrieval of the appropriate adapter based on the class name of the portfolio instance. This design enables separation of vectorized logic from event-driven strategy implementations, facilitating efficient backtesting while maintaining code clarity. 
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class VectorSignalResult:
    target_weights: pd.DataFrame
    signal_strength: pd.DataFrame


def _compute_rsi(price_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = price_df.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _compute_rmi(
    price_df: pd.DataFrame,
    period: int = 14,
    momentum_period: int = 14,
) -> pd.DataFrame:
    mom = price_df - price_df.shift(momentum_period)
    gains = mom.clip(lower=0.0)
    losses = -mom.clip(upper=0.0)

    avg_gain = gains.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rm = avg_gain / avg_loss.replace(0.0, np.nan)
    rmi = 100.0 - (100.0 / (1.0 + rm))
    return rmi.fillna(50.0)


def _normalise_active_weights(active_mask: pd.DataFrame) -> pd.DataFrame:
    active_float = active_mask.astype(float)
    row_sum = active_float.sum(axis=1).replace(0.0, np.nan)
    return active_float.div(row_sum, axis=0).fillna(0.0)


def momentum_strategy_adapter(price_df: pd.DataFrame) -> VectorSignalResult:
    """
    External vectorized approximation of MomentumStrategy indicator logic.

    This adapter intentionally lives outside strategy.py to preserve the event-mode
    strategy implementation while enabling fast vectorized backtests.
    """
    close = price_df.sort_index().ffill().dropna(how="all")
    if close.empty:
        empty = pd.DataFrame(index=price_df.index, columns=price_df.columns).fillna(0.0)
        return VectorSignalResult(target_weights=empty, signal_strength=empty)

    sma_fast = close.rolling(14, min_periods=14).mean()
    sma_slow = close.rolling(28, min_periods=28).mean()
    dma = close.rolling(14, min_periods=14).mean().shift(7)
    rsi = _compute_rsi(close, period=14)
    rmi = _compute_rmi(close, period=14, momentum_period=14)

    bullish = (sma_fast > sma_slow) & (dma > sma_slow)
    bearish = (sma_fast < sma_slow) & (dma < sma_slow)

    oversold = ((rsi > 10) & (rsi < 30)) | ((rmi > 10) & (rmi < 30))
    overbought = ((rsi > 70) & (rsi < 90)) | ((rmi > 70) & (rmi < 90))

    entry_signal = bullish | oversold
    # Explicit precedence: entry has priority over exit when mixed conditions appear.
    exit_signal = (bearish | overbought) & ~entry_signal

    target_state = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    target_state[entry_signal] = 1.0
    target_state[exit_signal] = 0.0
    target_state = target_state.ffill().fillna(0.0)

    weights = _normalise_active_weights(target_state > 0.0)

    signal_strength = (
        bullish.astype(float)
        + oversold.astype(float)
        - bearish.astype(float)
        - overbought.astype(float)
    )

    return VectorSignalResult(target_weights=weights, signal_strength=signal_strength)


def vol_momentum_adapter(price_df: pd.DataFrame) -> VectorSignalResult:
    """External vector approximation of VolMomentum with dynamic momentum threshold."""
    close = price_df.sort_index().ffill().dropna(how="all")
    if close.empty:
        empty = pd.DataFrame(index=price_df.index, columns=price_df.columns).fillna(0.0)
        return VectorSignalResult(target_weights=empty, signal_strength=empty)

    roc_20 = close.pct_change(20)
    vol_60 = close.pct_change().rolling(60, min_periods=20).std() * np.sqrt(252)
    threshold = vol_60 * 1.5

    bullish = roc_20 > threshold
    bearish = roc_20 < (-threshold)

    target_state = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    target_state[bullish] = 1.0
    target_state[bearish] = 0.0
    target_state = target_state.ffill().fillna(0.0)

    weights = _normalise_active_weights(target_state > 0.0)
    signal_strength = (roc_20 - threshold).fillna(0.0)
    return VectorSignalResult(target_weights=weights, signal_strength=signal_strength)


def regime_adaptive_adapter(price_df: pd.DataFrame) -> VectorSignalResult:
    """External vector approximation of RegimeAdaptiveStrategy using close data."""
    close = price_df.sort_index().ffill().dropna(how="all")
    if close.empty:
        empty = pd.DataFrame(index=price_df.index, columns=price_df.columns).fillna(0.0)
        return VectorSignalResult(target_weights=empty, signal_strength=empty)

    vix_col = "^VIX" if "^VIX" in close.columns else None
    if vix_col is None:
        regime_high_vol = pd.Series(False, index=close.index)
    else:
        regime_high_vol = close[vix_col] > 18

    momentum_5 = close.pct_change(5)
    mean_20 = close.rolling(20, min_periods=20).mean()
    std_20 = close.pct_change().rolling(20, min_periods=20).std().replace(0.0, np.nan)
    zscore_20 = (close - mean_20) / (close * std_20)

    buy_momentum = momentum_5 > 0.002
    sell_momentum = momentum_5 < -0.002
    buy_fade = zscore_20 < -1.0
    sell_fade = zscore_20 > 1.0

    regime_mask = pd.DataFrame(
        np.tile(regime_high_vol.to_numpy()[:, None], (1, close.shape[1])),
        index=close.index,
        columns=close.columns,
    )
    if vix_col in regime_mask.columns:
        regime_mask[vix_col] = False

    buy_signal = (regime_mask & buy_fade) | ((~regime_mask) & buy_momentum)
    sell_signal = (regime_mask & sell_fade) | ((~regime_mask) & sell_momentum)

    target_state = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    target_state[buy_signal] = 1.0
    target_state[sell_signal] = 0.0
    target_state = target_state.ffill().fillna(0.0)
    if vix_col in target_state.columns:
        target_state[vix_col] = 0.0

    weights = _normalise_active_weights(target_state > 0.0)
    signal_strength = (buy_signal.astype(float) - sell_signal.astype(float)).fillna(0.0)
    return VectorSignalResult(target_weights=weights, signal_strength=signal_strength)


def trend_rotate_adapter(price_df: pd.DataFrame) -> VectorSignalResult:
    """External vector approximation of TrendRotateStrategy using SMA regime rotation."""
    close = price_df.sort_index().ffill().dropna(how="all")
    if close.empty:
        empty = pd.DataFrame(index=price_df.index, columns=price_df.columns).fillna(0.0)
        return VectorSignalResult(target_weights=empty, signal_strength=empty)

    risk_on = [ticker for ticker in close.columns if not str(ticker).startswith("TLT")]
    risk_off = [ticker for ticker in close.columns if str(ticker).startswith("TLT")]
    if not risk_off and close.columns.size > 0:
        fallback_risk_off = close.columns[-1]
        risk_off = [fallback_risk_off]
        risk_on = [ticker for ticker in risk_on if ticker != fallback_risk_off]

    sma_fast = close.rolling(20, min_periods=20).mean()
    sma_slow = close.rolling(50, min_periods=50).mean()
    trend_up = sma_fast > sma_slow

    risk_on_count = (
        trend_up[risk_on].sum(axis=1) if risk_on else pd.Series(0, index=close.index)
    )
    risk_off_count = (
        trend_up[risk_off].sum(axis=1) if risk_off else pd.Series(0, index=close.index)
    )
    use_risk_on = risk_on_count >= risk_off_count

    target_state = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if risk_on:
        target_state.loc[use_risk_on, risk_on] = trend_up.loc[
            use_risk_on, risk_on
        ].astype(float)
    if risk_off:
        target_state.loc[~use_risk_on, risk_off] = trend_up.loc[
            ~use_risk_on, risk_off
        ].astype(float)

    weights = _normalise_active_weights(target_state > 0.0)
    signal_strength = (sma_fast - sma_slow).fillna(0.0)
    return VectorSignalResult(target_weights=weights, signal_strength=signal_strength)


ADAPTERS_BY_CLASSNAME: Dict[str, Callable[[pd.DataFrame], VectorSignalResult]] = {
    "VolMomentum": vol_momentum_adapter,
    "MomentumStrategy": momentum_strategy_adapter,
    "RegimeAdaptiveStrategy": regime_adaptive_adapter,
    "TrendRotateStrategy": trend_rotate_adapter,
}


def get_vector_adapter_for_portfolio(
    portfolio_instance,
) -> Optional[Callable[[pd.DataFrame], VectorSignalResult]]:
    return ADAPTERS_BY_CLASSNAME.get(portfolio_instance.__class__.__name__)
