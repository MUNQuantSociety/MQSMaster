# data_infra/tradingOps/backtest/reporting.py

import os
import logging
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

from portfolios.portfolio_BASE.strategy import BasePortfolio
from .multiTickerexecutor import MultiTickerExecutor

# --- Core Metric Calculations ---

def _compute_max_drawdown(portfolio_values: pd.Series) -> float:
    """Calculates the maximum drawdown from a series of portfolio values."""
    if len(portfolio_values) < 2:
        return 0.0
    arr = pd.Series(portfolio_values).ffill().bfill().to_numpy(dtype=float)
    if not np.all(np.isfinite(arr)):
        logging.warning("Non-finite values in portfolio values; drawdown set to 0.0")
        return 0.0
    # replace non-positive to avoid divide-by-zero
    arr[arr <= 0] = 1e-9
    peak = np.maximum.accumulate(arr)
    drawdowns = (arr - peak) / peak
    return float(np.min(drawdowns))

def _compute_sharpe_ratio(perf_df: pd.DataFrame) -> float:
    """
    Calculates the annualized Sharpe ratio from a performance DataFrame, robust to
    variable data frequencies. It achieves this by resampling returns to a daily
    frequency before calculating the ratio.
    """
    if perf_df.empty or 'timestamp' not in perf_df or 'portfolio_value' not in perf_df:
        logging.warning("Invalid perf_df for Sharpe Ratio calculation; returning 0.0")
        return 0.0

    # 1. Prepare the DataFrame
    df = perf_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # 2. Resample the portfolio value to get the last value for each day
    # This standardizes the frequency of our data points. 'D' stands for calendar day.
    daily_values = df['portfolio_value'].resample('D').last()

    # 3. Calculate the percentage change of the daily values to get daily returns
    daily_returns = daily_values.pct_change().dropna()

    if daily_returns.empty or len(daily_returns) < 2:
        logging.warning("Not enough daily returns data to calculate a meaningful Sharpe ratio.")
        return 0.0

    # 4. Calculate the mean and standard deviation of the standardized daily returns
    mean_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()

    if std_daily_return == 0 or np.isnan(std_daily_return):
        # If standard deviation is zero, there is no risk.
        # We return 0.0 to avoid division-by-zero errors
        return 0.0

    # 5. Annualize the Sharpe ratio. The standard factor for daily returns is sqrt(252).
    # The risk-free rate is assumed to be 0 for this calculation.
    annualization_factor = np.sqrt(252)
    sharpe_ratio = (mean_daily_return / std_daily_return) * annualization_factor

    return float(sharpe_ratio)

def aggregate_final_metrics(perf_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates key metrics from perf_df."""
    if perf_df.empty or 'portfolio_value' not in perf_df:
        return pd.DataFrame(columns=['metric', 'value'])

    final_val = perf_df['portfolio_value'].iloc[-1]
    max_dd = _compute_max_drawdown(perf_df['portfolio_value'])
    sharpe = _compute_sharpe_ratio(perf_df)

    summary = pd.DataFrame({
        'metric': ['Final Portfolio Value', 'Max Drawdown (%)', 'Annualized Sharpe Ratio'],
        'value': [
            f"{final_val:,.2f}",
            f"{max_dd:.2%}",
            f"{sharpe:.3f}"
        ]
    })
    return summary

# --- Advanced Analytics Calculations ---

def _compute_rolling_stats(
    df_pct_returns: pd.DataFrame,
    columns_to_analyze: List[str],
    windows_days: List[int] = [30, 90, 180],
    date_col: str = 'timestamp'
) -> Dict[str, pd.DataFrame]:
    """
    Computes rolling mean returns and volatility over specified day-windows.
    """
    out: Dict[str, pd.DataFrame] = {}
    df = df_pct_returns.set_index(date_col)

    for w in windows_days:
        window_str = f'{w}D'
        
        # This calculation remains the same
        rolling_mean = df[columns_to_analyze].rolling(window=window_str, min_periods=max(2, w // 2)).mean()
        

        rolling_vol = df[columns_to_analyze].rolling(window=window_str, min_periods=max(2, w // 2)).std()

        wdf = pd.DataFrame(index=df.index)
        for col in columns_to_analyze:
            wdf[f'{col}_mean_ret_{w}d'] = rolling_mean[col]
            wdf[f'{col}_vol_{w}d'] = rolling_vol[col]

        wdf.reset_index(inplace=True)
        out[f'{w}D_Rolling'] = wdf.dropna()

    return out

def _summarize_rolling_dataframe(rolling_df: pd.DataFrame) -> pd.DataFrame:
    """Summarizes mean, std, min, max of numeric columns in rolling_df."""
    numeric = rolling_df.select_dtypes(include=np.number)
    summary = numeric.agg(['mean', 'std', 'min', 'max']).transpose()
    summary.index.name = 'Rolling Statistic'
    summary.reset_index(inplace=True)
    return summary

def _compute_monthly_returns(
    df_pct_returns: pd.DataFrame,
    columns_to_analyze: List[str],
    date_col: str = 'timestamp'
) -> pd.DataFrame:
    """Calculates month-end returns for each column."""
    df = df_pct_returns.set_index(date_col)
    monthly_last = df[columns_to_analyze].resample('ME').last()
    monthly_ret = monthly_last.pct_change().fillna(0)
    monthly_ret.index = monthly_ret.index.strftime('%Y-%m')
    monthly_ret.reset_index(inplace=True)
    monthly_ret.rename(columns={'index': 'Month'}, inplace=True)
    return monthly_ret

def _compute_return_correlations(
    df_pct_returns: pd.DataFrame,
    columns_to_analyze: List[str],
    date_col: str = 'timestamp'
) -> pd.DataFrame:
    """Computes correlation matrix of daily changes in pct returns."""
    df = df_pct_returns.set_index(date_col)
    diffs = df[columns_to_analyze].diff().dropna()
    return diffs.corr()

# --- Main Reporting Function ---

def generate_backtest_report(
    portfolio: BasePortfolio,
    perf_df: pd.DataFrame,
    initial_capital_per_ticker: float
):
    """
    Generates and saves:
      1) trade_log.csv
      2) performance_timeseries_absolute.csv
      3) summary_metrics.csv
      4) performance_timeseries_percentage.csv
      5) rolling stats & summaries
      6) monthly_returns.csv
      7) ticker_return_correlations.csv
    """
    logger = portfolio.logger
    logger.info("Generating backtest report...")

    if perf_df.empty:
        logger.warning("Performance DataFrame is empty. Skipping report generation")
        return

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("data_infra", "data", f"{run_ts}_backtest_{portfolio.portfolio_id}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Report output directory: {out_dir}")

    # 1) Trade logs
    if isinstance(portfolio.executor, MultiTickerExecutor):
        try:
            logs = portfolio.executor.get_trade_logs()
            all_trades = []
            for t, tlog in logs.items():
                for trade in tlog:
                    trade.setdefault('ticker', t)
                    all_trades.append(trade)
            if all_trades:
                df_trades = pd.DataFrame(all_trades)
                cols = ['timestamp','portfolio_id','ticker','signal_type','shares','fill_price','confidence','cash_after']
                df_trades = df_trades[[c for c in cols if c in df_trades]]
                df_trades.sort_values('timestamp', inplace=True)
                df_trades.to_csv(os.path.join(out_dir, "trade_log.csv"), index=False)
                logger.info("Saved trade_log.csv")
            else:
                logger.info("No trades to log.")
        except Exception as e:
            logger.error(f"Error saving trade logs: {e}", exc_info=True)
    else:
        logger.warning("Executor not MultiTickerExecutor; skipping trade log.")

    # 2)  Save Raw Performance Timeseries (Absolute Values)
    try:
        perf_df.to_csv(os.path.join(out_dir, "performance_timeseries_absolute.csv"), index=False)
        logger.info("Saved performance_timeseries_absolute.csv")
    except Exception as e:
        logger.error(f"Error saving absolute timeseries: {e}", exc_info=True)

    # 3) Calculate and Save Final Summary Metrics
    try:
        metrics_df = aggregate_final_metrics(perf_df)
        if not metrics_df.empty:
            metrics_df.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)
            logger.info("Saved summary_metrics.csv")
    except Exception as e:
        logger.error(f"Error saving summary metrics: {e}", exc_info=True)

    # 4) Calculate Percentage Returns DataFrame
    try:
        df_pct = perf_df.copy()
        cols = []
        for t in portfolio.tickers:
            if t in df_pct:
                pct = (df_pct[t] / initial_capital_per_ticker) - 1.0
                df_pct[f"{t}_pct_ret"] = pct
                cols.append(f"{t}_pct_ret")
        # Convert overall portfolio value to percentage return relative to total initial capital
        total_init = len(portfolio.tickers) * initial_capital_per_ticker
        if 'portfolio_value' in df_pct and total_init > 0:
            df_pct['portfolio_pct_ret'] = (df_pct['portfolio_value'] / total_init) - 1.0
            cols.append('portfolio_pct_ret')

        pct_df = df_pct[['timestamp'] + cols]
        pct_df.to_csv(os.path.join(out_dir, "performance_timeseries_percentage.csv"), index=False)
        logger.info("Saved performance_timeseries_percentage.csv")
    except Exception as e:
        logger.error(f"Error saving percentage timeseries: {e}", exc_info=True)

    # 5) Advanced analytics
    try:
        if not pct_df.empty and len(cols) > 0:
            # rolling
            roll_map = _compute_rolling_stats(pct_df, cols)
            for name, rdf in roll_map.items():
                if not rdf.empty:
                    rdf.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
                    summary = _summarize_rolling_dataframe(rdf)
                    summary.to_csv(os.path.join(out_dir, f"{name}_summary.csv"), index=False)
            # monthly
            mdf = _compute_monthly_returns(pct_df, cols)
            if not mdf.empty:
                mdf.to_csv(os.path.join(out_dir, "monthly_returns.csv"), index=False)
            # correlations
            if len(cols) > 1:
                corr = _compute_return_correlations(pct_df, cols)
                corr.to_csv(os.path.join(out_dir, "ticker_return_correlations.csv"))
    except Exception as e:
        logger.error(f"Error in advanced analytics: {e}", exc_info=True)

    logger.info("Backtest report generation complete.")
