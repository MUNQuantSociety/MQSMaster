# data_infra/tradingOps/backtest/reporting.py

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Optional

# Assuming portfolio has portfolio.tickers, portfolio.executor (which is MultiTickerExecutor during saving)
from portfolios.portfolio_BASE.strategy import BasePortfolio
# Explicitly import from MultiTickerExecutor if type hinting is desired
from .multiTickerExecutor import MultiTickerExecutor

# --- Core Metric Calculations (Moved from utils.py) ---

def _compute_max_drawdown(portfolio_values: pd.Series) -> float:
    """Calculates the maximum drawdown from a series of portfolio values."""
    if len(portfolio_values) < 2: return 0.0
    arr = portfolio_values.to_numpy(dtype=float)
    # Replace non-finite values (like NaN or inf) with a default value (e.g., previous value or 0)
    # Or handle them based on specific requirements. Here, let's forward fill NaNs.
    arr = pd.Series(arr).ffill().bfill().to_numpy() # Fill NaNs first
    if not np.all(np.isfinite(arr)):
         logging.warning("Non-finite values found in portfolio values for drawdown calc, returning 0.0")
         return 0.0
    if np.any(arr <= 0): # Avoid division by zero or negative peaks
         logging.warning("Non-positive portfolio values found, drawdown calculation might be inaccurate.")
         # Potentially adjust values or return 0.0
         arr[arr <= 0] = 1e-9 # Replace non-positive with small number

    peak = np.maximum.accumulate(arr)
    # Ensure peak is never zero or negative before division
    peak[peak <= 0] = 1e-9
    dd = (arr - peak) / peak
    max_dd = float(np.min(dd))
    # Ensure drawdown is not positive (can happen with precision issues or all zero values)
    return min(max_dd, 0.0)


def _compute_sharpe_ratio(portfolio_values: pd.Series, periods_per_year: int = 252) -> float:
    """Calculates the annualized Sharpe ratio from portfolio values (assuming daily data)."""
    if len(portfolio_values) < 3: return 0.0 # Need at least 3 points for 2 returns
    # Calculate daily returns from portfolio values
    daily_returns = portfolio_values.pct_change().dropna()

    if daily_returns.empty or len(daily_returns) < 2: return 0.0

    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns)

    if std_ret == 0: return 0.0 # Avoid division by zero

    # Annualize (default assumes daily data -> 252 trading days)
    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year)
    return float(sharpe)


def aggregate_final_metrics(perf_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates key performance indicators from the performance DataFrame."""
    if perf_df.empty or 'portfolio_value' not in perf_df.columns:
        return pd.DataFrame(columns=['metric', 'value'])

    final_val = perf_df['portfolio_value'].iloc[-1]
    max_dd = _compute_max_drawdown(perf_df['portfolio_value'])
    sharpe = _compute_sharpe_ratio(perf_df['portfolio_value']) # Use value series for Sharpe

    # Add more metrics as needed (e.g., CAGR, Volatility)
    # cagr = _compute_cagr(perf_df)
    # volatility = _compute_annualized_volatility(perf_df['portfolio_value'])

    summary = pd.DataFrame({
        'metric': ['Final Portfolio Value', 'Max Drawdown (%)', 'Annualized Sharpe Ratio'],
        'value': [f"{final_val:,.2f}", f"{max_dd:.2%}", f"{sharpe:.3f}"] # Format values for readability
        # Add other metrics here
    })
    return summary


# --- Advanced Analytics Calculations ---

def _compute_rolling_stats(df_pct_returns: pd.DataFrame,
                           columns_to_analyze: List[str],
                           windows_days: List[int] = [30, 90, 180],
                           date_col: str = 'timestamp') -> Dict[str, pd.DataFrame]:
    """
    Computes rolling returns and volatility over specified windows.
    Assumes input DataFrame contains percentage returns relative to start for each column.
    """
    out_map = {}
    df = df_pct_returns.set_index(date_col) # Set timestamp as index for rolling

    for w_days in windows_days:
        # Use rolling based on time offset (more robust than fixed number of periods)
        window_str = f'{w_days}D'
        rolling_results = {}

        # Calculate rolling mean return (average percentage return over the window)
        rolling_mean = df[columns_to_analyze].rolling(window=window_str, min_periods=max(2, w_days // 2)).mean()

        # Calculate rolling volatility (std dev of *daily* changes in percentage return)
        daily_pct_change = df[columns_to_analyze].diff() # Daily change in the percentage return series
        rolling_vol = daily_pct_change.rolling(window=window_str, min_periods=max(2, w_days // 2)).std()

        # Combine results for this window
        wdf = pd.DataFrame(index=df.index)
        for col in columns_to_analyze:
            wdf[f'{col}_mean_ret_{w_days}d'] = rolling_mean[col]
            wdf[f'{col}_vol_{w_days}d'] = rolling_vol[col]

        wdf.reset_index(inplace=True) # Restore timestamp column
        out_map[f'{w_days}D_Rolling'] = wdf.dropna() # Drop rows where rolling window wasn't full enough

    return out_map


def _summarize_rolling_dataframe(rolling_df: pd.DataFrame) -> pd.DataFrame:
    """Summarizes a rolling stats DataFrame (mean, std, min, max of each column)."""
    numeric_cols = rolling_df.select_dtypes(include=np.number).columns
    summary = rolling_df[numeric_cols].agg(['mean', 'std', 'min', 'max']).transpose()
    summary.index.name = 'Rolling Statistic'
    summary.reset_index(inplace=True)
    return summary

def _compute_monthly_returns(df_pct_returns: pd.DataFrame,
                             columns_to_analyze: List[str],
                             date_col: str = 'timestamp') -> pd.DataFrame:
    """Calculates monthly returns based on the end-of-month percentage return."""
    df = df_pct_returns.set_index(date_col)
    # Resample to month-end, taking the last value of the percentage return
    monthly_pct = df[columns_to_analyze].resample('ME').last()
    # Calculate monthly return as the percentage change from the previous month's end value
    monthly_returns = monthly_pct.pct_change().fillna(0) # Fill first month's NaN return with 0

    # Format the index as YYYY-MM
    monthly_returns.index = monthly_returns.index.strftime('%Y-%m')
    monthly_returns.reset_index(inplace=True)
    monthly_returns.rename(columns={'index': 'Month'}, inplace=True)
    return monthly_returns


def _compute_return_correlations(df_pct_returns: pd.DataFrame,
                                 columns_to_analyze: List[str],
                                 date_col: str = 'timestamp') -> pd.DataFrame:
    """Calculates the correlation matrix of daily changes in percentage returns."""
    df = df_pct_returns.set_index(date_col)
    # Calculate daily *changes* in the percentage return series
    daily_diffs = df[columns_to_analyze].diff().dropna()
    correlation_matrix = daily_diffs.corr()
    return correlation_matrix


# --- Main Reporting Function ---

def generate_backtest_report(portfolio: BasePortfolio,
                             perf_df: pd.DataFrame,
                             initial_capital_per_ticker: float):
    """
    Generates and saves a comprehensive backtest report including trade logs,
    performance metrics, and advanced analytics.

    Args:
        portfolio: The portfolio instance (used for ID, tickers, executor access).
        perf_df: DataFrame containing the timestamped portfolio values and ticker values.
        initial_capital_per_ticker: The initial capital used per ticker.
    """
    logger = portfolio.logger
    logger.info("--- Generating Backtest Report ---")

    if perf_df.empty:
        logger.warning("Performance DataFrame is empty. Skipping report generation.")
        return

    # Create output directory
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use portfolio ID in the folder name for better organization
    out_dir = os.path.join("data_infra", "data", f"{run_ts}_backtest_{portfolio.portfolio_id}")
    try:
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"Report output directory: {out_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory {out_dir}: {e}")
        return # Cannot proceed without output directory

    # 1) Save Trade Logs
    try:
        # portfolio.executor should be the MultiTickerExecutor instance here
        if isinstance(portfolio.executor, MultiTickerExecutor):
            logs_dict = portfolio.executor.get_trade_logs()
            combined_trades = []
            for ticker, tlog in logs_dict.items():
                # Add ticker column if not present in individual logs (it should be)
                for trade in tlog:
                    trade['ticker'] = ticker # Ensure ticker association
                combined_trades.extend(tlog)

            if combined_trades:
                trades_df = pd.DataFrame(combined_trades)
                # Reorder columns for clarity
                cols_order = ['timestamp', 'portfolio_id', 'ticker', 'signal_type',
                              'shares', 'fill_price', 'confidence', 'cash_after']
                trades_df = trades_df[[col for col in cols_order if col in trades_df.columns]]
                trades_df.sort_values('timestamp', inplace=True)
                log_path = os.path.join(out_dir, "trade_log.csv")
                trades_df.to_csv(log_path, index=False, float_format='%.4f')
                logger.info(f"Trade log saved to {log_path}")
            else:
                logger.info("No trades executed during backtest, trade log not saved.")
        else:
             logger.warning("Executor is not MultiTickerExecutor, cannot get detailed trade logs.")

    except Exception as e:
        logger.error(f"Error saving trade logs: {e}", exc_info=True)


    # 2) Save Raw Performance Timeseries (Absolute Values)
    try:
        perf_path = os.path.join(out_dir, "performance_timeseries_absolute.csv")
        perf_df.to_csv(perf_path, index=False, float_format='%.2f')
        logger.info(f"Absolute performance timeseries saved to {perf_path}")
    except Exception as e:
        logger.error(f"Error saving performance timeseries: {e}", exc_info=True)

    # 3) Calculate and Save Final Summary Metrics
    try:
        final_metrics_df = aggregate_final_metrics(perf_df)
        if not final_metrics_df.empty:
            metrics_path = os.path.join(out_dir, "summary_metrics.csv")
            final_metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Summary metrics saved to {metrics_path}")
        else:
            logger.warning("Could not calculate summary metrics.")
    except Exception as e:
        logger.error(f"Error calculating/saving summary metrics: {e}", exc_info=True)

    # 4) Calculate Percentage Returns DataFrame
    df_pct = perf_df.copy()
    tickers = portfolio.tickers
    analysis_cols = [] # Columns containing percentage return data

    # Convert individual ticker absolute values to percentage returns
    for t in tickers:
        if t in df_pct.columns:
            # Calculate % return relative to the specific ticker's initial capital
            df_pct[f'{t}_pct_ret'] = (df_pct[t] / initial_capital_per_ticker) - 1.0
            analysis_cols.append(f'{t}_pct_ret')
        else:
            logger.warning(f"Ticker column '{t}' not found in perf_df for percentage calculation.")

    # Convert overall portfolio value to percentage return relative to total initial capital
    total_initial_capital = len(tickers) * initial_capital_per_ticker
    if 'portfolio_value' in df_pct.columns and total_initial_capital > 0:
        df_pct['portfolio_pct_ret'] = (df_pct['portfolio_value'] / total_initial_capital) - 1.0
        analysis_cols.append('portfolio_pct_ret')
    else:
         logger.warning("Could not calculate overall portfolio percentage return.")

    # Select relevant columns for percentage return DataFrame
    pct_cols_to_keep = ['timestamp'] + analysis_cols
    df_pct_returns = df_pct[pct_cols_to_keep].copy()

    # Save Percentage Performance Timeseries
    try:
        perf_pct_path = os.path.join(out_dir, "performance_timeseries_percentage.csv")
        df_pct_returns.to_csv(perf_pct_path, index=False, float_format='%.6f')
        logger.info(f"Percentage performance timeseries saved to {perf_pct_path}")
    except Exception as e:
        logger.error(f"Error saving percentage performance timeseries: {e}", exc_info=True)


    # 5) Advanced Analytics (using df_pct_returns)
    if not df_pct_returns.empty and len(analysis_cols) > 0:
        logger.info("Calculating advanced analytics...")
        # a) Rolling Windows Stats
        try:
            rolling_stats_map = _compute_rolling_stats(df_pct_returns, analysis_cols, windows_days=[30, 90, 180])
            for name, df_rolling in rolling_stats_map.items():
                if not df_rolling.empty:
                    r_path = os.path.join(out_dir, f"{name}_stats.csv")
                    df_rolling.to_csv(r_path, index=False, float_format='%.6f')
                    # Summarize the rolling data
                    summ_rolling = _summarize_rolling_dataframe(df_rolling)
                    sr_path = os.path.join(out_dir, f"{name}_summary.csv")
                    summ_rolling.to_csv(sr_path, index=False, float_format='%.6f')
            logger.info("Rolling statistics calculated and saved.")
        except Exception as e:
            logger.error(f"Error calculating/saving rolling statistics: {e}", exc_info=True)

        # b) Monthly Returns
        try:
            monthly_returns_df = _compute_monthly_returns(df_pct_returns, analysis_cols)
            if not monthly_returns_df.empty:
                m_path = os.path.join(out_dir, "monthly_returns.csv")
                monthly_returns_df.to_csv(m_path, index=False, float_format='%.6f')
                logger.info("Monthly returns calculated and saved.")
        except Exception as e:
            logger.error(f"Error calculating/saving monthly returns: {e}", exc_info=True)

        # c) Return Correlations
        try:
            # Select only ticker returns for correlation, exclude overall portfolio return
            ticker_pct_ret_cols = [col for col in analysis_cols if col != 'portfolio_pct_ret']
            if len(ticker_pct_ret_cols) > 1: # Need at least 2 tickers for correlation
                correlation_matrix = _compute_return_correlations(df_pct_returns, ticker_pct_ret_cols)
                c_path = os.path.join(out_dir, "ticker_return_correlations.csv")
                correlation_matrix.to_csv(c_path, float_format='%.4f')
                logger.info("Ticker return correlations calculated and saved.")
            else:
                logger.info("Skipping correlation calculation: less than 2 tickers.")
        except Exception as e:
            logger.error(f"Error calculating/saving return correlations: {e}", exc_info=True)
    else:
        logger.warning("Skipping advanced analytics due to empty percentage returns data.")

    logger.info("--- Backtest Report Generation Finished ---")