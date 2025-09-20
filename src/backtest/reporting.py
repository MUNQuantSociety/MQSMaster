import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from portfolios.portfolio_BASE.strategy import BasePortfolio
from .executor import BacktestExecutor


# --- Core Metric Calculations (Unchanged) ---

def _compute_max_drawdown(portfolio_values: pd.Series) -> float:
    """Calculates the maximum drawdown from a series of portfolio values."""
    if len(portfolio_values) < 2:
        return 0.0
    arr = pd.Series(portfolio_values).ffill().bfill().to_numpy(dtype=float)
    if not np.all(np.isfinite(arr)):
        logging.warning("Non-finite values in portfolio values; drawdown set to 0.0")
        return 0.0
    arr[arr <= 0] = 1e-9
    peak = np.maximum.accumulate(arr)
    drawdowns = (arr - peak) / peak
    return float(np.min(drawdowns))

def _compute_sharpe_ratio(perf_df: pd.DataFrame) -> float:
    """Calculates the annualized Sharpe ratio from a performance DataFrame."""
    if perf_df.empty or 'timestamp' not in perf_df or 'portfolio_value' not in perf_df:
        return 0.0
    df = perf_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    daily_values = df['portfolio_value'].resample('D').last()
    
    daily_values_filled = daily_values.ffill()
    daily_returns = daily_values_filled.pct_change().dropna()
    
    if daily_returns.empty or len(daily_returns) < 2:
        return 0.0
    mean_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    if std_daily_return == 0 or np.isnan(std_daily_return):
        return 0.0
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
        'value': [f"{final_val:,.2f}", f"{max_dd:.2%}", f"{sharpe:.3f}"]
    })
    return summary

# --- OPTIMIZED High-Frequency and Benchmark Reporting Helpers (Unchanged) ---

def _generate_minute_by_minute_performance(
    trade_log: List[Dict],
    full_historical_data: pd.DataFrame,
    initial_capital: float,
    tickers: List[str]
) -> pd.DataFrame:
    """
    Generates a minute-by-minute performance report using vectorized operations.
    """
    if full_historical_data.empty:
        return pd.DataFrame()

    price_pivot = full_historical_data.pivot(index='timestamp', columns='ticker', values='close_price')
    minute_prices = price_pivot.resample('min').ffill().bfill()

    if not trade_log:
        output_df = pd.DataFrame(index=minute_prices.index)
        output_df['portfolio_value'] = initial_capital
        output_df['cash_value'] = initial_capital
        for ticker in tickers:
             if ticker in minute_prices.columns:
                output_df[f'{ticker}_value'] = 0.0
        return output_df.reset_index()

    trades_df = pd.DataFrame(trade_log)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    trades_df['cash_change'] = np.where(
        trades_df['signal_type'] == 'BUY',
        -trades_df['shares'] * trades_df['fill_price'],
        trades_df['shares'] * trades_df['fill_price']
    )
    trades_df['position_change'] = np.where(
        trades_df['signal_type'] == 'BUY',
        trades_df['shares'],
        -trades_df['shares']
    )

    position_changes = trades_df.pivot_table(
        index='timestamp', columns='ticker', values='position_change', aggfunc='sum'
    ).fillna(0)
    
    cash_changes = trades_df.groupby('timestamp')['cash_change'].sum()
    
    cumulative_positions = position_changes.cumsum()
    cumulative_cash = initial_capital + cash_changes.cumsum()

    all_timestamps = minute_prices.index
    minute_positions = cumulative_positions.reindex(all_timestamps).ffill().fillna(0)
    
    minute_cash = cumulative_cash.reindex(all_timestamps).ffill()
    if pd.isna(minute_cash.iloc[0]):
        minute_cash.iloc[0] = initial_capital
    minute_cash = minute_cash.ffill()
    
    output_df = pd.DataFrame(index=minute_prices.index)
    aligned_tickers = [ticker for ticker in tickers if ticker in minute_prices.columns]

    for ticker in aligned_tickers:
        output_df[f'{ticker}_value'] = minute_positions.get(ticker, 0) * minute_prices.get(ticker, 0)

    holdings_value = output_df.sum(axis=1)
    output_df['cash_value'] = minute_cash
    output_df['portfolio_value'] = holdings_value + minute_cash

    return output_df.reset_index()


def _generate_buy_and_hold_benchmark(
    full_historical_data: pd.DataFrame,
    initial_capital: float,
    portfolio_weights: Dict[str, float]
) -> pd.DataFrame:
    """
    CORRECTED: Generates a robust minute-by-minute benchmark report that accounts
    for uninvested capital, ensuring the starting value is always correct.
    """
    if full_historical_data.empty or not portfolio_weights:
        return pd.DataFrame()

    price_pivot = full_historical_data.pivot(index='timestamp', columns='ticker', values='close_price')
    minute_prices = price_pivot.resample('min').ffill().bfill()
    
    first_day_prices = minute_prices.iloc[0]
    initial_shares = pd.Series(index=portfolio_weights.keys(), dtype=float)
    
    # Calculate the initial capital that is actually invested into assets.
    total_investment = 0.0
    for ticker, weight in portfolio_weights.items():
        if ticker in first_day_prices and first_day_prices[ticker] > 0:
            investment_amount = initial_capital * weight
            initial_shares[ticker] = investment_amount / first_day_prices[ticker]
            total_investment += investment_amount
            
    initial_shares = initial_shares.fillna(0)
    
    # Calculate the portion of capital that remains as cash.
    initial_cash_held = initial_capital - total_investment
    
    # Calculate the value of the asset holdings over time.
    aligned_tickers = [ticker for ticker in initial_shares.index if ticker in minute_prices.columns]
    benchmark_asset_values = minute_prices[aligned_tickers].dot(initial_shares[aligned_tickers])
    
    # The total benchmark value is the fluctuating asset value plus the fixed cash held.
    benchmark_total_values = benchmark_asset_values + initial_cash_held
    
    benchmark_df = pd.DataFrame({'timestamp': minute_prices.index, 'buy_and_hold_value': benchmark_total_values})
    benchmark_df['buy_and_hold_return'] = (benchmark_df['buy_and_hold_value'] / initial_capital) - 1.0
    
    return benchmark_df

# --- Advanced Analytics Calculations (Unchanged) ---

def _compute_rolling_stats(
    df_pct_returns: pd.DataFrame,
    columns_to_analyze: List[str],
    windows_days: List[int] = [30, 90, 180],
    date_col: str = 'timestamp'
) -> Dict[str, pd.DataFrame]:
    """Computes rolling statistics with a full window buffer."""
    out: Dict[str, pd.DataFrame] = {}
    df = df_pct_returns.set_index(date_col)
    for w in windows_days:
        window_str = f'{w}D'
        rolling_mean = df[columns_to_analyze].rolling(window=window_str, min_periods=w).mean()
        rolling_vol = df[columns_to_analyze].rolling(window=window_str, min_periods=w).std()
        wdf = pd.DataFrame(index=df.index)
        for col in columns_to_analyze:
            wdf[f'{col}_mean_ret_{w}d'] = rolling_mean[col]
            wdf[f'{col}_vol_{w}d'] = rolling_vol[col]
        wdf.reset_index(inplace=True)
        out[f'{w}D_Rolling'] = wdf.dropna()
    return out

def _summarize_rolling_dataframe(rolling_df: pd.DataFrame) -> pd.DataFrame:
    """Summarizes a rolling statistics DataFrame."""
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
    """Computes monthly returns from a DataFrame of daily percentage returns."""
    df = df_pct_returns.set_index(date_col)
    monthly_last = df[columns_to_analyze].resample('ME').last()

    monthly_last_filled = monthly_last.ffill()
    monthly_ret = monthly_last_filled.pct_change().fillna(0)
    
    monthly_ret.index = monthly_ret.index.strftime('%Y-%m')
    monthly_ret.reset_index(inplace=True)
    monthly_ret.rename(columns={'index': 'Month'}, inplace=True)
    return monthly_ret

def _compute_return_correlations(
    df_pct_returns: pd.DataFrame,
    columns_to_analyze: List[str],
    date_col: str = 'timestamp'
) -> pd.DataFrame:
    """Computes the correlation matrix for specified columns."""
    df = df_pct_returns.set_index(date_col)
    return df[columns_to_analyze].corr()

# --- Portfolio Risk Calculation Helpers (Unchanged) ---

def _calculate_portfolio_risk_components(
    full_historical_data: pd.DataFrame,
    portfolio_weights: Dict[str, float]
) -> (pd.DataFrame, pd.Series, pd.DataFrame):
    """
    Calculates risk components, returning the correlation matrix,
    individual volatilities, and weights DataFrame separately.
    """
    if full_historical_data.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()
        
    price_pivot = full_historical_data.pivot(index='timestamp', columns='ticker', values='close_price')
    price_pivot_filled = price_pivot.ffill()
    daily_returns = price_pivot_filled.pct_change().dropna()

    if daily_returns.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

    aligned_tickers = [ticker for ticker in portfolio_weights.keys() if ticker in daily_returns.columns]
    daily_returns = daily_returns[aligned_tickers]

    annualized_corr_matrix = daily_returns.corr()
    annualized_volatilities = daily_returns.std() * np.sqrt(252)
    weights_df = pd.DataFrame(list(portfolio_weights.items()), columns=['ticker', 'weight'])
    
    return annualized_corr_matrix, annualized_volatilities, weights_df


def _calculate_rolling_portfolio_risk(
    full_historical_data: pd.DataFrame,
    portfolio_weights: Dict[str, float],
    window_days: int = 30
) -> pd.DataFrame:
    """Calculates the rolling portfolio risk with a full window buffer."""
    if full_historical_data.empty:
        return pd.DataFrame()
        
    price_pivot = full_historical_data.pivot(index='timestamp', columns='ticker', values='close_price')
    price_pivot_filled = price_pivot.ffill()
    daily_returns = price_pivot_filled.pct_change().dropna()

    if len(daily_returns) < window_days:
        return pd.DataFrame()
        
    weights = pd.Series(portfolio_weights)
    aligned_tickers = [ticker for ticker in weights.index if ticker in daily_returns.columns]
    weights = weights[aligned_tickers]
    daily_returns = daily_returns[aligned_tickers]

    rolling_cov = daily_returns.rolling(window=window_days, min_periods=window_days).cov() * 252
    rolling_cov = rolling_cov.dropna()
    
    if rolling_cov.empty:
        return pd.DataFrame()

    rolling_portfolio_variance = rolling_cov.groupby(level='timestamp').apply(
        lambda cov_matrix: np.dot(
            weights.T, 
            np.dot(
                cov_matrix.droplevel('timestamp').loc[weights.index, weights.index].values, 
                weights
            )
        )
    )
    
    rolling_portfolio_risk = np.sqrt(rolling_portfolio_variance)
    
    return pd.DataFrame({
        'timestamp': rolling_portfolio_risk.index, 
        f'rolling_{window_days}d_portfolio_risk': rolling_portfolio_risk
    })


# --- Main Reporting Function (CORRECTED) ---
def generate_backtest_report(
    portfolio: BasePortfolio,
    perf_df: pd.DataFrame,
    initial_capital: float,
    full_historical_data: pd.DataFrame
):
    """
    Generates and saves a full backtest report with enhanced risk analysis.
    """
    logger = portfolio.logger
    logger.info("Generating backtest report...")
    if perf_df.empty:
        logger.warning("Performance DataFrame is empty. Skipping report generation")
        return
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("src", "backtest", "data", f"{run_ts}_backtest_{portfolio.portfolio_id}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Report output directory: {out_dir}")

    # Section 1: Trade logs
    if isinstance(portfolio.executor, BacktestExecutor):
        try:
            all_trades = portfolio.executor.trade_log
            if all_trades:
                df_trades = pd.DataFrame(all_trades)
                cols = ['timestamp','portfolio_id','ticker','signal_type','shares','fill_price','confidence','cash_after']
                df_trades = df_trades[[c for c in cols if c in df_trades.columns]]
                df_trades.sort_values('timestamp', inplace=True)
                df_trades.to_csv(os.path.join(out_dir, "trade_log.csv"), index=False)
        except Exception as e:
            logger.error(f"Error saving trade logs: {e}", exc_info=True)
    
    # Section 2: Raw Performance Timeseries
    perf_df.to_csv(os.path.join(out_dir, "performance_timeseries_absolute.csv"), index=False)
    
    # Section 3: Final Summary Metrics
    metrics_df = aggregate_final_metrics(perf_df)
    metrics_df.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)
    
    # Section 4: Percentage Returns DataFrame
    pct_df = pd.DataFrame()
    if 'portfolio_value' in perf_df and initial_capital > 0:
        pct_df = perf_df[['timestamp']].copy()
        pct_df['portfolio_pct_ret'] = (perf_df['portfolio_value'] / initial_capital) - 1.0
        pct_df.to_csv(os.path.join(out_dir, "performance_timeseries_percentage.csv"), index=False)

    # Section 5: High-frequency performance report
    try:
        minute_perf_df = _generate_minute_by_minute_performance(
            trade_log=portfolio.executor.trade_log,
            full_historical_data=full_historical_data,
            initial_capital=initial_capital,
            tickers=portfolio.tickers
        )
        if not minute_perf_df.empty:
            minute_perf_df.to_csv(os.path.join(out_dir, "performance_timeseries_minute_by_minute.csv"), index=False)
    except Exception as e:
        logger.error(f"Error generating minute-by-minute performance report: {e}", exc_info=True)
    
    # Section 6: Buy-and-hold benchmark report
    try:
        benchmark_df = _generate_buy_and_hold_benchmark(
            full_historical_data=full_historical_data,
            initial_capital=initial_capital,
            portfolio_weights=portfolio.portfolio_weights
        )
        if not benchmark_df.empty:
            benchmark_df.to_csv(os.path.join(out_dir, "benchmark_buy_and_hold_performance.csv"), index=False)
    except Exception as e:
        logger.error(f"Error generating buy-and-hold benchmark report: {e}", exc_info=True)
        
    # Section 7: Advanced Analytics
    try:
        if not pct_df.empty:
            cols_to_analyze = ['portfolio_pct_ret']
            roll_map = _compute_rolling_stats(pct_df, cols_to_analyze)
            for name, rdf in roll_map.items():
                rdf.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
                _summarize_rolling_dataframe(rdf).to_csv(os.path.join(out_dir, f"{name}_summary.csv"), index=False)
            mdf = _compute_monthly_returns(pct_df, cols_to_analyze)
            mdf.to_csv(os.path.join(out_dir, "monthly_returns.csv"), index=False)
            if len(cols_to_analyze) > 1:
                 _compute_return_correlations(pct_df, cols_to_analyze).to_csv(os.path.join(out_dir, "portfolio_return_correlations.csv"))
    except Exception as e:
        logger.error(f"Error in advanced analytics: {e}", exc_info=True)

    # Section 8: Portfolio Risk Analytics
    try:
        corr_matrix, indiv_vols, weights_df = _calculate_portfolio_risk_components(
            full_historical_data, portfolio.portfolio_weights
        )
        if not corr_matrix.empty:
            aligned_weights_df = weights_df[weights_df['ticker'].isin(corr_matrix.columns)]
            
            risk_components_summary = pd.concat(
                [aligned_weights_df.set_index('ticker'), indiv_vols.rename('annualized_volatility')], 
                axis=1
            )
            
            risk_components_summary.to_csv(os.path.join(out_dir, "portfolio_risk_components.csv"))
            corr_matrix.to_csv(os.path.join(out_dir, "annualized_correlation_matrix.csv"))
            logger.info("Saved portfolio_risk_components.csv and annualized_correlation_matrix.csv")

        rolling_risk_df = _calculate_rolling_portfolio_risk(
            full_historical_data, portfolio.portfolio_weights
        )
        if not rolling_risk_df.empty:
            rolling_risk_df.to_csv(os.path.join(out_dir, "rolling_portfolio_risk.csv"), index=False)
            logger.info("Saved rolling_portfolio_risk.csv")

    except Exception as e:
        logger.error(f"Error in portfolio risk analytics: {e}", exc_info=True)

    logger.info("Backtest report generation complete.")
