import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.portfolios.portfolio_BASE.strategy import BasePortfolio

from .executor import BacktestExecutor

# --- Core Metric Calculations (Unchanged) ---


def _compute_max_drawdown(portfolio_values: pd.Series) -> float:
    """Calculates the maximum drawdown from a series of portfolio values."""
    if len(portfolio_values) < 2:
        return 0.0
    arr = pd.Series(portfolio_values).ffill().bfill().to_numpy(dtype=float).copy()
    if not np.all(np.isfinite(arr)):
        logging.warning("Non-finite values in portfolio values; drawdown set to 0.0")
        return 0.0
    arr[arr <= 0] = 1e-9
    peak = np.maximum.accumulate(arr)
    drawdowns = (arr - peak) / peak
    return float(np.min(drawdowns))


def _compute_sharpe_ratio(perf_df: pd.DataFrame) -> float:
    """Calculates the annualized Sharpe ratio from a performance DataFrame."""
    if perf_df.empty or "timestamp" not in perf_df or "portfolio_value" not in perf_df:
        return 0.0
    df = perf_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    daily_values = df["portfolio_value"].resample("D").last()

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


def _compute_annual_return(perf_df: pd.DataFrame) -> float:
    """Calculates annualized return (CAGR) from a performance DataFrame."""
    if perf_df.empty or "timestamp" not in perf_df or "portfolio_value" not in perf_df:
        return 0.0

    df = perf_df[["timestamp", "portfolio_value"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["portfolio_value"] = pd.to_numeric(df["portfolio_value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "portfolio_value"]).sort_values("timestamp")

    if len(df) < 2:
        return 0.0

    start_value = float(df["portfolio_value"].iloc[0])
    end_value = float(df["portfolio_value"].iloc[-1])

    if not np.isfinite(start_value) or not np.isfinite(end_value) or start_value <= 0:
        return 0.0

    elapsed_days: float = ((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400.0)
    if elapsed_days <= 0:
        return 0.0

    ratio = end_value / start_value
    if ratio <= 0:
        return -1.0
    annual_return = ratio ** (365.25 / elapsed_days) - 1.0
    if not np.isfinite(annual_return):
        return 0.0
    return annual_return


def aggregate_final_metrics(perf_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates key metrics from perf_df."""
    if perf_df.empty or "portfolio_value" not in perf_df:
        return pd.DataFrame(columns=["metric", "value"])
    final_val = perf_df["portfolio_value"].iloc[-1]
    max_dd = _compute_max_drawdown(perf_df["portfolio_value"])
    annual_return = _compute_annual_return(perf_df)
    sharpe = _compute_sharpe_ratio(perf_df)
    summary = pd.DataFrame(
        {
            "metric": [
                "Final Portfolio Value",
                "Max Drawdown (%)",
                "Annual Return",
                "Annualized Sharpe Ratio",
            ],
            "value": [
                f"{final_val:,.2f}",
                f"{max_dd:.2%}",
                f"{annual_return:.2%}",
                f"{sharpe:.3f}",
            ],
        }
    )
    return summary


# --- OPTIMIZED High-Frequency and Benchmark Reporting Helpers (Unchanged) ---


MINUTE_RESAMPLE_CELL_LIMIT = 5_000_000


def _minute_resample_too_large(price_pivot: pd.DataFrame) -> bool:
    if price_pivot.empty:
        return False
    span_minutes = int(
        (price_pivot.index.max() - price_pivot.index.min()).total_seconds() // 60
    )
    return span_minutes * len(price_pivot.columns) > MINUTE_RESAMPLE_CELL_LIMIT


def _generate_minute_by_minute_performance(
    trade_log: list[dict],
    full_historical_data: pd.DataFrame,
    initial_capital: float,
    tickers: list[str],
) -> pd.DataFrame:
    """
    Generates a minute-by-minute performance report using vectorized operations.
    """
    if full_historical_data.empty:
        return pd.DataFrame()

    price_pivot = full_historical_data.pivot(
        index="timestamp", columns="ticker", values="close_price"
    )
    price_pivot.index = pd.to_datetime(price_pivot.index, errors="coerce")
    price_pivot = price_pivot[price_pivot.index.notna()].sort_index()
    if _minute_resample_too_large(price_pivot):
        logging.warning(
            "Skipping minute-by-minute performance: %d tickers x %d-min span exceeds %d-cell limit",
            len(price_pivot.columns),
            int((price_pivot.index.max() - price_pivot.index.min()).total_seconds() // 60),
            MINUTE_RESAMPLE_CELL_LIMIT,
        )
        return pd.DataFrame()
    minute_prices = price_pivot.resample("min").ffill().bfill()

    if not trade_log:
        output_df = pd.DataFrame(index=minute_prices.index)
        output_df["portfolio_value"] = initial_capital
        output_df["cash_value"] = initial_capital
        for ticker in tickers:
            if ticker in minute_prices.columns:
                output_df[f"{ticker}_value"] = 0.0
        return output_df.reset_index()

    trades_df = pd.DataFrame(trade_log)
    trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])

    trades_df["cash_change"] = np.where(
        trades_df["signal_type"] == "BUY",
        -trades_df["shares"] * trades_df["fill_price"],
        trades_df["shares"] * trades_df["fill_price"],
    )
    trades_df["position_change"] = np.where(
        trades_df["signal_type"] == "BUY", trades_df["shares"], -trades_df["shares"]
    )

    position_changes = trades_df.pivot_table(
        index="timestamp", columns="ticker", values="position_change", aggfunc="sum"
    ).fillna(0)

    cash_changes = trades_df.groupby("timestamp")["cash_change"].sum()

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
        output_df[f"{ticker}_value"] = minute_positions.get(
            ticker, 0
        ) * minute_prices.get(ticker, 0)

    holdings_value = output_df.sum(axis=1)
    output_df["cash_value"] = minute_cash
    output_df["portfolio_value"] = holdings_value + minute_cash

    return output_df.reset_index()


def _generate_buy_and_hold_benchmark(
    full_historical_data: pd.DataFrame,
    initial_capital: float,
    portfolio_weights: dict[str, float],
) -> pd.DataFrame:
    """
    CORRECTED: Generates a robust minute-by-minute benchmark report that accounts
    for uninvested capital, ensuring the starting value is always correct.
    """
    if full_historical_data.empty or not portfolio_weights:
        return pd.DataFrame()

    price_pivot = full_historical_data.pivot(
        index="timestamp", columns="ticker", values="close_price"
    )
    price_pivot.index = pd.to_datetime(price_pivot.index, errors="coerce")
    price_pivot = price_pivot[price_pivot.index.notna()].sort_index()
    if _minute_resample_too_large(price_pivot):
        logging.warning(
            "Skipping buy-and-hold benchmark: %d tickers x %d-min span exceeds %d-cell limit",
            len(price_pivot.columns),
            int((price_pivot.index.max() - price_pivot.index.min()).total_seconds() // 60),
            MINUTE_RESAMPLE_CELL_LIMIT,
        )
        return pd.DataFrame()
    minute_prices = price_pivot.resample("min").ffill().bfill()

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
    aligned_tickers = [
        ticker for ticker in initial_shares.index if ticker in minute_prices.columns
    ]
    benchmark_asset_values = minute_prices[aligned_tickers].dot(
        initial_shares[aligned_tickers]
    )

    # The total benchmark value is the fluctuating asset value plus the fixed cash held.
    benchmark_total_values = benchmark_asset_values + initial_cash_held

    benchmark_df = pd.DataFrame(
        {"timestamp": minute_prices.index, "buy_and_hold_value": benchmark_total_values}
    )
    benchmark_df["buy_and_hold_return"] = (
        benchmark_df["buy_and_hold_value"] / initial_capital
    ) - 1.0

    return benchmark_df


# --- Advanced Analytics Calculations (Unchanged) ---


def _compute_rolling_stats(
    df_pct_returns: pd.DataFrame,
    columns_to_analyze: list[str],
    windows_days: list[int] = [30, 90, 180],
    date_col: str = "timestamp",
) -> dict[str, pd.DataFrame]:
    """
    Computes rolling statistics allowing partial-window estimates
    (min_periods = w // 2), so results begin once at least half of
    the window has data.
    """
    out: dict[str, pd.DataFrame] = {}
    df = df_pct_returns.set_index(date_col)
    for w in windows_days:
        window_str = f"{w}D"
        rolling_mean = (
            df[columns_to_analyze]
            .rolling(window=window_str, min_periods=w // 2).mean()
        )
        rolling_vol = (
            df[columns_to_analyze]
            .rolling(window=window_str, min_periods=w // 2).std()
        )
        wdf = pd.DataFrame(index=df.index)
        for col in columns_to_analyze:
            wdf[f"{col}_mean_ret_{w}d"] = rolling_mean[col]
            wdf[f"{col}_vol_{w}d"] = rolling_vol[col]
        wdf.reset_index(inplace=True)
        out[f"{w}D_Rolling"] = wdf.dropna()
    return out


def _summarize_rolling_dataframe(rolling_df: pd.DataFrame) -> pd.DataFrame:
    """Summarizes a rolling statistics DataFrame."""
    numeric = rolling_df.select_dtypes(include=np.number)
    summary = numeric.agg(["mean", "std", "min", "max"]).transpose()
    summary.index.name = "Rolling Statistic"
    summary.reset_index(inplace=True)
    return summary


def _compute_monthly_returns(
    df_pct_returns: pd.DataFrame,
    columns_to_analyze: list[str],
    date_col: str = "timestamp",
) -> pd.DataFrame:
    """Computes monthly returns from a DataFrame of daily percentage returns."""
    df = df_pct_returns.set_index(date_col)
    monthly_last = df[columns_to_analyze].resample("ME").last()

    monthly_last_filled = monthly_last.ffill()
    monthly_ret = monthly_last_filled.pct_change().fillna(0)

    monthly_ret.index = monthly_ret.index.strftime("%Y-%m")
    monthly_ret.reset_index(inplace=True)
    monthly_ret.rename(columns={"index": "Month"}, inplace=True)
    return monthly_ret


def _compute_return_correlations(
    df_pct_returns: pd.DataFrame,
    columns_to_analyze: list[str],
    date_col: str = "timestamp",
) -> pd.DataFrame:
    """Computes the correlation matrix for specified columns."""
    df = df_pct_returns.set_index(date_col)
    return df[columns_to_analyze].corr()


# --- Portfolio Risk Calculation Helpers (Unchanged) ---


def _calculate_portfolio_risk_components(
    full_historical_data: pd.DataFrame, portfolio_weights: dict[str, float]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Calculates risk components, returning the correlation matrix,
    individual volatilities, and weights DataFrame separately.
    """
    if full_historical_data.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

    price_pivot = full_historical_data.pivot(
        index="timestamp", columns="ticker", values="close_price"
    )
    price_pivot.index = pd.to_datetime(price_pivot.index, errors="coerce")
    price_pivot = price_pivot[price_pivot.index.notna()].sort_index()
    daily_close = price_pivot.resample("1D").last().ffill()
    daily_returns = daily_close.pct_change().dropna()

    if daily_returns.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

    aligned_tickers = [
        ticker for ticker in portfolio_weights.keys() if ticker in daily_returns.columns
    ]
    daily_returns = daily_returns[aligned_tickers]

    annualized_corr_matrix = daily_returns.corr()
    annualized_volatilities = daily_returns.std() * np.sqrt(252)
    weights_df = pd.DataFrame(
        list(portfolio_weights.items()), columns=["ticker", "weight"]
    )

    return annualized_corr_matrix, annualized_volatilities, weights_df


def _calculate_rolling_portfolio_risk(
    full_historical_data: pd.DataFrame,
    portfolio_weights: dict[str, float],
    window_days: int = 30,
) -> pd.DataFrame:
    """Calculates the rolling portfolio risk with a full window buffer."""
    if full_historical_data.empty:
        return pd.DataFrame()

    price_pivot = full_historical_data.pivot(
        index="timestamp", columns="ticker", values="close_price"
    )
    price_pivot.index = pd.to_datetime(price_pivot.index, errors="coerce")
    price_pivot = price_pivot[price_pivot.index.notna()].sort_index()
    daily_close = price_pivot.resample("1D").last().ffill()
    daily_returns = daily_close.pct_change().dropna()

    if len(daily_returns) < window_days:
        return pd.DataFrame()

    weights = pd.Series(portfolio_weights)
    aligned_tickers = [
        ticker for ticker in weights.index if ticker in daily_returns.columns
    ]
    weights = weights[aligned_tickers]
    daily_returns = daily_returns[aligned_tickers]

    rolling_cov = (
        daily_returns.rolling(window=window_days, min_periods=window_days).cov() * 252
    )
    rolling_cov = rolling_cov.dropna()

    if rolling_cov.empty:
        return pd.DataFrame()

    rolling_portfolio_variance = rolling_cov.groupby(level="timestamp").apply(
        lambda cov_matrix: np.dot(
            weights.T,
            np.dot(
                cov_matrix.droplevel("timestamp")
                .loc[weights.index, weights.index]
                .values,
                weights,
            ),
        )
    )

    rolling_portfolio_risk = np.sqrt(rolling_portfolio_variance)

    return pd.DataFrame(
        {
            "timestamp": rolling_portfolio_risk.index,
            f"rolling_{window_days}d_portfolio_risk": rolling_portfolio_risk,
        }
    )

def _build_csv(df: pd.DataFrame, filename: str, logger: logging.Logger, out_dir: str) -> None:
    """Helper function to save a DataFrame to CSV with error handling."""
    try:
        if not df.empty:
            path = os.path.join(out_dir, filename)
            logger.info(f"saving {filename} to {path}")
            df.to_csv(path, index=False)
        else:
            logger.warning(f"{filename} is empty; skipping CSV export")
    except Exception as e:
        logger.error(f"Error saving {filename}: {e}", exc_info=True)

# --- Main Reporting Function (CORRECTED) ---
def generate_backtest_report(
    portfolio: BasePortfolio,
    perf_df: pd.DataFrame,
    initial_capital: float,
    full_historical_data: pd.DataFrame,
) -> None:
    """
    Generates and saves a full backtest report with enhanced risk analysis.
    """
    logger = portfolio.logger
    reports = {}
    logger.info("Generating backtest report...")
    if perf_df.empty:
        logger.warning("Performance DataFrame is empty. Skipping report generation")
        return None
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        "src", "backtest", "data", f"{run_ts}_backtest_{portfolio.portfolio_id}"
    )
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Report output directory: {out_dir}")
    # Section 1: Trade logs
    if isinstance(portfolio.executor, BacktestExecutor):
        try:
            all_trades = portfolio.executor.trade_log
            logger.info(f"Trade log contains {len(all_trades)} trades")
            if all_trades:
                df_trades = pd.DataFrame(all_trades)
                cols = [
                    "timestamp",
                    "portfolio_id",
                    "ticker",
                    "signal_type",
                    "shares",
                    "fill_price",
                    "confidence",
                    "cash_after",
                ]
                df_trades = df_trades[[c for c in cols if c in df_trades.columns]]
                df_trades.sort_values("timestamp", inplace=True)
                reports["trade_log"] = df_trades.copy()
            else:
                logger.warning(
                    "Trade log is empty - no trades were executed during backtest"
                )
        except Exception as e:
            logger.error(f"Error saving trade logs: {e}", exc_info=True)

    # Section 2: Raw Performance Timeseries
    reports["performance_timeseries_absolute"] = perf_df.copy()

    # Section 3: Final Summary Metrics
    metrics_df = aggregate_final_metrics(perf_df)
    reports["summary_metrics"] = metrics_df.copy()

    # Section 4: Percentage Returns DataFrame
    pct_df = pd.DataFrame()
    if "portfolio_value" in perf_df and initial_capital > 0:
        pct_df = perf_df[["timestamp"]].copy()
        pct_df["portfolio_pct_ret"] = (
            perf_df["portfolio_value"] / initial_capital
        ) - 1.0
        reports["performance_timeseries_percentage"] = pct_df.copy()

    # Section 5: High-frequency performance report
    try:
        minute_perf_df = _generate_minute_by_minute_performance(
            trade_log=portfolio.executor.trade_log,
            full_historical_data=full_historical_data,
            initial_capital=initial_capital,
            tickers=portfolio.tickers,
        )
        if not minute_perf_df.empty:
            reports["performance_timeseries_minute_by_minute"] = minute_perf_df.copy()
    except Exception as e:
        logger.error(
            f"Error generating minute-by-minute performance report: {e}", exc_info=True
        )

    # Section 6: Buy-and-hold benchmark report
    try:
        benchmark_df = _generate_buy_and_hold_benchmark(
            full_historical_data=full_historical_data,
            initial_capital=initial_capital,
            portfolio_weights=portfolio.portfolio_weights,
        )
        if not benchmark_df.empty:
            reports["benchmark_buy_and_hold"] = benchmark_df.copy()
    except Exception as e:
        logger.error(
            f"Error generating buy-and-hold benchmark report: {e}", exc_info=True
        )

    # Section 7: Advanced Analytics
    try:
        if not pct_df.empty:
            cols_to_analyze = ["portfolio_pct_ret"]
            roll_map = _compute_rolling_stats(pct_df, cols_to_analyze)
            for name, rdf in roll_map.items():
                reports[name] = rdf.copy()
                reports[f"{name}_summary"] = _summarize_rolling_dataframe(rdf).copy()
            mdf = _compute_monthly_returns(pct_df, cols_to_analyze)
            reports["monthly_returns"] = mdf.copy()
            if len(cols_to_analyze) > 1:
                reports["portfolio_return_correlations"] = _compute_return_correlations(pct_df, cols_to_analyze).copy()
    except Exception as e:
        logger.error(f"Error in advanced analytics: {e}", exc_info=True)

    # Section 8: Portfolio Risk Analytics
    try:
        corr_matrix, indiv_vols, weights_df = _calculate_portfolio_risk_components(
            full_historical_data, portfolio.portfolio_weights
        )
        if not corr_matrix.empty:
            aligned_weights_df = weights_df[
                weights_df["ticker"].isin(corr_matrix.columns)
            ]

            risk_components_summary = pd.concat(
                [
                    aligned_weights_df.set_index("ticker"),
                    indiv_vols.rename("annualized_volatility"),
                ],
                axis=1,
            )

            reports["portfolio_risk_components"] = risk_components_summary.copy()

            reports["annualized_correlation_matrix"] = corr_matrix.copy()

        rolling_risk_df = _calculate_rolling_portfolio_risk(
            full_historical_data, portfolio.portfolio_weights
        )
        if not rolling_risk_df.empty:
            reports["rolling_portfolio_risk"] = rolling_risk_df.copy()

    except Exception as e:
        logger.error(f"Error in portfolio risk analytics: {e}", exc_info=True)


    from concurrent.futures import ThreadPoolExecutor
    try:
        # Save all reports to CSV
        logger.info("Preparing to save report CSVs")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for name, df in reports.items():
                futures.append(executor.submit(_build_csv, df, f"{name}.csv", logger, out_dir))
            for future in futures:
                future.result()  # Wait for all to complete and catch exceptions
    except Exception as e:
        logger.error(f"Error saving reports to CSV: {e}", exc_info=True)
    logger.info("Backtest report generation complete.")
