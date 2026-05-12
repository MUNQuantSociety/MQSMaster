from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from summary_metrics_formatter import format_backtest_date_range, get_summary_value

MIN_WEIGHT = 0.05
RUN_NAME_RE = re.compile(r"(?P<timestamp>\d{8}_\d{6})_backtest_(?P<portfolio_id>.+)")
SUMMARY_FILE = "summary_metrics.csv"
RISK_FILE = "portfolio_risk_components.csv"
CORR_FILE = "annualized_correlation_matrix.csv"
PERFORMANCE_FILE = "performance_timeseries_absolute.csv"


@dataclass(frozen=True)
class BacktestRun:
    portfolio_id: str
    timestamp: str
    path: Path


def default_data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "backtest" / "data"


def discover_backtest_runs(data_root: Path) -> list[BacktestRun]:
    if not data_root.exists():
        raise FileNotFoundError(f"Backtest data root does not exist: {data_root}")

    runs: list[BacktestRun] = []
    for entry in data_root.iterdir():
        if not entry.is_dir():
            continue
        match = RUN_NAME_RE.fullmatch(entry.name)
        if not match:
            continue
        runs.append(
            BacktestRun(
                portfolio_id=match.group("portfolio_id"),
                timestamp=match.group("timestamp"),
                path=entry,
            )
        )

    if not runs:
        raise FileNotFoundError(f"No backtest run folders found under: {data_root}")

    return runs


def select_latest_runs_with_files(
    runs: list[BacktestRun],
    required_files: tuple[str, ...],
    portfolio_filter: set[str] | None = None,
) -> dict[str, BacktestRun]:
    by_portfolio: dict[str, list[BacktestRun]] = {}
    for run in runs:
        if portfolio_filter and run.portfolio_id not in portfolio_filter:
            continue
        by_portfolio.setdefault(run.portfolio_id, []).append(run)

    selected: dict[str, BacktestRun] = {}
    for portfolio_id, portfolio_runs in by_portfolio.items():
        sorted_runs = sorted(portfolio_runs, key=lambda x: x.timestamp, reverse=True)
        for run in sorted_runs:
            if all((run.path / name).exists() for name in required_files):
                selected[portfolio_id] = run
                break
    return dict(sorted(selected.items()))


def load_csv(run: BacktestRun, filename: str) -> pd.DataFrame:
    path = run.path / filename
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    return pd.read_csv(path)


def load_csv_if_exists(run: BacktestRun, filename: str) -> pd.DataFrame:
    path = run.path / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_covariance_matrix(
    risk_df: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    expected_cols = {"ticker", "weight", "annualized_volatility"}
    missing = expected_cols.difference(set(risk_df.columns))
    if missing:
        raise ValueError(f"{RISK_FILE} missing columns: {sorted(missing)}")

    risk_df = risk_df.copy()
    risk_df["ticker"] = risk_df["ticker"].astype(str)
    risk_df["weight"] = pd.to_numeric(risk_df["weight"], errors="coerce")
    risk_df["annualized_volatility"] = pd.to_numeric(
        risk_df["annualized_volatility"], errors="coerce"
    )
    risk_df = risk_df.dropna(subset=["ticker", "weight", "annualized_volatility"])
    if risk_df.empty:
        raise ValueError(f"{RISK_FILE} has no usable rows.")

    tickers = risk_df["ticker"].tolist()
    current_weights = risk_df["weight"].to_numpy(dtype=float)
    total = current_weights.sum()
    if np.isclose(total, 0.0):
        current_weights = np.full(
            len(current_weights),
            1.0 / len(current_weights),
            dtype=float,
        )
    else:
        current_weights = current_weights / total
    annualized_vols = risk_df["annualized_volatility"].to_numpy(dtype=float)

    corr_df = corr_df.copy()
    index_col = corr_df.columns[0]
    corr_df[index_col] = corr_df[index_col].astype(str)
    corr_df = corr_df.set_index(index_col)
    corr_df.columns = corr_df.columns.astype(str)

    missing_tickers = [
        ticker
        for ticker in tickers
        if ticker not in corr_df.index or ticker not in corr_df.columns
    ]
    if missing_tickers:
        raise ValueError(f"{CORR_FILE} missing tickers: {missing_tickers}")

    corr_matrix = corr_df.loc[tickers, tickers].astype(float).to_numpy()
    vol_diag = np.diag(annualized_vols)
    cov_matrix = vol_diag @ corr_matrix @ vol_diag

    return tickers, current_weights, cov_matrix


def optimize_weights(volatilities: np.ndarray, risk_appetite: float) -> np.ndarray:
    clipped_vols = np.clip(volatilities, 1e-12, None)
    inv_vol = 1.0 / clipped_vols
    base_weights = inv_vol / inv_vol.sum()

    equal_weights = np.full_like(base_weights, 1.0 / len(base_weights), dtype=float)
    weights = (1.0 - risk_appetite) * base_weights + risk_appetite * equal_weights

    effective_min_weight = min(MIN_WEIGHT, 1.0 / len(weights))
    weights = np.maximum(weights, effective_min_weight)
    weights /= weights.sum()
    return weights


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    variance = float(weights.T @ cov_matrix @ weights)
    return float(np.sqrt(max(variance, 0.0)))


def _fmt_pct(value: float) -> str:
    return "N/A" if np.isnan(value) else f"{value:.2%}"


def _fmt_num(value: float, decimals: int = 4) -> str:
    return "N/A" if np.isnan(value) else f"{value:.{decimals}f}"


def print_analysis(run: BacktestRun, risk_appetite: float) -> None:
    summary_df = load_csv(run, SUMMARY_FILE)
    risk_df = load_csv(run, RISK_FILE)
    corr_df = load_csv(run, CORR_FILE)
    perf_df = load_csv_if_exists(run, PERFORMANCE_FILE)
    date_range = format_backtest_date_range(perf_df)

    tickers, current_weights, cov_matrix = build_covariance_matrix(risk_df, corr_df)
    asset_vols = np.sqrt(np.diag(cov_matrix))
    optimized_weights = optimize_weights(asset_vols, risk_appetite)

    current_vol = portfolio_volatility(current_weights, cov_matrix)
    optimized_vol = portfolio_volatility(optimized_weights, cov_matrix)

    annual_return = get_summary_value(
        summary_df, ["annual_return", "Annual Return", "Annualized Return", "cagr"]
    )
    sharpe = get_summary_value(
        summary_df,
        ["sharpe", "Annualized Sharpe Ratio", "sharpe_ratio", "Sharpe Ratio"],
    )
    max_drawdown = get_summary_value(
        summary_df, ["max_drawdown", "Max Drawdown (%)", "max_dd"]
    )
    final_value = get_summary_value(
        summary_df,
        ["Final Portfolio Value", "final_portfolio_value", "Final Capital"],
    )

    print(f"\nPortfolio: {run.portfolio_id} | Date Range: {date_range}")
    print(f"Source folder: {run.path}")
    print(f"Risk appetite: {risk_appetite:.2f}")
    print("\nCurrent weights from backtest data:")
    for ticker, weight in zip(tickers, current_weights):
        print(f"  {ticker:<8} {weight:>8.2%}")

    print("\nOptimized weights:")
    for ticker, weight in zip(tickers, optimized_weights):
        print(f"  {ticker:<8} {weight:>8.2%}")

    print("\nBacktest summary metrics:")
    if not np.isnan(final_value):
        print(f"  Final Portfolio Value:  {final_value:>10,.2f}")
    print(f"  Annual Return:          {_fmt_pct(annual_return):>10}")
    print(f"  Sharpe:                 {_fmt_num(sharpe):>10}")
    print(f"  Max Drawdown:           {_fmt_pct(max_drawdown):>10}")
    print(f"  Current Ann. Vol:       {current_vol:>10.2%}")
    print(f"  Optimized Ann. Vol:     {optimized_vol:>10.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the latest backtest runs per portfolio by dynamically loading "
            "risk and summary artifacts from src/backtest/data."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="Backtest data root directory.",
    )
    parser.add_argument(
        "--portfolio",
        action="append",
        dest="portfolios",
        help="Optional portfolio_id filter. Repeat flag for multiple portfolios.",
    )
    parser.add_argument(
        "--risk-appetite",
        type=float,
        default=None,
        help="Risk appetite in [0, 1]. If omitted, prompt in terminal.",
    )
    return parser.parse_args()


def resolve_risk_appetite(risk_appetite_arg: float | None) -> float:
    if risk_appetite_arg is not None:
        risk_appetite = risk_appetite_arg
    else:
        raw_input = input("Enter risk appetite (0=low risk, 1=high risk): ").strip()
        risk_appetite = float(raw_input)

    if not 0.0 <= risk_appetite <= 1.0:
        raise ValueError("Risk appetite must be between 0 and 1.")
    return risk_appetite


def main() -> None:
    args = parse_args()
    risk_appetite = resolve_risk_appetite(args.risk_appetite)
    portfolio_filter = set(args.portfolios) if args.portfolios else None

    runs = discover_backtest_runs(args.data_root)
    selected_runs = select_latest_runs_with_files(
        runs,
        required_files=(SUMMARY_FILE, RISK_FILE, CORR_FILE),
        portfolio_filter=portfolio_filter,
    )

    if not selected_runs:
        print(
            "No matching backtest runs found with required files "
            f"({SUMMARY_FILE}, {RISK_FILE}, {CORR_FILE})."
        )
        return

    for _, run in selected_runs.items():
        print_analysis(run, risk_appetite)


if __name__ == "__main__":
    main()
