from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from summary_metrics_formatter import format_backtest_date_range, print_summary_metrics

RUN_NAME_RE = re.compile(r"(?P<timestamp>\d{8}_\d{6})_backtest_(?P<portfolio_id>.+)")
PERFORMANCE_FILE = "performance_timeseries_absolute.csv"
SUMMARY_FILE = "summary_metrics.csv"
RISK_FILE = "portfolio_risk_components.csv"


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


def select_latest_runs(
    runs: list[BacktestRun],
    portfolio_filter: set[str] | None = None,
) -> dict[str, BacktestRun]:
    latest_runs: dict[str, BacktestRun] = {}
    for run in sorted(runs, key=lambda x: x.timestamp):
        if portfolio_filter and run.portfolio_id not in portfolio_filter:
            continue
        latest_runs[run.portfolio_id] = run
    return dict(sorted(latest_runs.items()))


def load_csv_if_exists(run: BacktestRun, filename: str) -> pd.DataFrame:
    path = run.path / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def print_portfolio_snapshot(run: BacktestRun, sample_rows: int) -> pd.DataFrame:
    perf_df = load_csv_if_exists(run, PERFORMANCE_FILE)
    summary_df = load_csv_if_exists(run, SUMMARY_FILE)
    risk_df = load_csv_if_exists(run, RISK_FILE)
    date_range = format_backtest_date_range(perf_df)

    print(f"\nPortfolio: {run.portfolio_id} | Date Range: {date_range}")
    print(f"Source folder: {run.path}")

    if not summary_df.empty:
        print_summary_metrics(summary_df)
    else:
        print(f"Summary metrics not found ({SUMMARY_FILE}).")

    if not risk_df.empty:
        print("Risk components sample:")
        print(risk_df.head(sample_rows).to_string(index=False))
    else:
        print(f"Risk components not found ({RISK_FILE}).")

    if perf_df.empty:
        print(f"Performance timeseries not found ({PERFORMANCE_FILE}).")
        return pd.DataFrame()

    print("Performance sample:")
    print(perf_df.head(sample_rows).to_string(index=False))

    perf_df = perf_df.copy()
    perf_df["portfolio_id"] = run.portfolio_id
    perf_df["date_range"] = date_range
    return perf_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dynamically load latest backtest outputs by portfolio and print "
            "portfolio/backtest samples to screen."
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
        "--sample-rows",
        type=int,
        default=5,
        help="Number of sample rows to print from each DataFrame.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    portfolio_filter = set(args.portfolios) if args.portfolios else None

    runs = discover_backtest_runs(args.data_root)
    selected_runs = select_latest_runs(runs, portfolio_filter=portfolio_filter)
    if not selected_runs:
        print("No matching portfolio runs found.")
        return

    combined_perf_frames: list[pd.DataFrame] = []
    for _, run in selected_runs.items():
        perf_df = print_portfolio_snapshot(run, sample_rows=max(args.sample_rows, 1))
        if not perf_df.empty:
            combined_perf_frames.append(perf_df)

    if combined_perf_frames:
        combined_perf = pd.concat(combined_perf_frames, ignore_index=True)
        print("\nCombined performance snapshot across selected portfolios:")
        print(f"Rows: {len(combined_perf)}, Columns: {len(combined_perf.columns)}")
        print(combined_perf.head(max(args.sample_rows, 1)).to_string(index=False))
    else:
        print("\nNo performance timeseries files were available to combine.")


if __name__ == "__main__":
    main()
