from __future__ import annotations

import re
from typing import Any

import pandas as pd


def _normalize_metric_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def to_float(value: Any) -> float:
    """Best-effort conversion for summary values (supports %, $, commas)."""
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return float("nan")

    text = str(value).strip()
    if not text:
        return float("nan")

    is_percent = text.endswith("%")
    cleaned = text.replace("$", "").replace(",", "").replace("%", "")

    try:
        parsed = float(cleaned)
    except ValueError:
        return float("nan")

    return parsed / 100.0 if is_percent else parsed


def format_summary_metric(metric_name: str, raw_value: Any) -> str:
    value = to_float(raw_value)
    if pd.isna(value):
        return str(raw_value)

    label = metric_name.lower()
    if "drawdown" in label or "return" in label or "(%)" in label:
        return f"{value:.2%}"
    if "value" in label or "capital" in label or "notional" in label:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def format_backtest_date_range(
    perf_df: pd.DataFrame, timestamp_col: str = "timestamp"
) -> str:
    """Return a readable date range label from a performance DataFrame."""
    if perf_df.empty or timestamp_col not in perf_df.columns:
        return "N/A"

    timestamps = pd.to_datetime(perf_df[timestamp_col], errors="coerce", utc=True)
    timestamps = timestamps.dropna()
    if timestamps.empty:
        return "N/A"

    start_date = timestamps.min().date().isoformat()
    end_date = timestamps.max().date().isoformat()
    if start_date == end_date:
        return start_date
    return f"{start_date} to {end_date}"


def summary_to_metric_map(summary_df: pd.DataFrame) -> dict[str, Any]:
    """Return summary metrics as a name->value mapping for either supported format."""
    if summary_df.empty:
        return {}

    if {"metric", "value"}.issubset(summary_df.columns):
        metric_map = {}
        for item in summary_df.itertuples(index=False):
            name = str(getattr(item, "metric", "")).strip()
            if name:
                metric_map[name] = getattr(item, "value", "")
        return metric_map

    row = summary_df.iloc[0]
    return {str(col): row[col] for col in summary_df.columns}


def get_summary_value(summary_df: pd.DataFrame, candidate_keys: list[str]) -> float:
    """Get a numeric summary metric by trying multiple key aliases."""
    metric_map = summary_to_metric_map(summary_df)
    if not metric_map:
        return float("nan")

    normalized = {_normalize_metric_name(k): v for k, v in metric_map.items()}

    for key in candidate_keys:
        key_norm = _normalize_metric_name(key)
        if key_norm in normalized:
            return to_float(normalized[key_norm])

    return float("nan")


def print_summary_metrics(summary_df: pd.DataFrame, prefix: str = "  ") -> None:
    print("Summary metrics:")
    metric_map = summary_to_metric_map(summary_df)
    if not metric_map:
        print(f"{prefix}<no summary metrics>")
        return

    width = max(len(str(metric_name)) for metric_name in metric_map)
    for metric_name, metric_value in metric_map.items():
        formatted_value = format_summary_metric(metric_name, metric_value)
        print(f"{prefix}{metric_name:<{width}} : {formatted_value}")
