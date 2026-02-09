#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail CI if any file coverage is below a minimum percent."
    )
    parser.add_argument(
        "--coverage-file",
        default="coverage.json",
        help="Path to coverage.py JSON output (default: coverage.json).",
    )
    parser.add_argument(
        "--min",
        type=float,
        default=50.0,
        help="Minimum per-file coverage percentage.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    coverage_path = Path(args.coverage_file)
    if not coverage_path.exists():
        print(f"Coverage file not found: {coverage_path}")
        return 2

    data = json.loads(coverage_path.read_text(encoding="utf-8"))
    files = data.get("files", {})
    if not files:
        print("No files found in coverage data.")
        return 2

    failures = []
    for file_path, info in files.items():
        summary = info.get("summary", {})
        num_statements = summary.get("num_statements", 0)
        percent_covered = summary.get("percent_covered")
        if num_statements == 0 or percent_covered is None:
            continue
        if percent_covered < args.min:
            failures.append((file_path, percent_covered, num_statements))

    if failures:
        failures.sort(key=lambda item: item[1])
        print("Per-file coverage below minimum:")
        for file_path, percent_covered, num_statements in failures:
            print(
                f"- {file_path}: {percent_covered:.2f}% "
                f"({num_statements} statements)"
            )
        print(f"Minimum required: {args.min:.2f}%")
        return 1

    print(f"All files meet per-file coverage >= {args.min:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
