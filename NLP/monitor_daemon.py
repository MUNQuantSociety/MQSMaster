#!/usr/bin/env python3
"""Monitoring + synthetic-check tooling for the NLP scraper daemon."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil

# Script-mode bootstrap - see ``daemon.py`` for the same pattern.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from NLP.core import DAEMON_LOG_FILE

SCRIPT_DIR = Path(__file__).parent
LOG_FILE = DAEMON_LOG_FILE


def get_daemon_process() -> Optional[psutil.Process]:
    """Return the daemon ``psutil.Process`` if it is running, else None."""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = proc.info["name"] or ""
            if "python" in name.lower() and proc.info["cmdline"]:
                cmdline = " ".join(proc.info["cmdline"])
                if "daemon.py start" in cmdline:
                    return psutil.Process(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


class DaemonLogStats:
    """Parse :data:`LOG_FILE` and surface aggregate metrics."""

    EMPTY_STATS: dict = {
        "total_cycles": 0,
        "successful_cycles": 0,
        "failed_cycles": 0,
        "skipped_tickers": 0,
        "processed_tickers": 0,
        "last_cycle_time": None,
        "average_cycle_time": 0,
        "skip_rate": 0,
    }

    def __init__(self, log_file: Path = LOG_FILE):
        self.log_file = log_file

    def parse(self) -> dict:
        if not self.log_file.exists():
            return dict(self.EMPTY_STATS)

        stats = dict(self.EMPTY_STATS)
        cycle_times: list[int] = []

        try:
            with open(self.log_file, "r") as f:
                lines = f.readlines()

            for line in lines:
                if "Starting scraping cycle #" in line:
                    stats["total_cycles"] += 1
                elif "Scraping cycle completed successfully" in line:
                    stats["successful_cycles"] += 1
                elif "Scraping cycle failed" in line:
                    stats["failed_cycles"] += 1
                elif "SKIPPING" in line and "No new articles" in line:
                    stats["skipped_tickers"] += 1
                elif "Successfully processed sentiment" in line:
                    stats["processed_tickers"] += 1
                elif "completed in" in line and "seconds" in line:
                    try:
                        time_part = line.split("completed in ")[1].split(" seconds")[0]
                        cycle_times.append(int(time_part))
                    except Exception:
                        pass
                elif "Current skip rate:" in line:
                    try:
                        rate_part = line.split("Current skip rate: ")[1].split("%")[0]
                        stats["skip_rate"] = float(rate_part)
                    except Exception:
                        pass

            if cycle_times:
                stats["average_cycle_time"] = sum(cycle_times) / len(cycle_times)
                stats["last_cycle_time"] = cycle_times[-1]

        except Exception as exc:
            print(f"Error parsing log: {exc}")

        return stats


# ----------------------------------------------------------------------
# Backwards-compatible module-level shim used by old callers.
# ----------------------------------------------------------------------


def parse_log_stats() -> dict:
    return DaemonLogStats().parse()


class DaemonMonitor:
    """Pretty-print runtime status to stdout."""

    def __init__(self, log_file: Path = LOG_FILE):
        self.log_stats = DaemonLogStats(log_file=log_file)
        self.log_file = log_file

    def report(self) -> None:
        print("NLP Daemon Monitor")
        print("=" * 50)

        daemon_proc = get_daemon_process()
        if daemon_proc:
            print(f"✓ Daemon is RUNNING (PID: {daemon_proc.pid})")
            try:
                cpu_percent = daemon_proc.cpu_percent(interval=1)
                memory_mb = daemon_proc.memory_info().rss / 1024 / 1024
                print(f"  CPU Usage: {cpu_percent:.1f}%")
                print(f"  Memory Usage: {memory_mb:.1f} MB")
                print(
                    f"  Running since: "
                    f"{datetime.fromtimestamp(daemon_proc.create_time()).strftime('%Y-%m-%d %H:%M:%S')}"
                )
            except Exception as exc:
                print(f"  Error getting process info: {exc}")
        else:
            print("✗ Daemon is NOT RUNNING")
        print()

        stats = self.log_stats.parse()

        if stats["total_cycles"] > 0:
            print("Performance Statistics:")
            print(f"  Total Cycles: {stats['total_cycles']}")
            print(f"  Successful: {stats['successful_cycles']}")
            print(f"  Failed: {stats['failed_cycles']}")
            print(
                f"  Success Rate: "
                f"{(stats['successful_cycles'] / stats['total_cycles'] * 100):.1f}%"
            )
            print()

            print("Processing Statistics:")
            print(f"  Tickers Processed: {stats['processed_tickers']}")
            print(f"  Tickers Skipped: {stats['skipped_tickers']}")
            if stats["skip_rate"] > 0:
                print(f"  Current Skip Rate: {stats['skip_rate']:.1f}%")
            print()

            if stats["average_cycle_time"] > 0:
                print("Timing Statistics:")
                print(f"  Average Cycle Time: {stats['average_cycle_time']:.0f} seconds")
                if stats["last_cycle_time"]:
                    print(f"  Last Cycle Time: {stats['last_cycle_time']} seconds")
        else:
            print("No cycle statistics available yet.")

        if self.log_file.exists():
            log_size = self.log_file.stat().st_size / 1024 / 1024
            print(f"\nLog File: {log_size:.1f} MB")

        print(f"\nLast updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


class DaemonHealthCheck:
    """CI-friendly synthetic check used by GitHub Actions."""

    def __init__(self, max_log_age_hours: int = 48, log_file: Path = LOG_FILE):
        self.max_log_age_hours = max_log_age_hours
        self.log_file = log_file
        self.log_stats = DaemonLogStats(log_file=log_file)

    def run(self) -> int:
        stats = self.log_stats.parse()
        daemon_proc = get_daemon_process()

        result: dict = {
            "daemon_running": bool(daemon_proc),
            "log_exists": self.log_file.exists(),
            "total_cycles": stats.get("total_cycles", 0),
            "successful_cycles": stats.get("successful_cycles", 0),
            "failed_cycles": stats.get("failed_cycles", 0),
            "status": "ok",
            "reasons": [],
        }

        if self.log_file.exists():
            age_hours = (time.time() - self.log_file.stat().st_mtime) / 3600
            result["log_age_hours"] = round(age_hours, 2)
            if age_hours > float(self.max_log_age_hours):
                result["status"] = "failed"
                result["reasons"].append(
                    f"daemon.log older than {self.max_log_age_hours}h"
                )
        else:
            result["status"] = "warn"
            result["reasons"].append("daemon.log not found")

        total_cycles = int(result["total_cycles"])
        failed_cycles = int(result["failed_cycles"])
        successful_cycles = int(result["successful_cycles"])
        if total_cycles >= 3 and failed_cycles > successful_cycles:
            result["status"] = "failed"
            result["reasons"].append("failed cycles exceed successful cycles")

        print(json.dumps(result))
        return 1 if result["status"] == "failed" else 0


# Back-compat free-function wrappers.

def monitor_daemon() -> None:
    DaemonMonitor().report()


def run_synthetic_check(max_log_age_hours: int = 48) -> int:
    return DaemonHealthCheck(max_log_age_hours=max_log_age_hours).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor NLP daemon health")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run CI-friendly synthetic checks and exit with status code",
    )
    parser.add_argument(
        "--max-log-age-hours",
        type=int,
        default=48,
        help="Warn when daemon.log age exceeds this threshold",
    )
    args = parser.parse_args()

    if args.synthetic:
        raise SystemExit(run_synthetic_check(max_log_age_hours=args.max_log_age_hours))

    try:
        monitor_daemon()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
