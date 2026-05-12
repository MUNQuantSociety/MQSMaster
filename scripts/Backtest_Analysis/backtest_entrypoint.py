from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from backtest_analyzer import main as analyze_main
from backtest_reader import main as read_main

COMMANDS: dict[str, Callable[[], None]] = {
    "analyze": analyze_main,
    "read": read_main
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Unified entry point for backtest reader/analyzer utilities. "
            "Pass tool-specific options after the command."
        )
    )
    parser.add_argument(
        "command",
        choices=sorted(COMMANDS.keys()),
        help="Tool to run.",
    )
    return parser.parse_known_args()


def run_command(command: str, forwarded_args: list[str]) -> None:
    target = COMMANDS[command]
    original_argv = sys.argv
    try:
        # Forward remaining CLI args so each underlying script keeps its own parser.
        sys.argv = [original_argv[0], *forwarded_args]
        target()
    finally:
        sys.argv = original_argv


def main() -> None:
    args, forwarded_args = parse_args()
    run_command(args.command, forwarded_args)


if __name__ == "__main__":
    main()
