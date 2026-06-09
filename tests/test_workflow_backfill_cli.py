from orchestrator.backfill.concurrent_backfill import concurrent_backfill
import pytest

from src.orchestrator.backfill.backfill_cli import DATE_FMT, build_parser

pytestmark = [
    pytest.mark.integration,
    pytest.mark.smoke,
    pytest.mark.workflow_backfill,
]


def test_backfill_cli_specific_parsing():
    parser = build_parser()
    args = parser.parse_args(
        [
            "specific",
            "--start",
            "010124",
            "--end",
            "050124",
            "--tickers",
            "AAPL",
            "MSFT",
            "--interval",
            "5",
        ]
    )

    assert args.command == "specific"
    assert args.start.strftime(DATE_FMT) == "010124"
    assert args.end.strftime(DATE_FMT) == "050124"
    assert args.tickers == ["AAPL", "MSFT"]
    assert args.interval == 5

@pytest.mark.db
def test_concurrent_backfill():
    parser = build_parser()
    args = parser.parse_args(
        [
            "concurrent",
            "--start",
            "010124",
            "--end",
            "050124",
            "--tickers",
            "AAPL",
            "MSFT",
            "--interval",
            "5",
            "--threads",
            "4"
        ]
    )

    assert args.command == "concurrent"
    assert args.start.strftime(DATE_FMT) == "010124"
    assert args.end.strftime(DATE_FMT) == "050124"
    assert args.tickers == ["AAPL", "MSFT"]
    assert args.interval == 5
    assert args.threads == 4

    concurrent_backfill(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        interval=args.interval,
        threads=args.threads,
    )

    

def test_backfill_cli_inject_csv_parsing():
    parser = build_parser()
    args = parser.parse_args(
        ["inject-csv", "--csv-dir", "data/cache", "--threads", "4"]
    )

    assert args.command == "inject-csv"
    assert args.csv_dir == "data/cache"
    assert args.threads == 4
