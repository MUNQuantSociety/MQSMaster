# Utilities Scripts

This folder contains utility scripts for testing, data fetching, and backtest analysis.

## File Overview

### Backtest Utilities

#### `backtest_entrypoint.py`
Main entry point for running backtest analysis tools. Provides a unified command interface for reading and analyzing backtest results.

**Usage:**
```bash
python scripts/backtest_entrypoint.py <command> [options]
```

**Available commands:**
- `read` - Loads latest portfolio runs and prints summary metrics, risk components, and performance
- `analyze` - Computes optimized weights from backtest covariance data and prints risk/return metrics

**Common options:**
- `--data-root <path>` - Override default backtest data location (default: src/backtest/data)
- `--portfolio <id>` - Filter to specific portfolio ID (repeatable)
- `--sample-rows <n>` - Number of displayed rows for read command
- `--risk-appetite <0..1>` - Risk appetite parameter for analyze command

**Examples:**
```bash
python scripts/backtest_entrypoint.py read --sample-rows 3
python scripts/backtest_entrypoint.py analyze --risk-appetite 0.35 --portfolio 2
```

#### `backtest_reader.py`
Reads and displays backtest output data from the latest runs. Dynamically discovers backtest folders named `YYYYMMDD_HHMMSS_backtest_<portfolio_id>` under `src/backtest/data`.

**Usage:**
```bash
python scripts/backtest_reader.py --sample-rows 3
```

**Output:**
- Summary metrics
- Risk component samples
- Performance samples

#### `backtest_analyzer.py`
Analyzes backtest results and computes optimized portfolio weights using real backtest covariance data.

**Usage:**
```bash
python scripts/backtest_analyzer.py --risk-appetite 0.5
```

**Output:**
- Current portfolio weights
- Optimized weights
- Risk and return metrics

#### `summary_metrics_formatter.py`
Utility for normalizing and formatting summary metrics. Provides type conversion and normalization for financial metrics including percentages, currency values, and comma-separated numbers.

**Functions:**
- `_normalize_metric_name(name)` - Converts metric names to lowercase alphanumeric format
- `to_float(value)` - Converts various metric formats to float (supports %, $, commas)

### Data Fetching & Testing

#### `alpha_test.py`
Fetches news sentiment data from Alpha Vantage API. Useful for testing news data pipeline integration and validating sentiment data retrieval.

**Functions:**
- `scrape_alpha(ticker, time_from, time_to, apikey)` - Fetches news articles and sentiment data for specified tickers

**Environment:**
- Requires `ALPHA_KEY` environment variable

**Usage:**
```python
scrape_alpha(ticker=["AAPL"], time_from="20251201T1200", time_to="20251231T1200")
```

#### `api_test.py`
Tests yfinance API for fetching financial statements and data. Validates access to income statements, balance sheets, cash flows, earnings data, and SEC filings.

**Functions:**
- `test_statements()` - Fetches and displays various financial data for AAPL

**Data retrieved:**
- Annual and quarterly income statements
- TTM (trailing twelve months) statements
- Balance sheets
- Cash flow statements
- Earnings and earnings dates
- SEC filings
- Calendar data

## Typical Workflow

```bash
# 1) Quick view of all latest portfolio runs
python scripts/backtest_entrypoint.py read --sample-rows 2

# 2) Analyze one portfolio with explicit risk appetite
python scripts/backtest_entrypoint.py analyze --portfolio 2 --risk-appetite 0.40

# 3) Test data API connections
python scripts/api_test.py
python scripts/alpha_test.py
```

## Direct Script Usage

Run individual scripts directly without the entry point wrapper when you only need one specific tool:

```bash
python scripts/backtest_reader.py --sample-rows 3
python scripts/backtest_analyzer.py --risk-appetite 0.5
python scripts/api_test.py
```
