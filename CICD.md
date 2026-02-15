# CI/CD Documentation

## Current State

The repository has three workflow files under `.github/workflows/`:

### 1. `main.yml` — The primary, well-structured pipeline

This is the most mature workflow and the one that should be used going forward.

- Triggers on PRs to `main` and pushes to `main`.
- Uses concurrency groups to cancel stale runs.
- CI job: Python 3.11, pip + pytest caching, ruff linting, unit tests (always), DB-gated integration tests (only when secrets are present), per-file coverage enforcement at 50%.
- CD job: builds and pushes a Docker image to GHCR on push to `main`, tagged with `latest` and the commit SHA. Only runs if CI passes and a Dockerfile exists.
- Sets `timeout-minutes: 20` on CI and `timeout-minutes: 30` on CD.
- Treats Python warnings as errors during test runs (`PYTHONWARNINGS=error`).

### 2. `cicd_main.yml` — Older duplicate targeting `main`

- Triggers on the same branches as `main.yml` (push/PR to `main`).
- Uses flake8 instead of ruff, Python 3.10 instead of 3.11.
- No coverage enforcement, no CD step, no concurrency control, no timeouts.
- Has a fragile "Verify tests folder" step that runs `ls` before the directory check.

### 3. `cicd.yml` — Copy of `cicd_main.yml` targeting `fallRefactor`

- Identical to `cicd_main.yml` but triggers on the `fallRefactor` branch.
- Same issues as above.

---

## Issues Found

### Duplicate / conflicting workflows
- `cicd_main.yml` and `main.yml` both trigger on push/PR to `main`. This means every PR and merge fires two separate CI runs doing roughly the same thing, wasting runner minutes and creating confusing status checks.
- `cicd.yml` targets a feature branch (`fallRefactor`) that may no longer exist. If the branch is merged or abandoned, this file is dead weight.

### Python version mismatch
- `main.yml` uses Python 3.11, the older workflows use 3.10, and the Dockerfile uses 3.12. There's no single source of truth for the Python version. Tests could pass in CI and fail in production (or vice versa) due to version differences.

### Linter mismatch
- `main.yml` uses ruff; the older workflows use flake8. Two different linters with different rule sets means inconsistent code quality checks depending on which workflow runs.

### No branch protection enforcement documented
- Nothing in the repo enforces that CI must pass before merging. Without branch protection rules on `main`, the CD job can be triggered by a direct push that skips CI.

### Secrets leak risk
- `cicd_main.yml` and `cicd.yml` pass DB secrets directly as env vars to the entire pytest step. `main.yml` handles this better by gating DB tests behind a secrets-availability check, but the older workflows don't.

### No deployment step
- The CD job builds and pushes a Docker image to GHCR, but there's no deployment trigger (no SSH deploy, no ECS/EKS update, no webhook). The image sits in the registry until someone manually pulls it.

### Coverage config gap
- `pyproject.toml` sets `source = ["."]` for coverage, which means coverage measures everything including scripts, NLP, RBP, etc. The per-file check at 50% could fail on files that aren't really part of the core application.

### `.env` is tracked
- `.env` is present in the repo root. If it contains real credentials, that's a security issue. It should be in `.gitignore` (it may already be — worth verifying it's not committed with real values).

### Workflow `name` collision
- Both `cicd_main.yml` and `cicd.yml` use the name `Python CI/CD Pipeline`. GitHub Actions UI will show two workflows with the same name, making it hard to tell them apart.

---

## What Needs to Be Done

1. Delete `cicd_main.yml` and `cicd.yml`. They are superseded by `main.yml` and cause duplicate runs.
2. Align the Python version across `main.yml` and the Dockerfile. Pick one (3.11 or 3.12) and use it everywhere.
3. Scope coverage to `src/` only in `pyproject.toml` so the per-file check doesn't flag unrelated scripts or notebooks.
4. Verify `.env` is gitignored and not committed with real secrets.
5. Set up GitHub branch protection rules on `main`: require the `ci` status check to pass, require PR reviews, disallow direct pushes.

---

## Improvements to Consider

### Add a deployment step
After the Docker image is pushed, trigger an actual deployment — SSH to a server, update an ECS service, call a webhook, etc. Without this, CD is really just "continuous delivery to a registry."

### Add a security scanning step
Add a step like `trivy` or `grype` to scan the Docker image for vulnerabilities before pushing, or use `pip-audit` to check Python dependencies.

### Add matrix testing
If you need to support multiple Python versions, use a matrix strategy in the CI job instead of maintaining separate workflow files.

### Pin action versions to SHAs
Instead of `actions/checkout@v4`, pin to a specific commit SHA for supply-chain security (e.g., `actions/checkout@<sha>`).

### Add artifact upload for coverage
Upload the coverage JSON as a GitHub Actions artifact so you can download and inspect it after a run, or post a coverage summary as a PR comment using something like `coverage-comment-action`.

### Add a manual deploy workflow
Create a `workflow_dispatch` workflow that lets you deploy a specific image tag on demand, useful for rollbacks or staging deployments.

### Consolidate the READMECICD.MD
The existing `READMECICD.MD` inside `.github/workflows/` duplicates what this document covers. Consider removing it or linking to this file instead to avoid drift.

---

## Test Coverage Gap Analysis

The current test suite has 6 test files covering a small fraction of the codebase. Below is a module-by-module breakdown of what exists, what's missing, and what tests need to be written so that CI can actually gate merges on whether the major functionality works.

### What's Currently Tested

| Test file | What it covers | Type |
|---|---|---|
| `test_strategy_api.py` | `AssetData`, `MarketData`, `PortfolioManager`, `StrategyContext` — data access, history, edge cases, buy/sell dispatch | Unit (mocked) |
| `test_relative_momentum_index.py` | `RelativeMomentumIndex` — readiness, value range, flat-price edge case | Unit |
| `test_backtest.py` | Runs full backtests for Portfolio 1, 2, 3 via `main_backtest.main()` | Integration (requires DB) |
| `test_trade_executor.py` | Wiring check: `tradeExecutor` + `RunEngine` can be constructed together | Integration (requires DB) |
| `test_api_endpoints.py` | `FMPMarketData` — historical, intraday, realtime, current price | Integration (requires FMP API key) |
| `tests_samp.py` | DB connectivity smoke test (`SELECT NOW()`) | Integration (requires DB) |

### What's Not Tested (by module)

#### Backtest Engine (`src/backtest/`)

The backtest system is the core of the project. Currently only tested via end-to-end runs in `test_backtest.py` which require a live DB connection. No isolated unit tests exist.

Tests needed:
- `BacktestExecutor.execute_trade()` — verify trade sizing, margin constraints, slippage application, buying power limits, cash updates, position updates, and trade log recording. This is the most critical untested function.
- `BacktestExecutor._apply_slippage()` — BUY increases price, SELL decreases price, other signals return unchanged price.
- `BacktestExecutor._calculate_buying_power()` — verify leverage math, gross position value calculation, zero-floor behavior.
- `BacktestExecutor.get_port_notional()` — verify cash + positions value calculation.
- `BacktestExecutor.get_data_feeds()` — verify the shape and content of CASH_EQUITY, POSITIONS, PORT_NOTIONAL DataFrames.
- `BacktestRunner._prepare_data()` — verify data fetching and cleaning pipeline (can mock the DB call).
- `BacktestRunner._run_event_loop()` — verify the event loop iterates correctly over timestamps and calls strategy `OnData`.
- `BacktestRunner._calculate_results()` — verify results DataFrame structure.
- `BacktestEngine.run()` — verify it loads portfolio configs, creates runners, and orchestrates execution.
- `generate_backtest_report()` — verify Sharpe ratio, max drawdown, rolling stats, monthly returns calculations against known inputs.

#### Live Trading Engine (`src/live_trading/`)

No unit tests exist. The only test (`test_trade_executor.py`) checks that objects can be constructed.

Tests needed:
- `tradeExecutor.execute_trade()` — verify the full trade flow: numeric conversion, signal validation, buying power calculation, price fetching, position sizing, and database update call. Mock the DB and FMP API.
- `tradeExecutor._calculate_buying_power()` — verify leverage math with real position DataFrames, including the safety check that returns 0 when a price fetch fails.
- `tradeExecutor.update_database()` — verify it writes to `trade_execution_logs`, updates `positions_book`, and updates `cash_equity_book` atomically.
- `RunEngine.load_portfolios()` — verify it reads `config.json` from each portfolio directory and instantiates the correct classes.
- `RunEngine._run_portfolio()` — verify the polling loop, failure counting, and circuit breaker logic (stops after `max_consecutive_failures`).

#### Portfolio Strategies (`src/portfolios/`)

Only `test_strategy_api.py` tests the API layer. No tests exist for individual strategy signal generation logic.

Tests needed:
- `VolMomentum.OnData()` — given known indicator values and portfolio state, verify it produces the correct BUY/SELL/HOLD signals. Test the risk-off mode (cash < 10% of total value), the volatility threshold logic, and the weight-based position sizing.
- `MomentumStrategy.OnData()` — verify SMA crossover logic, RSI/RMI overbought/oversold thresholds, DMA signals, and the combined signal aggregation.
- `RegimeAdaptiveStrategy.OnData()` — verify VIX-based regime detection (high vol → mean reversion via VWAP/ATR bands, low vol → momentum via RateOfChange), cooldown timer, and confidence decay.
- `TrendRotateStrategy.OnData()` — verify risk-on/risk-off group selection based on SMA trend, rotation execution (buy underweight, sell overweight, exit non-trending), and weight simulation.
- `CrossoverRmiStrategy.OnData()` — verify bullish/bearish SMA crossover detection, RMI confirmation, volatility stop-loss, risk-off mode, and max weight constraint.
- `BasePortfolio.AddIndicator()` — verify dynamic indicator loading, warmup with historical data, and error handling for unknown indicator names.
- `BasePortfolio.RegisterIndicatorSet()` — verify batch indicator registration across all tickers.

#### Indicators (`src/portfolios/indicators/`)

Only `RelativeMomentumIndex` has tests. The other 7 indicators have zero coverage.

Tests needed for each indicator (same pattern: feed known data, verify output):
- `SimpleMovingAverage` — verify readiness after `period` updates, correct average calculation, sliding window behavior.
- `RelativeStrengthIndex` — verify readiness, 0-100 range, known RSI values for trending/flat/reversing data.
- `AverageTrueRange` — verify it uses high/low/close correctly, readiness, known ATR values.
- `DisplacedMovingAverage` — verify displacement offset, readiness timing (needs `period + displacement` points).
- `RateOfChange` — verify percentage and absolute modes, readiness, known ROC values for trending data.
- `VWAP` — verify it uses both price and volume, readiness, known VWAP values.

#### Database Connector (`src/common/database/MQSDBConnector.py`)

Only tested via the smoke test in `tests_samp.py` (which just runs `SELECT NOW()`).

Tests needed:
- `execute_query()` — verify SELECT, INSERT, UPDATE, DELETE operations return correct status/data structures. Mock `psycopg2` for unit tests.
- `read_db()` — verify it returns `{'status': 'success', 'data': [...]}` format.
- `bulk_insert()` — verify batch insert with correct parameterization.
- Connection pooling — verify `get_connection()` returns a live connection, `release_connection()` returns it to the pool, `close_all_connections()` cleans up.
- Error handling — verify behavior when the pool is exhausted, when a connection is closed mid-query, when credentials are wrong.

#### Data Orchestration (`src/orchestrator/`)

Zero test coverage.

Tests needed:
- `backfill.backfill_data()` — verify batch date generation, CSV caching, data preparation (column ordering, ticker injection). Mock the FMP API.
- `backfill.prepare_data()` — verify column renaming, datetime parsing, ticker injection, column ordering.
- `backfill.generate_output_filename()` — verify dynamic filename generation for single/multiple tickers.
- `concurrent_backfill` — verify parallel execution doesn't corrupt shared state.
- `injectBackfill` — verify database injection from CSV files.
- `realtimeDataIngestor.process_market_data()` — verify data transformation, ticker filtering, volume delta calculation.
- `realtimeDataIngestor.initialize_volume_state()` — verify state recovery from DB.
- `FMPMarketData._make_request()` — verify retry logic, rate limiting, timeout handling. Mock `requests`.
- `FMPMarketData.get_historical_data()` — verify single-ticker and multi-ticker response parsing.

#### Risk Manager (`src/risk_manager/`)

Zero test coverage.

Tests needed:
- `DailyAllocator._get_current_cash()` — verify it reads cash from the correct portfolio.
- `DailyAllocator._get_positions_value()` — verify position valuation.
- `DailyAllocator._execute_internal_transfer()` — verify capital moves between portfolios correctly.
- `manage_capital.get_current_cash()` — verify DB query and return value.
- `manage_capital.update_capital()` — verify deposit/withdrawal operations.

#### Utilities and Helpers

Tests needed:
- `src/backtest/utils.fetch_historical_data()` — verify timezone handling (UTC → America/New_York), numeric coercion, NaN dropping, empty result handling.
- `src/portfolios/common.read_config_param()` — verify it reads the correct config.json from the caller's directory.
- `src/portfolios/toolkit.QuantToolkitAccessor.gaussian_smooth()` — verify smoothing output shape and values.
- `src/portfolios/toolkit.QuantToolkitAccessor.winsorize()` — verify outlier capping at specified quantiles for both Series and DataFrame.
- `src/common/auth/apiAuth.APIAuth` — verify it reads `FMP_API_KEY` from env, handles missing/empty key.

---

## Recommended CI Test Tiers

The workflow should be structured so that fast, isolated tests run first and gate the slower integration tests. All tiers must pass before a PR can merge.

### Tier 1: Unit Tests (no DB, no API, no network — must always pass)

These tests use mocks/fixtures only. They should run in under 60 seconds and catch the vast majority of regressions.

Mark with: `@pytest.mark.unit` (or simply no marker — run with `-m "not db and not api"`)

| Area | What to test |
|---|---|
| BacktestExecutor | Trade execution, slippage, buying power, position sizing, portfolio notional, data feeds |
| Backtest Reporting | Sharpe ratio, max drawdown, rolling stats, monthly returns against known inputs |
| All 8 Indicators | Readiness, value ranges, known outputs for known inputs |
| Strategy API | AssetData, MarketData, PortfolioManager, StrategyContext (already exists, expand) |
| Strategy Signal Logic | Each of the 5 strategies' OnData with mocked context — verify correct BUY/SELL/HOLD |
| Toolkit | gaussian_smooth, winsorize |
| Config Loading | read_config_param, BasePortfolio config parsing |
| Auth | APIAuth with/without env var set |
| Data Utils | fetch_historical_data with mocked DB, timezone handling |
| Backfill Helpers | prepare_data, generate_output_filename |
| FMP Client | _make_request retry/timeout logic with mocked requests |
| Risk Manager | Capital allocation math with mocked DB |

### Tier 2: Integration Tests — DB (requires secrets)

These tests hit the real database. Gate them behind the existing secrets check in `main.yml`.

Mark with: `@pytest.mark.db`

| Area | What to test |
|---|---|
| MQSDBConnector | Connection pooling, execute_query, bulk_insert, read_db, error recovery |
| Schema Validation | Run `SchemaDefinitions.create_all_tables()` against test DB, verify all 8 tables exist |
| Live Executor DB Writes | tradeExecutor.update_database() writes correct rows to trade_execution_logs, positions_book, cash_equity_book |
| Backtest End-to-End | Full backtest run for each strategy (already exists in test_backtest.py) |
| Backfill Injection | injectBackfill writes to market_data table correctly |
| Risk Manager DB | DailyAllocator reads/writes capital correctly |

### Tier 3: Integration Tests — External API (requires FMP key)

These tests hit the FMP API. Gate them behind an `FMP_API_KEY` secrets check.

Mark with: `@pytest.mark.api`

| Area | What to test |
|---|---|
| FMPMarketData | get_historical_data, get_intraday_data, get_realtime_data, get_current_price (already exists in test_api_endpoints.py) |

---

## Recommended Changes to `main.yml`

To enforce that all major functionality works before a PR merges:

```yaml
# Add to the CI job, after installing dependencies:

# Run Tier 1: Unit tests (always, no secrets needed)
- name: Run unit tests
  env:
    PYTHONWARNINGS: error
  run: |
    python -m pytest -q -m "not db and not api" \
      --cov=src \
      --cov-report=term-missing \
      --cov-report=json

# Run Tier 2: DB integration tests (only when secrets exist)
- name: Run DB integration tests
  if: steps.db.outputs.available == 'true'
  env:
    DB_HOST: ${{ secrets.DB_HOST }}
    DB_PORT: ${{ secrets.DB_PORT }}
    DB_NAME: ${{ secrets.DB_NAME }}
    DB_USER: ${{ secrets.DB_USER }}
    DB_PASSWORD: ${{ secrets.DB_PASSWORD }}
    DB_SSLMODE: ${{ secrets.DB_SSLMODE }}
    PYTHONWARNINGS: error
  run: |
    python -m pytest -q -m "db" \
      --cov=src \
      --cov-append \
      --cov-report=term-missing \
      --cov-report=json

# Run Tier 3: API integration tests (only when FMP key exists)
- name: Check FMP API key
  id: fmp
  run: |
    if [ -n "${{ secrets.FMP_API_KEY }}" ]; then
      echo "available=true" >> $GITHUB_OUTPUT
    else
      echo "available=false" >> $GITHUB_OUTPUT
    fi

- name: Run API integration tests
  if: steps.fmp.outputs.available == 'true'
  env:
    FMP_API_KEY: ${{ secrets.FMP_API_KEY }}
    PYTHONWARNINGS: error
  run: |
    python -m pytest -q -m "api" \
      --cov=src \
      --cov-append \
      --cov-report=term-missing \
      --cov-report=json

# Enforce coverage on src/ only
- name: Enforce per-file coverage
  run: |
    python scripts/check_per_file_coverage.py --min 50 --coverage-file coverage.json
```

Also update `pytest.ini` to register the custom markers:

```ini
[pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_functions = test_*
addopts = -ra
markers =
    db: tests that require a live database connection
    api: tests that require the FMP API key
filterwarnings =
    error::DeprecationWarning
    error::PendingDeprecationWarning
    error::FutureWarning
```

And update `pyproject.toml` coverage scope:

```toml
[tool.coverage.run]
source = ["src"]
omit = [
  "src/mqs_bot.egg-info/*",
  "*/__pycache__/*",
]
```

---

## Priority Order for Writing Tests

If you're tackling this incrementally, here's the order that gives the most protection per test written:

1. `BacktestExecutor` unit tests — this is the financial math core. A bug here means wrong P&L, wrong position sizes, wrong margin calculations. Pure Python, no mocks needed beyond the class itself.
2. All 7 remaining indicator unit tests — pure math, easy to write, catches regressions in signal generation inputs.
3. Strategy `OnData` unit tests (all 5 strategies) — mock the `StrategyContext` and verify each strategy produces correct signals for known market conditions. This is what catches "the strategy stopped buying when it should" type bugs.
4. `tradeExecutor.execute_trade()` unit test — mock DB and FMP, verify the live trading sizing/constraint logic matches the backtest executor.
5. `generate_backtest_report()` unit test — feed known portfolio values, verify Sharpe/drawdown/returns are correct.
6. `FMPMarketData` retry/rate-limit unit tests — mock `requests`, verify the client handles failures gracefully.
7. DB connector unit tests — mock `psycopg2`, verify query execution and error handling.
8. Backfill helper unit tests — verify data preparation and filename generation.
9. Risk manager unit tests — verify capital allocation math.
10. Expand existing `test_strategy_api.py` — it's already good, but add edge cases for `StrategyContext._trade()` validation logic.

---

## Summary

The CI pipeline in `main.yml` has a solid structure (lint → unit tests → DB tests → coverage → Docker build/push), but the test suite behind it is thin. Most of the codebase's critical paths — trade execution, indicator calculations, strategy signal logic, data pipelines — have no unit tests. The CI currently gates on coverage percentage, but that's meaningless if the tests themselves don't exercise the important code paths.

The highest-impact change is writing Tier 1 unit tests for the `BacktestExecutor`, all indicators, and all strategy `OnData` methods. These require no secrets, no DB, no API keys — they run everywhere, fast, and catch the bugs that actually cost money.
