# Test Modes and Markers — Comprehensive Guide

MQS uses a tiered testing strategy with pytest markers to organize tests by scope, performance, and external dependencies. This document explains the complete testing architecture and how to work with each tier.

## Overview

The test suite is organized into four tiers:

1. **Smoke** — Fast PR gate (2 min, deterministic)
2. **Deep/Slow** — Scheduled full integration (10+ min)
3. **Integration (DB)** — Deterministic external resources
4. **Synthetic Monitoring** — Health checks outside pytest

---

## Test Tiers

### 1. Smoke (Default PR Gate)

**Purpose**: Fast, deterministic unit tests validating core functionality without external dependencies.

- **Duration**: < 2 minutes
- **External dependencies**: None (all mocked)
- **Marker**: `@pytest.mark.smoke`
- **CI trigger**: Every PR to `main`
- **Local**: Run before commit

**When to use:**
- Testing business logic (strategy calculations, portfolio math, indicators)
- Core trade execution validation
- API instantiation and basic flow

**Example:**
```python
@pytest.mark.smoke
def test_strategy_api_init():
    """Validate strategy API instantiation."""
    strategy = StrategyAPI()
    assert strategy is not None
```

**Run locally:**
```bash
pytest -m smoke
```

### 2. Deep / Slow (Scheduled Runs)

**Purpose**: Longer-running tests including concurrency, stress, edge cases, and real data sources.

- **Duration**: 5–15 minutes
- **External dependencies**: May use real APIs, databases, market data
- **Markers**: `@pytest.mark.slow`, `@pytest.mark.e2e`
- **CI trigger**: Scheduled every 6 hours
- **Local**: Run before major commits or when testing complex workflows

**When to use:**
- Full end-to-end workflows (entire backtest cycle)
- Concurrency and race condition testing
- Multi-day indicator warmup validation
- Database schema and migration tests
- Live engine resilience under load

**Example:**
```python
@pytest.mark.slow
@pytest.mark.workflow_backtest
def test_full_backtest_workflow(deep_window):
    """Validate complete backtest from config to teardown."""
    start, end = deep_window  # 90-day window
    result = run_full_backtest(start, end)
    assert result.total_trades > 0
```

**Run locally:**
```bash
pytest -m "slow or e2e"
```

### 3. Integration / Database Tests

**Purpose**: Deterministic tests requiring stable external resources (database, APIs) but with mocked or recorded responses where possible.

- **Marker**: `@pytest.mark.db`
- **Environment**: Requires `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_SSLMODE`
- **CI trigger**: Only when database secrets are available (3.11 only)

**When to use:**
- Testing database queries and schema
- Validating ORM/connector behavior
- Testing data migrations

**Example:**
```python
@pytest.mark.db
def test_portfolio_query(db_connection):
    """Validate portfolio data retrieval from database."""
    portfolios = db_connection.get_portfolios()
    assert len(portfolios) > 0
```

**Run locally (with DB):**
```bash
# Ensure environment variables are set
export DB_HOST=localhost DB_PORT=5432 DB_NAME=mqs_test ...
pytest -m db
```

**Skip if unavailable:**
```python
@pytest.mark.db
def test_with_database(require_db_env):
    """Fixture will skip if DB credentials missing."""
    pass
```

### 4. Synthetic Monitoring (Scheduled Health Checks)

**Purpose**: Non-pytest health checks validating API availability, latency, and response schemas outside the test cycle.

- **Marker**: `@pytest.mark.synthetic` (metadata only)
- **CI trigger**: Scheduled jobs in GitHub Actions (every 6 hours)
- **Scope**: FMP API provider checks, NLP pipeline health, latency validation

**When to use:**
- Monitoring external service availability
- Validating API response schemas haven't changed
- Detecting latency regressions
- NLP pipeline heartbeat checks

**Scripts:**
```bash
# FMP API provider checks
python scripts/api_test.py --symbol AAPL --require-fmp

# NLP pipeline health
python NLP/monitor_daemon.py --synthetic --max-log-age-hours 72
```

---

## Marker Reference

### Core Tier Markers

| Marker | Purpose | Scope | PR? | Scheduled? |
|--------|---------|-------|-----|-----------|
| `smoke` | Fast unit tests, mocked | Core logic | ✅ | ✅ |
| `slow` | Long-running tests | Integration | ❌ | ✅ |
| `e2e` | End-to-end workflows | System | ❌ | ✅ |
| `integration` | Multi-module integration | Cross-module | ❌ | ✅ |
| `db` | Requires live database | Database | ✅* | ✅ |
| `synthetic` | Scheduled monitoring (non-pytest) | External health | ❌ | ✅ |

*Only runs if `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` are configured in CI secrets.

### Workflow Markers

Combine with tier markers to test specific features. Can be marked `smoke` (fast path) or `slow` (deep path).

| Marker | Feature | Example |
|--------|---------|---------|
| `workflow_backfill` | Backfill data pipelines | `@pytest.mark.smoke @pytest.mark.workflow_backfill` |
| `workflow_live` | Live trading engine | `@pytest.mark.slow @pytest.mark.workflow_live` |
| `workflow_backtest` | Backtesting engine | `@pytest.mark.smoke @pytest.mark.workflow_backtest` |
| `workflow_indicators` | Indicator calculations | `@pytest.mark.smoke @pytest.mark.workflow_indicators` |
| `workflow_nlp` | NLP/sentiment analysis | `@pytest.mark.slow @pytest.mark.workflow_nlp` |

### Legacy Markers (Being Migrated)

| Marker | Status | Migration Path |
|--------|--------|-----------------|
| `api` | Deprecated | Move to scheduled synthetic monitoring in `scripts/api_test.py` |

---

## Common Commands

### Run all smoke tests (PR gate)
```bash
pytest -m smoke
```

### Run all deep/scheduled tests
```bash
pytest -m "slow or e2e"
```

### Run database integration tests
```bash
# Requires DB environment variables
pytest -m db
```

### Run a specific workflow (smoke tier)
```bash
pytest -m "smoke and workflow_backtest"
```

### Run a specific workflow (deep tier)
```bash
pytest -m "slow and workflow_live"
```

### Run all tests except slow/e2e
```bash
# Typical local dev: smoke + db (if available)
pytest -m "not slow and not e2e and not synthetic"
```

### Run with coverage
```bash
pytest -q --cov=src --cov-report=term-missing
```

### Show test markers and counts
```bash
pytest --collect-only -q
```

### Find unknown markers (strict mode)
```bash
pytest --strict-markers --collect-only 2>&1 | grep "unknown"
```

### Profile slow tests
```bash
pytest --durations=10
```

### Run verbose with short tracebacks
```bash
pytest -v --tb=short tests/test_backtest.py
```

---

## Environment Variables

### Required for Smoke Tests
None — smoke tests are fully mocked.

### Optional for Deep / Integration Tests
```bash
# PostgreSQL connection (for @pytest.mark.db tests)
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=mqs_test
export DB_USER=mqs_user
export DB_PASSWORD=your-password
export DB_SSLMODE=require

# FMP API (for legacy @pytest.mark.api tests; prefer synthetic monitoring)
export FMP_API_KEY=your-api-key
```

### Required for Synthetic Monitoring Scripts
```bash
# For scheduled FMP provider checks (main branch only)
export FMP_API_KEY=your-api-key

# For NLP pipeline health checks
export NLP_ENDPOINT=http://localhost:5000
```

---

## CI/CD Integration

### Pull Request Workflow

1. **Smoke tests** (always) — validates core functionality
2. **Database integration** (if secrets available) — validates schema and queries
3. **Coverage enforcement** — per-file minimum thresholds

```bash
# Run by CI automatically
pytest -m "not db and not api and not slow and not e2e and not synthetic" --cov=src
pytest -m "db" --cov=src --cov-append  # if DB secrets exist
```

**Failure criteria:**
- Any test fails → PR blocked
- Coverage below threshold → PR blocked

### Scheduled Workflow (Every 6 Hours)

1. **Smoke tests** (quick validation)
2. **Deep/slow tests** (full integration)
3. **Synthetic monitoring** (API/NLP health)

```bash
# Smoke
pytest -q -m "not slow and not e2e"

# Deep
pytest -q -m "slow or e2e"

# Synthetic (outside pytest)
python scripts/api_test.py --symbol AAPL --require-fmp
python NLP/monitor_daemon.py --synthetic --max-log-age-hours 72
```

**Failure criteria:**
- Test failures reported as CI job failure (non-blocking for synthetic)
- Synthetic failures logged to artifacts/alerts (PR not blocked)

---

## Best Practices

### Writing Tests

#### 1. Always mark tests with at least one tier marker

```python
# Fast, no external deps
@pytest.mark.smoke
def test_strategy_api_init():
    strategy = StrategyAPI()
    assert strategy is not None

# Longer-running, real data
@pytest.mark.slow
def test_full_backtest_workflow():
    result = run_backtest()
    assert result.total_trades > 0

# Database test
@pytest.mark.db
def test_portfolio_query(db_connection):
    portfolios = db_connection.get_portfolios()
    assert len(portfolios) > 0
```

#### 2. Use workflow markers to categorize by feature

```python
@pytest.mark.smoke
@pytest.mark.workflow_backtest
def test_backtest_initialization():
    """Fast backtest smoke test."""
    pass

@pytest.mark.slow
@pytest.mark.workflow_live
def test_live_engine_resilience():
    """Deep live trading resilience test."""
    pass
```

#### 3. Use shared fixtures for window sizes

```python
# Smoke tests get 21-day window
@pytest.mark.smoke
def test_indicator_warmup(smoke_window):
    start, end = smoke_window
    result = calculate_indicator(start, end)
    assert result is not None

# Deep tests get 90-day window
@pytest.mark.slow
def test_long_backtest(deep_window):
    start, end = deep_window
    result = run_backtest(start, end)
    assert result.total_trades > 0
```

**Fixtures defined in [conftest.py](../tests/conftest.py):**
```python
@pytest.fixture
def smoke_window():
    """21-day window for fast tests."""
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=21)
    return start, end

@pytest.fixture
def deep_window():
    """90-day window for deep tests."""
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=90)
    return start, end
```

#### 4. Skip tests conditionally if environment not available

```python
@pytest.mark.db
def test_with_database(require_db_env):
    """Will skip if DB credentials missing."""
    # Test runs only if all DB_* env vars are set
    pass

@pytest.mark.api
def test_with_fmp(require_fmp_env):
    """Will skip if FMP_API_KEY missing."""
    # Test runs only if FMP_API_KEY is set
    pass
```

#### 5. Mock external services in smoke tests

```python
@pytest.mark.smoke
@pytest.mark.workflow_live
def test_order_execution(mocker):
    """Test order execution with mocked broker."""
    mock_broker = mocker.Mock()
    mock_broker.place_order.return_value = {"status": "filled"}
    
    executor = TradeExecutor(broker=mock_broker)
    result = executor.execute_trade()
    
    assert result["status"] == "filled"
```

### Running Tests Locally

#### 1. Before committing any code
```bash
# Smoke only (fast feedback)
pytest -m smoke
```

#### 2. Before pushing to PR
```bash
# Full suite (smoke + available integrations)
pytest -q
```

#### 3. Before major feature merges
```bash
# Include deep tests if time permits
pytest -q -m "slow or e2e"
```

#### 4. Debugging a failing test
```bash
# Verbose output with short traceback
pytest -v --tb=short tests/test_backtest.py::test_specific_test

# With print statements visible
pytest -v -s tests/test_backtest.py::test_specific_test
```

#### 5. Checking coverage locally
```bash
pytest -q --cov=src --cov-report=term-missing
```

### Debugging & Diagnostics

#### Check registered markers
```bash
pytest --collect-only -q | head -30
```

#### List all tests by marker
```bash
pytest --collect-only -q | grep "@pytest.mark.smoke"
```

#### Find tests that take longest
```bash
pytest --durations=10
```

#### Validate markers in strict mode
```bash
pytest --strict-markers --collect-only 2>&1 | grep "unknown"
```

#### Run single test with maximum verbosity
```bash
pytest -vv -s tests/test_backtest.py::test_name --tb=long
```

---

## Migration Notes

### Legacy `api` Marker
Tests marked `@pytest.mark.api` are being migrated to scheduled synthetic monitoring jobs:

- **Old**: `@pytest.mark.api` pytest tests with real API calls (blocking PR)
- **New**: Scheduled synthetic monitoring in `scripts/api_test.py` (non-blocking)

**Migration steps:**
1. Remove `@pytest.mark.api` from test
2. Move logic to `scripts/api_test.py` as synthetic probe
3. Add to scheduled workflow in `.github/workflows/main.yml`

### No Direct `main_backtest.main` Calls
Tests should use stable programmatic seams instead of CLI:

```python
# ❌ Old way (fragile)
from src.main_backtest import main
main(config_file="config.yaml")

# ✅ New way (stable)
from src.main_backtest import init_backtest, run_backtest
config = init_backtest("config.yaml")
result = run_backtest(config)
```

### Strict Marker Enforcement
All markers must be registered in [pytest.ini](../pytest.ini). Unknown markers will cause failures:

```bash
# ❌ Will fail
@pytest.mark.unknown_marker
def test_something():
    pass

# ✅ Must register first in pytest.ini
# Then:
@pytest.mark.smoke  # registered
def test_something():
    pass
```

---

## See Also

- [pytest.ini](../pytest.ini) — Marker definitions and configuration
- [tests/conftest.py](../tests/conftest.py) — Shared fixtures and environment checks
- [tests/README_TEST_MODES.md](../tests/README_TEST_MODES.md) — Quick command reference
- [.github/workflows/main.yml](../.github/workflows/main.yml) — CI/CD configuration and scheduled jobs
- [plan-mqsTestRobustness.prompt.md](../plan-mqsTestRobustness.prompt.md) — Detailed strategy and roadmap
