# MQS Trading System

MQSMaster contains the live trading loop, historical backtests, market-data ingestion, NLP sentiment pipeline, relevance-based prediction (RBP), portfolio strategies, and capital-management tools. The Python package is named `mqs_bot`; application code lives in `src/`, with research packages in `NLP/` and `RBP/`.

Start with the [full live-stack diagram](docs/workflows/live_trading_workflow.md), [real-time ingestion diagram](docs/workflows/realtime-ingestor-workflow.md), or [NLP overview](docs/workflows/nlp_workflow.md). The [documentation index](docs/README.md) links to detailed workflows.

## 1. Prerequisites

- Git and Python **3.10** for the local instructions below (the dev PR workflow uses 3.10). Main CI also tests 3.11/3.12; the Docker image uses 3.12. Do not assume arbitrary newer Python versions work with the pinned scientific dependencies.
- A PostgreSQL database and a user allowed to create the application tables, read data, and write results. An existing team development database or a local PostgreSQL installation works. Cloning the repository does not provision a database or historical data.
- An FMP API key with access to the news and price endpoints used by the projects you run.
- For NLP scoring, the separate fine-tuned model download described in [NLP setup](NLP/README.md#setup-download-the-finbert-model).
- For the complete `start.sh` supervisor, a **Linux environment** with Bash, `curl`, `jq`, and `/proc`: use WSL2 on Windows or a Linux container on macOS. Individual Python modules can run natively on either OS.

Run commands from the repository root unless a section says otherwise. Long-running commands each need their own terminal and activated environment.

## 2. Clone and create a virtual environment

### Windows (PowerShell)

Install Git and Python 3.10 first, then:

```powershell
git clone https://github.com/MUNQuantSociety/MQSMaster.git
Set-Location MQSMaster
py -3.10 -m venv MQS
. .\MQS\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
# Explicit runtime dependencies used by NLP; some otherwise arrive transitively.
python -m pip install psutil aiohttp beautifulsoup4
Copy-Item .env.example .env
notepad .env
```

If PowerShell blocks activation, allow it for this terminal with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. You can also invoke `.\MQS\Scripts\python.exe` directly. On later visits, activate the existing environment instead of recreating it. Do not copy `.env.example` over an already configured `.env`.

### macOS (Terminal: zsh or Bash)

Install Git and Python 3.10 first, then:

```bash
git clone https://github.com/MUNQuantSociety/MQSMaster.git
cd MQSMaster
python3.10 -m venv MQS
source MQS/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install psutil aiohttp beautifulsoup4
cp .env.example .env
open -e .env
```

Use the same Python architecture as your installation on Apple Silicon. A Windows venv cannot be reused on macOS, Linux, or WSL; create one inside each environment. `deactivate` leaves the venv on both systems.

`requirements.txt` pins pandas to `2.2.2` and numpy to `<=1.26.4`. Installing the editable package alone does **not** install this requirements file. The instructions explicitly install NLP's `psutil`, `aiohttp`, and `beautifulsoup4` imports because they are not all declared directly in that file.

## 3. Configure credentials and initialize the database

Fill in the root `.env` using your database and API credentials:

```dotenv
db_user=your_database_user
password='your_database_password'
host=localhost
port=5432
database=mqs_dev
sslmode=prefer
FMP_API_KEY=your_fmp_key
ALPHA_KEY=
APIFY_KEY=
```

The database keys are lowercase and case-sensitive in the application. Use the SSL mode required by your database host. `ALPHA_KEY` enables optional Alpha Vantage news; `APIFY_KEY` is needed for the optional Truth Social scraper. Keep `.env` out of Git. `start.sh` sources it as Bash, so values must also be valid shell assignments; quote passwords containing spaces or shell metacharacters.

Create the database first if you are using a new local PostgreSQL instance. With PostgreSQL's client tools on your PATH, the following example prompts for the password of a role that has database-creation privileges (replace the role name):

```bash
createdb -h localhost -U your_database_user -W mqs_dev
```

Set `.env` to that database and role. With the venv activated, check connectivity and bootstrap tables:

```bash
python -m src.common.database.test
python -m src.common.database.create_all_tables
```

Read the printed connection/schema results: these helper scripts can print failures without a nonzero exit status. Table creation is not a migration system and does not fill historical prices.

**NLP schema prerequisite:** the base schema creates `market_data.avg_sentiment`, while the NLP writer updates `market_data.sentiment_score`. On a fresh development database, add the writer's column before running DB-backed NLP (execute in your PostgreSQL client):

```sql
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS sentiment_score DOUBLE PRECISION;
```

The NLP repository adds `news_sentiment.content_length` and creates its article-URL unique index at initialization. Existing duplicate URLs can prevent that index from being created; check startup logs. See the [detailed NLP workflow](NLP/WORKFLOW.md#database-contract).

## 4. Run the projects

All commands below work in the activated native Windows/macOS venv unless marked Linux-only. They use the database named in `.env`.

| Project | Setup / configuration | Run from repository root | Result |
|---|---|---|---|
| Live bot | DB, recent market data, portfolio capital; choose classes in `src/main.py` | `python -m src.main` | Portfolio loops write cash, positions, and execution logs |
| Real-time ingestor | FMP key, schema, `src/orchestrator/backfill/tickers.json` | `python -m src.orchestrator.realTime.realtimeDataIngestor` | Quote snapshots in `market_data`, target interval 60 seconds |
| PnL worker | Existing books and market prices | `python -m src.orchestrator.realTime.pnl_script` | Valuations in `pnl_book`, target interval 60 seconds |
| Backtest | Historical data; edit dates, classes, and mode in `src/main_backtest.py` | `python -m src.main_backtest` | Reports and CSVs under `src/backtest/data/` |
| NLP live | Model, news API key, NLP schema above | `python -m NLP.main_NLP` | Article/score CSVs, `news_sentiment`, recent market sentiment |
| RBP research | Sufficient historical DB prices; edit `RBP/config.py` | `python -m RBP.main_rbp` | `rbp_predictions.csv` and `rbp_rbi_scores.csv` in the working directory |
| RBP forecast worker | DB history; positive portfolio weights and ticker configs | `python -m src.orchestrator.rbp_runner` | Forecast rows in `rbp_forecasts`, 300-second pause between refreshes |
| OMS | Set `OMS.enabled` in a strategy config that uses context order routing | Runs inside `python -m src.main` or event-mode `python -m src.main_backtest` | In-memory parent/child orders and scheduled fills; no standalone service |
| Daily capital allocation | Fund master portfolio; review manager weights | `python -m src.risk_manager.daily_allocator` | Internal cash transfers; one run, not a daemon |
| CFA calculator | Shared Python environment; no market database needed | `python -m scripts.CFA.src.cli` | Interactive calculator |
| Backtest analysis | Existing report folders | `python scripts/Backtest_Analysis/backtest_entrypoint.py read --sample-rows 3` | Report summary in terminal |

The live executor uses current prices and settles fills in PostgreSQL; this path does not submit orders to an external broker. The live entrypoint currently selects P1 and P2. Positive manager weights do not automatically start additional strategy threads. Review [portfolio setup and workflow](src/portfolios/README.md) before changing the selection.

OMS configuration supports `default_algo` (`MARKET`, `TWAP`, or `VWAP`), `duration_minutes`, and slicing settings. P1's checked-in config shows a 30-minute TWAP split into ten slices. Strategies that call the executor directly bypass the context OMS route; see the portfolio guide before assuming a config flag changes their execution.

### Historical market data and capital

Backfill dates use **DDMMYY** (NLP dates instead use ISO `YYYY-MM-DD`):

```bash
python -m src.orchestrator.backfill.backfill_cli concurrent --start 010125 --end 310125 --tickers AAPL MSFT --interval 1 --threads 4 --on-conflict ignore
```

Select enough history for the strategies you use; RBP's longest return feature needs 252 daily observations plus its forward-target window. Provider coverage controls what can be fetched. See the [backfill CLI](src/orchestrator/backfill/Readme.md) and [ticker refresh guide](src/orchestrator/backfill/update/refresh_README.md).

For a development database, the following example records USD 100,000 in the master book, then allocates capital according to `portfolio_manager_config.json`. These commands change book balances:

```bash
python -m src.risk_manager.manage_capital --action ADD --amount 100000
python -m src.risk_manager.daily_allocator
```

For a native local live session, start the ingestor and PnL worker in separate terminals, then the bot. Start the RBP/NLP workers when their prerequisites are ready. Standalone Python loops do not inherit the launcher's market-close shutdown; stop each foreground process with Ctrl+C.

### NLP: download, score, and inspect

Place the team model at `NLP/finbert-combined-final/` following [NLP/README.md](NLP/README.md). The scorer has no automatic model fallback. To fetch articles without a database or model, then score and persist them after those prerequisites are ready:

```bash
python -m NLP.fetch_articles AAPL,MSFT 2025-01-01 2025-12-31 --fmp-only --restart
python -m NLP.process_sentiment_pipeline AAPL MSFT
python -m NLP.update_database --query AAPL MSFT
```

For historical fetch + score + persistence in one command:

```bash
python -m NLP.backfill_NLP --start 2025-01-01 --end 2025-12-31 --tickers AAPL MSFT
```

Backfill refuses to start during weekday US cash-session hours; `--wait` waits for the close. The helper has no exchange-holiday calendar. For local notebook comparisons (no PostgreSQL), follow the preserved [two-model experiment](NLP/README.md#local-two-model-experiment-windows); optionally install Jupyter with `python -m pip install jupyterlab`, register the documented kernel, and run `python -m jupyterlab`.

### RBP: research and forecast service

`python -m RBP.main_rbp` runs the configurable research split and exports predictions and relevance-based importance (RBI). It needs no downloaded model. `python -m src.orchestrator.rbp_runner` runs the DB forecast service. These are separate from Portfolio 5's local model and Portfolio 8's ranking extension. See [RBP setup](RBP/README.md) and [detailed diagrams](RBP/WORKFLOW.md), including the current forecast freshness limitation.

### Complete supervisor: Linux / WSL / Docker

For WSL, clone inside the Linux filesystem and create `MQS` with Linux Python using the macOS-style venv commands above. Install `curl` and `jq` in the Linux environment, configure `.env`, install the NLP model, then:

```bash
bash start.sh
```

On Windows or macOS with a Linux Docker engine, build from the repository root after downloading the model:

```bash
docker build -t mqsmaster-local .
docker run --rm --name mqsmaster-local --env-file .env mqsmaster-local
```

The image creates its own Linux venv. `.env` is excluded from the image and injected at runtime; when using a database on the Docker Desktop host, set `host=host.docker.internal` in the environment file used for that container. Stop it from another terminal with `docker stop mqsmaster-local`.

The full launcher also starts the **retention pruner**, which can delete old `market_data` rows. It needs AWS SSM read/write access for its last-run parameter (default `/mqsmaster-prod/jobs/market_data_prune_last_run`, region `us-east-2`); use a separate development parameter and database for local experiments. Default retention is 545 days, checked daily and gated to at least 365 days between successful prunes. Individual-module commands above avoid launching this maintenance job.

Read the [launcher lifecycle](docs/workflows/live_trading_workflow.md) before using it: workers launch before the first market check; persistent watchers are detached, restart after every exit, and are not stopped by market close. In a container they still end when the container ends. `SKIP_PERSISTENT_SCRIPTS` appears in `.env.example` but is not read by the current `start.sh`.

## 5. Verify and troubleshoot

```bash
python -m pip check
python -m pytest -m "smoke and workflow_backtest"
```

Tests enforce strict markers and treat pandas/deprecation/future warnings as errors. For an explicitly offline smoke selection, use `python -m pytest -m "smoke and not db and not api"`; DB/API markers require those services. See [test modes](docs/TEST_MODES.md).

| Symptom | Check |
|---|---|
| `No module named src`, `NLP`, or `RBP` | Activate the venv, run from the repository root, and install `-e .`. Prefer the module commands above. |
| Missing Python import | Install `requirements.txt` and the explicit NLP dependencies above in the same interpreter. |
| Missing FinBERT directory | Download the model; check for an accidental extra nested folder. |
| No trades or empty research results | Verify data dates, ticker coverage, warmup, capital, strategy selection, and RBP split date. |
| NLP article insert works but market sync fails | Check `market_data.sentiment_score` and repository retry warnings. |
| Ingestor cannot write `/var/log/...` | It continues on stderr; set `MARKET_DATA_INGESTOR_LOG` to a writable path for file logs. |
| Launcher reports a skipped/failed worker | Read its traceback and startup summary; other workers can still be running. |

## Project guides

- [Documentation index](docs/README.md) and [workflow index](docs/workflows/README.md)
- [Live trading details](docs/workflows/live-trading-workflow_detailed.md)
- [Ingestor setup and behavior](src/orchestrator/realTime/README.md)
- [NLP commands](NLP/README.md) and [detailed NLP workflow](NLP/WORKFLOW.md)
- [RBP setup](RBP/README.md) and [detailed RBP workflow](RBP/WORKFLOW.md)
- [Portfolio development and workflow](src/portfolios/README.md)
- [OMS design and implementation status](docs/OMS/OMS_DESIGN.md)
- [Capital management](docs/workflows/capital-management.md), [CFA calculator](scripts/CFA/README.md), [analysis tools](scripts/Backtest_Analysis/README.md), and [CI/CD](docs/CICD/CICD.md)
