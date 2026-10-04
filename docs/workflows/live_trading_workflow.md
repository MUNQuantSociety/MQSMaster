# Live trading stack: from start.sh to shutdown

[Setup and commands](../../README.md) · [Inside the bot](live-trading-workflow_detailed.md) · [Ingestor](realtime-ingestor-workflow.md) · [NLP](nlp_workflow.md)

This is the behavior of the checked-in [start.sh](../../start.sh), including its supporting processes. It describes repository code, not the status of a deployed task.

## Startup and process tree

```mermaid
flowchart TD
    START["bash start.sh"] --> ENV["Resolve repository and MQS/bin/python<br/>Create .env from injected variables if absent<br/>Source .env"]
    ENV --> CHECK{"Required credentials, curl, jq<br/>and executable Python present?"}
    CHECK -->|No| EXIT["Exit 1"]
    CHECK -->|Yes| VALIDATE["Set working directory and PYTHONPATH<br/>Validate each script: exists, readable, compiles"]
    VALIDATE --> SKIP["Record invalid scripts as skipped<br/>Build DB, persistent, and market run lists"]
    SKIP --> DB["Run validated DB scripts sequentially<br/>database/test.py then create_all_tables.py"]
    DB --> DBOK{"A DB script exits nonzero?"}
    DBOK -->|Yes| EXIT
    DBOK -->|No| PERSIST["Launch persistent watchers first<br/>Skip worker if matching process already exists"]
    PERSIST --> NLP["NLP/main_NLP.py<br/>News fetch, FinBERT, sentiment persistence"]
    PERSIST --> PRUNE["retention/prune_market_data.py<br/>SSM-gated deletion of old price rows"]
    PERSIST --> MARKET["Launch market scripts sequentially<br/>3-second startup grace for each by default"]
    MARKET --> BOT["src/main.py<br/>Portfolio threads and optional OMS pump"]
    MARKET --> ING["realTime/realtimeDataIngestor.py<br/>Price ingestion"]
    MARKET --> PNL["realTime/pnl_script.py<br/>Book valuation"]
    MARKET --> RBP["orchestrator/rbp_runner.py<br/>Forecast refresh"]
    MARKET --> SUMMARY["Report running, skipped, and failed scripts"]
    SUMMARY --> ANY{"Any market PID survived startup?"}
    ANY -->|No| EXIT
    ANY -->|Yes| WATCH["Log memory and enter NASDAQ market watchdog"]
```

Missing scripts and syntax failures skip only that script. Import/initialization failures surface in the startup grace check; other workers still launch. DB scripts that return nonzero abort startup, but the current DB helpers can print errors and return zero, so their printed output matters.

**Launch order is not a readiness dependency:** the bot starts before the ingestor. Historical data and initial capital must already be prepared. The first market-hours check occurs **after** workers launch, so starting outside market hours can briefly run the market workers before they are terminated.

## How data reaches the trading books

```mermaid
flowchart TD
    FMP["FMP quote feeds"] --> ING["Real-time ingestor"]
    BF["Historical backfill"] --> MD[("market_data")]
    ING --> MD
    MD --> STRAT["Portfolio data + indicators<br/>OnData"]
    NEWS["FMP and optional news providers"] --> NLP["NLP fetch and FinBERT"]
    NLP --> NS[("news_sentiment")]
    NLP -->|"Recent daily sentiment sync"| MD
    NS -.->|"P7 when selected"| STRAT
    MD --> RBP["RBP forecast service"]
    RBP --> FC[("rbp_forecasts")]
    STRAT --> SIGNAL["Buy / sell and confidence"]
    SIGNAL --> SIZE["Shared live executor sizing"]
    FC -.->|"Optional RBP confidence overlay"| SIZE
    SIZE --> ROUTE{"Execution route"}
    ROUTE -->|Direct| FILL["Fill using current FMP quote"]
    FMP --> FILL
    ROUTE -->|OMS enabled| OMS["Per-portfolio parent and child orders"]
    OMS --> PUMP["Dedicated OMS tick thread<br/>Read portfolio state at fill time"]
    PUMP --> FILL
    FILL --> BOOKS[("cash_equity_book<br/>positions_book<br/>trade_execution_logs")]
    BOOKS --> PNL["PnL worker"]
    MD --> PNL
    PNL --> PB[("pnl_book")]
    BOOKS --> STRAT
    PB --> STRAT
```

The live executor records fills in PostgreSQL; this path contains no external broker submission. The RBP overlay is constructed only when the manager config contains `rbp_overlay.enabled: true`; it is absent in the checked-in manager config. NLP is consumed by sentiment-aware strategies such as P7, not automatically by every portfolio. P1 and P2 are the current `src/main.py` selection.

## Market watchdog and persistent-worker lifecycle

```mermaid
flowchart TD
    CHECK["Check FMP NASDAQ exchange-market-hours"] --> STATUS{"Boolean market status?"}
    STATUS -->|Open| RESET["Reset unknown streak<br/>Check market PIDs; report exited workers"]
    RESET --> ANY{"Any market worker alive?"}
    ANY -->|Yes| MEM["Log worker memory<br/>Sleep 180 seconds"]
    MEM --> CHECK
    ANY -->|No| FAIL["Exit watchdog with code 1"]
    STATUS -->|Unknown or API error| COUNT["Increment unknown streak"]
    COUNT --> LIMIT{"Streak reaches limit?<br/>Default 20 checks"}
    LIMIT -->|No| RETRY["Keep workers running<br/>Sleep 180 seconds"]
    RETRY --> CHECK
    LIMIT -->|Yes| STOP["SIGTERM surviving market PIDs<br/>Wait for termination; exit watchdog"]
    STATUS -->|Closed| STOP

    WORKER["Persistent watcher starts Python worker"] --> DONE["Worker exits with any code"]
    DONE --> BACKOFF["Append exit to logs/NAME.watcher.log<br/>Sleep 30 seconds"]
    BACKOFF --> WORKER
```

Market workers are **not restarted** by the watchdog after a crash. Persistent watchers restart after both successful and failed exits. They start before the market workers, use `nohup`/detachment, and survive normal watchdog exit on a host. A container stopping terminates its remaining processes; these are not independent always-on services merely because they are called persistent.

`start.sh` has no general signal-cleanup trap. Do not treat Ctrl+C on the supervisor as proof that all children stopped. For local development, foreground module commands make process ownership explicit; for a container, stop the container. The `SKIP_PERSISTENT_SCRIPTS` key in `.env.example` is not implemented in this script.

## Process reference

| Worker | Cadence / condition | Output |
|---|---|---|
| Bot | Per-portfolio `INTERVAL`; OMS pump defaults to 5 seconds | Execution and book updates |
| Ingestor | Target 60 seconds including work | `market_data`; stderr and optional file log |
| PnL | Target 60 seconds including work | `pnl_book` |
| RBP runner | Work, then 300-second interruptible sleep | `rbp_forecasts` |
| NLP runner | Target 300 seconds including work; longer sweeps run back-to-back | CSVs, DB sentiment, `logs/daemon.log` |
| Retention | Daily check; minimum 365 days between successful prunes | Deletes price rows older than 545 days by default; records completion in SSM |

Daily capital allocation and historical backfill are separate commands; `start.sh` does not schedule them. Persistent logs are `logs/main_NLP.watcher.log` and `logs/prune_market_data.watcher.log`. The launcher uses Linux `/proc` for duplicate-process checks and memory reporting, so native macOS Bash does not provide its full supervision behavior.
