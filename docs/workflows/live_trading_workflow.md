# Live trading stack: from start.sh to shutdown

[Setup and commands](../../README.md) · [Inside the bot](live-trading-workflow_detailed.md) · [Ingestor](realtime-ingestor-workflow.md) · [NLP](nlp_workflow.md)

This is the behavior of the checked-in [start.sh](../../start.sh), including its supporting processes. It describes repository code, not the status of a deployed task.

## Startup and process tree

```mermaid
flowchart TD
    START["bash start.sh"] --> ENV["Resolve repository and MQS/bin/python<br/>Create .env from injected variables if absent<br/>Source .env"]
    ENV --> CHECK{"Required credentials, curl, jq<br/>and executable Python present?"}
    CHECK -->|No| ENVFAIL["Exit 1: preflight failed"]
    CHECK -->|Yes| VALIDATE["Set working directory and PYTHONPATH<br/>Validate each script: exists, readable, compiles"]
    VALIDATE --> SKIP["Record invalid scripts as skipped<br/>Build DB, persistent, and market run lists"]
    SKIP --> DB["Run validated DB scripts sequentially<br/>database/test.py then create_all_tables.py"]
    DB --> DBOK{"A DB script exits nonzero?"}
    DBOK -->|Yes| DBFAIL["Exit 1: DB script failed"]
    DBOK -->|No| PERSIST["Launch persistent watchers first<br/>Skip worker if matching process already exists"]
    PERSIST --> NLP["NLP/main_NLP.py<br/>News fetch, FinBERT, sentiment persistence"]
    PERSIST --> PRUNE["retention/prune_market_data.py<br/>SSM-gated deletion of old price rows"]
    PERSIST --> MARKET["Launch market scripts sequentially<br/>3-second startup grace for each by default"]
    MARKET --> BOT["src/main.py<br/>Portfolio threads and optional OMS pump"]
    MARKET --> ING["realTime/realtimeDataIngestor.py<br/>Price ingestion"]
    MARKET --> PNL["realTime/pnl_script.py<br/>Book valuation"]
    MARKET --> RBP["orchestrator/rbp_runner.py<br/>Forecast refresh"]
    BOT --> SUMMARY["Report running, skipped, and failed scripts"]
    ING --> SUMMARY
    PNL --> SUMMARY
    RBP --> SUMMARY
    SUMMARY --> ANY{"Any market PID survived startup?"}
    ANY -->|No| MARKETFAIL["Exit 1: no market worker started"]
    ANY -->|Yes| WATCH["Log memory and enter NASDAQ market watchdog"]
    style START fill:#dbeafe,stroke:#2563eb,stroke-width:2px
```

Missing scripts and syntax failures skip only that script. Import/initialization failures surface in the startup grace check; other workers still launch. DB scripts that return nonzero abort startup, but the current DB helpers can print errors and return zero, so their printed output matters.

**Launch order is not a readiness dependency:** the bot starts before the ingestor. Historical data and initial capital must already be prepared. The first market-hours check occurs **after** workers launch, so starting outside market hours can briefly run the market workers before they are terminated.

## How data reaches the trading books

```mermaid
flowchart TD
    START(["start.sh: workers are running"]) --> INPUTS["Read prepared inputs and fetch new data"]
    INPUTS --> PRICES["Real-time ingestor: fetch FMP quotes"]
    INPUTS --> HISTORY["Read historical backfill already in market_data"]
    INPUTS --> NEWS["NLP: fetch FMP and optional news sources"]
    PRICES --> MD[("market_data")]
    HISTORY --> MD
    NEWS --> NLP["FinBERT scoring"]
    NLP --> NS[("news_sentiment")]
    NS --> SYNC["Sync recent daily sentiment into market_data"]
    SYNC --> MD
    MD --> STRAT["Portfolio reads prices and books<br/>Updates indicators; calls OnData<br/>P7 may also read news_sentiment"]
    MD --> RBP["RBP forecast service"]
    RBP --> FC[("rbp_forecasts")]
    STRAT --> SIGNAL["Buy / sell and confidence"]
    SIGNAL --> SIZE["Shared live executor sizing"]
    FC -.->|"Optional confidence overlay"| SIZE
    SIZE --> ROUTE{"Execution route"}
    ROUTE -->|Direct| FILL["Fetch current FMP quote and settle fill"]
    ROUTE -->|OMS enabled| OMS["Per-portfolio parent and child schedule"]
    OMS --> PUMP["OMS tick: read fresh portfolio state"]
    PUMP --> FILL
    FILL --> BOOKS[("Cash, positions, and execution logs")]
    BOOKS --> PNL["PnL worker reads books and market prices"]
    PNL --> PB[("pnl_book")]
    PB --> NEXT["Next portfolio poll reads updated books"]
    style START fill:#dbeafe,stroke:#2563eb,stroke-width:2px
```

The live executor records fills in PostgreSQL; this path contains no external broker submission. The RBP overlay is constructed only when the manager config contains `rbp_overlay.enabled: true`; it is absent in the checked-in manager config. NLP is consumed by sentiment-aware strategies such as P7, not automatically by every portfolio. P1 and P2 are the current `src/main.py` selection.

## Market watchdog and persistent-worker lifecycle

```mermaid
flowchart TD
    ROOT(["start.sh: supervision begins"]) --> MODE{"Independent supervision branches"}
    MODE -->|Market workers| CHECK["Check FMP NASDAQ exchange-market-hours"]
    CHECK --> STATUS{"Boolean market status?"}
    STATUS -->|Open| RESET["Reset unknown streak<br/>Check market PIDs; report exited workers"]
    RESET --> ANY{"Any market worker alive?"}
    ANY -->|Yes| MEM["Log memory; sleep 180 seconds"]
    MEM --> NEXT["Next market-status check"]
    ANY -->|No| FAIL["Exit watchdog with code 1"]
    STATUS -->|Unknown or API error| COUNT["Increment unknown streak"]
    COUNT --> LIMIT{"Streak reaches limit?<br/>Default 20 checks"}
    LIMIT -->|No| RETRY["Keep workers running; sleep 180 seconds"]
    RETRY --> NEXT
    LIMIT -->|Yes| STOP["SIGTERM surviving market PIDs<br/>Wait; exit watchdog"]
    STATUS -->|Closed| STOP
    MODE -->|Persistent workers| WORKER["Detached watcher starts Python worker"]
    WORKER --> DONE["Worker exits with any code"]
    DONE --> BACKOFF["Log exit; sleep 30 seconds"]
    BACKOFF --> RESTART["Restart this worker<br/>Independent of market-status checks"]
    style ROOT fill:#dbeafe,stroke:#2563eb,stroke-width:2px
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
