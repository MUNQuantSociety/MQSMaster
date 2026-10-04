# Data Pipeline

Three flows feed the `market_data` table:

1. **Backfill** — bulk historical ingest from FMP, driven by the `backfill_cli`.
2. **Real-time ingestor** — minute-resolution scrape during market hours.
3. **Strategy reads** — backtests warm a per-ticker parquet cache to skip repeated DB hits.

Plus an out-of-band ticker-universe refresh (`refresh.py`) that updates the seed list and optionally kicks off a backfill.

## High-level Flow

```mermaid
flowchart TD
    subgraph Source["Source"]
        FMP["Financial Modeling Prep API"]
    end

    subgraph Client["FMPMarketData (src/orchestrator/marketData/)"]
        RL["Rate limiter<br/>(3000 req/min, sliding window)"]
        RT["Retry / timeout / internet check"]
        EP["Endpoints:<br/>get_historical_data,<br/>get_intraday_data,<br/>get_realtime_data,<br/>get_current_price"]
    end

    subgraph Ingest["Ingest paths"]
        BF["Backfill CLI:<br/>specific / concurrent / inject-csv"]
        REFRESH["refresh.py<br/>(updates ticker universe<br/>+ optional concurrent backfill)"]
        RTI["realtimeDataIngestor<br/>(per-minute live cycle<br/>during market hours)"]
    end

    subgraph Storage["Storage"]
        DB[(PostgreSQL<br/>market_data)]
        PARQUET["src/backtest/data/backfill_cache/<br/>{ticker}.parquet"]
    end

    subgraph Consumers["Consumers"]
        STRAT["Strategy.get_data() →<br/>BasePortfolio queries"]
        EXEC["tradeExecutor.get_current_price()"]
        BTUTILS["fetch_historical_data()<br/>(reads parquet first,<br/>fills gaps from DB)"]
    end

    FMP --> RL --> RT --> EP

    EP --> BF
    EP --> REFRESH
    EP --> RTI
    EP --> EXEC

    BF --> DB
    REFRESH --> DB
    RTI --> DB

    DB --> STRAT
    DB --> BTUTILS
    BTUTILS <--> PARQUET

    classDef source fill:#e3f2fd,stroke:#1565c0
    classDef client fill:#fff3e0,stroke:#ef6c00
    classDef ingest fill:#fce4ec,stroke:#c2185b
    classDef storage fill:#f3e5f5,stroke:#7b1fa2
    classDef consumer fill:#e8f5e9,stroke:#2e7d32

    class FMP source
    class RL,RT,EP client
    class BF,REFRESH,RTI ingest
    class DB,PARQUET storage
    class STRAT,EXEC,BTUTILS consumer
```

## FMPMarketData Client

A single `FMPMarketData` instance is shared across threads in a process; rate limiting and retries are thread-safe.

```mermaid
sequenceDiagram
    participant T1 as Thread / caller
    participant Limiter as RateLimiter (lock)
    participant Net as Internet check
    participant FMP

    T1->>Limiter: _check_rate_limit()
    Limiter->>Limiter: Acquire lock
    Limiter->>Limiter: Drop timestamps > 60s old
    alt Count >= 3000
        Limiter->>Limiter: Sleep until oldest expires
    end
    Limiter->>Limiter: Record this request
    Limiter->>Limiter: Release lock
    T1->>Net: Probe connectivity (cached briefly)
    Net-->>T1: ok
    T1->>FMP: HTTP request (timeout 10s, up to 6 retries)
    FMP-->>T1: response
```

Key methods: `get_historical_data`, `get_intraday_data`, `get_realtime_data` (batch exchange quote), `get_current_price`.

## Backfill CLI

Entrypoint: `python -m src.orchestrator.backfill.backfill_cli <command>`.

```mermaid
flowchart TD
    CLI["backfill_cli"] --> CMD{command}

    CMD -->|"specific"| SPEC["specific: sequential<br/>per-ticker fetch over a<br/>continuous date range"]
    CMD -->|"concurrent"| CONC["concurrent: parallel<br/>workers (--threads)"]
    CMD -->|"inject-csv"| INJ["inject-csv: load CSV<br/>dumps from --csv-dir"]

    SPEC --> FETCH["FMP intraday/historical fetch<br/>(--interval 1|5|15|30|60)"]
    CONC --> FETCH
    INJ --> PARSE["Parse CSV files"]

    FETCH --> PREP["prepare_data()<br/>schema alignment"]
    PARSE --> PREP

    PREP --> WRITE["bulk_inject_to_db<br/>(--on-conflict fail|ignore)"]
    WRITE --> DB[(market_data)]

    classDef cli fill:#e3f2fd,stroke:#1565c0
    classDef path fill:#fce4ec,stroke:#c2185b
    classDef db fill:#f3e5f5,stroke:#7b1fa2
    class CLI,CMD cli
    class SPEC,CONC,INJ,FETCH,PARSE,PREP,WRITE path
    class DB db
```

| Argument | Default | Notes |
|----------|---------|-------|
| `--start DDMMYY` / `--end DDMMYY` | — | Inclusive range |
| `--tickers T1 T2 ...` | falls back to `tickers.json` | Sibling of `backfill_cli.py` |
| `--exchange` | `NASDAQ` | |
| `--interval` | `1` | Minutes; one of `1, 5, 15, 30, 60` |
| `--threads` (concurrent / inject-csv) | `6` / `5` | Keep under DB pool size |
| `--on-conflict` | `fail` | `ignore` appends `ON CONFLICT DO NOTHING` |
| `--dry-run` | off | Fetch + parse, skip DB writes |

Full command reference: [../BackFill/Readme.md](../BackFill/Readme.md).

## Ticker Refresh + Universe Update

`src/orchestrator/backfill/update/refresh.py` keeps the seed ticker list in sync with current S&P 500 / commodity / crypto coverage:

```mermaid
flowchart LR
    R["python refresh.py"] --> LOAD["Load extra_tickers/nasdaq_tickers.json"]
    LOAD --> FETCH["Fetch latest S&P 500,<br/>commodity, crypto tickers"]
    FETCH --> MERGE["Merge + dedupe"]
    MERGE --> WRITE["Write extra_tickers/<br/>nasdaq_tickers.json"]
    WRITE --> DECIDE{--skip-backfill?}
    DECIDE -->|"yes"| END["Exit"]
    DECIDE -->|"no"| BF["concurrent_backfill<br/>(default --threads 8)"]
    BF --> DB[(market_data)]

    classDef refresh fill:#fff8e1,stroke:#ff6f00
    class R,LOAD,FETCH,MERGE,WRITE,DECIDE,BF,END refresh
```

CLI reference: [../BackFill/refresh_README.md](../BackFill/refresh_README.md).

## Real-time Ingestor

See the [separate detailed ingestion workflow](realtime-ingestor-workflow.md) and [local worker setup](../../src/orchestrator/realTime/README.md).

```mermaid
flowchart TD
    START["realtimeDataIngestor.main()"] --> VERIFY["verify_upsert_constraint()<br/>(UNIQUE index on ticker, timestamp<br/>absence exits; inspection errors warn)"]
    VERIFY --> LOAD["load_tickers()<br/>from backfill/tickers.json"]
    LOAD --> INIT["initialize_volume_state()<br/>(read latest 'volume' per ticker for today)"]
    INIT --> LOOP["Every 60s (start.sh enforces market hours)"]
    LOOP --> FEEDS["For each feed:<br/>NASDAQ, NYSE, AMEX (batch-exchange-quote)<br/>CRYPTO (batch-crypto-quotes)<br/>COMMODITY (batch-commodity-quotes)"]
    FEEDS --> PROCESS["process_market_data:<br/>filter tracked tickers not yet seen this cycle,<br/>exchange = row's FMP value, else feed label,<br/>compute interval volume<br/>(diff vs last_known_volume)"]
    PROCESS --> INSERT["bulk_inject_to_db(<br/>conflict_columns=[ticker, timestamp])"]
    INSERT --> REPORT["Log prepared / inserted / ignored,<br/>failed feeds, uncovered tickers"]
    REPORT --> LOOP

    classDef rti fill:#e8f5e9,stroke:#2e7d32
    class START,VERIFY,LOAD,INIT,LOOP,FEEDS,PROCESS,INSERT,REPORT rti
```

Each cycle polls up to five FMP feeds (one request each before retries), stopping early if all tracked symbols have been matched. `exchange` on a row is FMP's own per-row value (`NASDAQ`, `NYSE`, `AMEX`, `CRYPTO`, `COMMODITY`), falling back to the feed label only when FMP returns null — the column is `NOT NULL` and a single null would roll back the whole batch. A ticker matched by one feed is excluded from later feeds in the same cycle so its volume state is never advanced twice.

`open_price` / `high_price` / `low_price` are deliberately left `NULL` on realtime rows: FMP's quote endpoints return session-level `open` / `dayHigh` / `dayLow`, not per-minute values, so only backfilled bars carry real OHL.

The ingestor stores *interval* volume (delta vs previously seen cumulative volume), not the cumulative API value. There is a known fragility: a crash mid-day re-initializes volume state from the stored interval volume, which is incorrect — see the warning comment at the top of `realtimeDataIngestor.py`. Crypto volumes from FMP are rolling-window rather than session-cumulative, so negative deltas (logged per cycle) are more common there.

Logs go to stderr and, when writable, `/var/log/market_data_ingestor.log` (override with `MARKET_DATA_INGESTOR_LOG`). Tracked tickers that no feed serves are listed by name on the first cycle and every 30th cycle after.

## Backfill Cache (parquet)

The backtest path warms a per-ticker parquet file under `src/backtest/data/backfill_cache/{ticker}.parquet`. This is purely a *read* cache: it speeds up repeat backtests over overlapping date ranges by avoiding the DB round trip for the bars already on disk.

```mermaid
flowchart LR
    REQ["fetch_historical_data(portfolio,<br/>start, end)"] --> CACHE{cache hit?}
    CACHE -->|"covers full range"| RET["Return cached frame"]
    CACHE -->|"partial / miss"| FETCH["Query market_data for missing range"]
    FETCH --> MERGE["Append + dedupe"]
    MERGE --> SAVE["Write parquet"]
    SAVE --> RET

    classDef cache fill:#f3e5f5,stroke:#7b1fa2
    class REQ,CACHE,RET,FETCH,MERGE,SAVE cache
```

The cache directory is committed (the `.parquet` files act as a checked-in dataset for tests). Delete a ticker's parquet to force a fresh DB pull for that symbol.

## `market_data` Schema (subset)

| Column | Type | Notes |
|--------|------|-------|
| `id` | `SERIAL PK` | |
| `ticker` | `VARCHAR(10)` | |
| `timestamp` | `TIMESTAMP WITH TIME ZONE` | UTC stored, NY for query convenience |
| `date` | `DATE` | Denormalized for daily-window filters |
| `exchange` | `VARCHAR(50)` | |
| `open_price` / `high_price` / `low_price` / `close_price` | `NUMERIC` | |
| `volume` | `BIGINT` | Real-time ingestor stores *interval* volume |
| `avg_sentiment` | `NUMERIC` | Base-schema column; NLP writes a separate `sentiment_score` column (see [NLP setup](../../NLP/WORKFLOW.md#database-contract)) |
| `created_at` | `TIMESTAMP DEFAULT NOW()` | |

Unique constraint: `(ticker, timestamp)` — required for `--on-conflict ignore` semantics.
