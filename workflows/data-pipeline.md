# Data Pipeline Workflow

Market data ingestion, backfill, and storage pipeline.

```mermaid
flowchart TD
    subgraph Sources["Data Sources"]
        FMP_API["Financial Modeling Prep API"]
    end

    subgraph FMPClient["FMPMarketData Client"]
        RATE_LIMIT["Rate Limiter<br/>(3000 req/min)"]
        RETRY["Retry Logic<br/>(6 attempts)"]
        TIMEOUT["Timeout Handler<br/>(10s)"]
        INTERNET["Internet Check"]
    end

    subgraph Endpoints["API Endpoints"]
        HISTORICAL["get_historical_data()<br/>/historical-price-full/"]
        INTRADAY["get_intraday_data()<br/>/historical-chart/"]
        REALTIME["get_realtime_data()<br/>/batch-exchange-quote"]
        QUOTE["get_current_price()<br/>/quote/"]
    end

    subgraph Backfill["Backfill Service"]
        BF_START["backfill_data()"]
        BF_DATES["Generate business days"]
        BF_BATCH["Batch by BATCH_DAYS (3)"]
        BF_LOOP["Loop: tickers × date batches"]
        BF_FETCH["Fetch intraday data"]
        BF_PREP["prepare_data()"]
        BF_CACHE["Write to CSV cache"]
    end

    subgraph Storage["Data Storage"]
        CACHE["backfill_cache/<br/>(Temporary CSV)"]
        DB[(PostgreSQL<br/>market_data table)]
    end

    subgraph DBOps["Database Operations"]
        INJECT["inject_to_db()"]
        BULK["bulk_inject_to_db()"]
        READ["read_db()"]
        QUERY["execute_query()"]
    end

    subgraph LiveData["Live Data Flow"]
        LIVE_FETCH["Executor fetches price"]
        LIVE_QUOTE["get_current_price()"]
        LIVE_EXEC["Execute trade at price"]
    end

    %% FMP Client flow
    FMP_API --> RATE_LIMIT
    RATE_LIMIT --> RETRY
    RETRY --> TIMEOUT
    TIMEOUT --> INTERNET

    %% Endpoints
    INTERNET --> HISTORICAL
    INTERNET --> INTRADAY
    INTERNET --> REALTIME
    INTERNET --> QUOTE

    %% Backfill flow
    BF_START --> BF_DATES
    BF_DATES --> BF_BATCH
    BF_BATCH --> BF_LOOP
    BF_LOOP --> BF_FETCH
    BF_FETCH --> INTRADAY
    INTRADAY --> BF_PREP
    BF_PREP --> BF_CACHE
    BF_CACHE --> CACHE
    CACHE --> BULK
    BULK --> DB

    %% Live data flow
    LIVE_FETCH --> LIVE_QUOTE
    LIVE_QUOTE --> QUOTE
    QUOTE --> LIVE_EXEC

    %% DB reads
    DB --> READ
    DB --> QUERY

    %% Styling
    classDef source fill:#e3f2fd,stroke:#1565c0
    classDef client fill:#fff3e0,stroke:#ef6c00
    classDef endpoint fill:#e8f5e9,stroke:#2e7d32
    classDef backfill fill:#fce4ec,stroke:#c2185b
    classDef storage fill:#f3e5f5,stroke:#7b1fa2
    classDef dbops fill:#e0f7fa,stroke:#00838f
    classDef live fill:#fff8e1,stroke:#f9a825

    class FMP_API source
    class RATE_LIMIT,RETRY,TIMEOUT,INTERNET client
    class HISTORICAL,INTRADAY,REALTIME,QUOTE endpoint
    class BF_START,BF_DATES,BF_BATCH,BF_LOOP,BF_FETCH,BF_PREP,BF_CACHE backfill
    class CACHE,DB storage
    class INJECT,BULK,READ,QUERY dbops
    class LIVE_FETCH,LIVE_QUOTE,LIVE_EXEC live
```

## Rate Limiting Strategy

```mermaid
sequenceDiagram
    participant Thread1
    participant Thread2
    participant RateLimiter
    participant FMP_API

    Thread1->>RateLimiter: _check_rate_limit()
    RateLimiter->>RateLimiter: Acquire lock
    RateLimiter->>RateLimiter: Clean old timestamps (>60s)
    RateLimiter->>RateLimiter: Check count < 3000
    RateLimiter->>RateLimiter: Record timestamp
    RateLimiter->>RateLimiter: Release lock
    Thread1->>FMP_API: Make request

    Thread2->>RateLimiter: _check_rate_limit()
    RateLimiter->>RateLimiter: Acquire lock
    Note over RateLimiter: If limit reached, sleep
    RateLimiter->>RateLimiter: Release lock
    Thread2->>FMP_API: Make request
```

## Data Schema

### market_data Table
| Column | Type | Description |
|--------|------|-------------|
| ticker | VARCHAR | Stock symbol |
| timestamp | TIMESTAMP | Bar timestamp |
| open_price | DECIMAL | Opening price |
| high_price | DECIMAL | High price |
| low_price | DECIMAL | Low price |
| close_price | DECIMAL | Closing price |
| volume | BIGINT | Trading volume |

## Backfill Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| BATCH_DAYS | 3 | Days per API call |
| MAX_RETRIES | 6 | Retry attempts |
| TIMEOUT_SECONDS | 10 | Request timeout |
| MAX_REQUESTS_PER_MIN | 3000 | Rate limit |
