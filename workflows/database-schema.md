# Database Schema & Data Flow

Entity relationships and data flow through the PostgreSQL database.

```mermaid
erDiagram
    MARKET_DATA {
        bigint id PK
        varchar ticker
        timestamp timestamp
        decimal open_price
        decimal high_price
        decimal low_price
        decimal close_price
        bigint volume
    }

    CASH_EQUITY_BOOK {
        bigint id PK
        timestamp timestamp
        date date
        varchar portfolio_id FK
        varchar currency
        decimal notional
    }

    POSITIONS_BOOK {
        bigint position_id PK
        varchar portfolio_id FK
        varchar ticker
        decimal quantity
        timestamp updated_at
    }

    TRADE_EXECUTION_LOGS {
        bigint id PK
        varchar portfolio_id FK
        varchar ticker
        timestamp exec_timestamp
        varchar side
        decimal quantity
        decimal arrival_price
        decimal exec_price
        decimal slippage_bps
        decimal notional
        decimal notional_local
        varchar currency
        decimal fx_rate
    }

    PNL_BOOK {
        bigint id PK
        varchar portfolio_id FK
        timestamp timestamp
        decimal notional
    }

    CASH_EQUITY_BOOK ||--o{ POSITIONS_BOOK : "portfolio_id"
    CASH_EQUITY_BOOK ||--o{ TRADE_EXECUTION_LOGS : "portfolio_id"
    CASH_EQUITY_BOOK ||--o{ PNL_BOOK : "portfolio_id"
    MARKET_DATA ||--o{ TRADE_EXECUTION_LOGS : "ticker"
    MARKET_DATA ||--o{ POSITIONS_BOOK : "ticker"
```

## Data Flow Diagram

```mermaid
flowchart TD
    subgraph Ingestion["Data Ingestion"]
        FMP["FMP API"]
        BACKFILL["Backfill Service"]
        REALTIME["Realtime Ingestor"]
    end

    subgraph Storage["PostgreSQL"]
        MD[(market_data)]
        CASH[(cash_equity_book)]
        POS[(positions_book)]
        TRADES[(trade_execution_logs)]
        PNL[(pnl_book)]
    end

    subgraph Read["Read Operations"]
        STRATEGY["Strategy<br/>get_data()"]
        ALLOCATOR["DailyAllocator"]
        REPORTING["Backtest Reporting"]
    end

    subgraph Write["Write Operations"]
        LIVE_EXEC["Live Executor"]
        CAPITAL["Capital Manager"]
        ALLOC_WRITE["Allocator Transfers"]
    end

    %% Ingestion to storage
    FMP --> BACKFILL
    FMP --> REALTIME
    BACKFILL --> MD
    REALTIME --> MD

    %% Read operations
    MD --> STRATEGY
    CASH --> STRATEGY
    POS --> STRATEGY
    PNL --> STRATEGY
    
    CASH --> ALLOCATOR
    POS --> ALLOCATOR
    
    MD --> REPORTING
    TRADES --> REPORTING

    %% Write operations
    LIVE_EXEC --> CASH
    LIVE_EXEC --> POS
    LIVE_EXEC --> TRADES
    
    CAPITAL --> CASH
    CAPITAL --> TRADES
    
    ALLOC_WRITE --> CASH
    ALLOC_WRITE --> TRADES

    %% Styling
    classDef ingest fill:#e3f2fd,stroke:#1565c0
    classDef storage fill:#fff3e0,stroke:#ef6c00
    classDef read fill:#e8f5e9,stroke:#2e7d32
    classDef write fill:#fce4ec,stroke:#c2185b

    class FMP,BACKFILL,REALTIME ingest
    class MD,CASH,POS,TRADES,PNL storage
    class STRATEGY,ALLOCATOR,REPORTING read
    class LIVE_EXEC,CAPITAL,ALLOC_WRITE write
```

## Atomic State Query

The system uses a single atomic query to fetch consistent portfolio state:

```mermaid
flowchart LR
    subgraph Query["Atomic State Query"]
        CTE1["CTE: latest_cash<br/>ORDER BY timestamp DESC<br/>LIMIT 1"]
        CTE2["CTE: latest_positions<br/>DISTINCT ON (ticker)<br/>ORDER BY updated_at DESC"]
        RESULT["SELECT<br/>cash_data,<br/>positions_data"]
    end

    CASH[(cash_equity_book)] --> CTE1
    POS[(positions_book)] --> CTE2
    CTE1 --> RESULT
    CTE2 --> RESULT
```

## Connection Pool Architecture

```mermaid
flowchart TD
    subgraph Application["Application Threads"]
        T1["Thread 1<br/>(Portfolio 1)"]
        T2["Thread 2<br/>(Portfolio 2)"]
        T3["Thread 3<br/>(Portfolio N)"]
    end

    subgraph Pool["ThreadedConnectionPool"]
        POOL["Connection Pool<br/>minconn=1, maxconn=6"]
        C1["Connection 1"]
        C2["Connection 2"]
        C3["Connection 3"]
        CN["Connection N"]
    end

    subgraph DB["PostgreSQL"]
        PG[(Database)]
    end

    T1 -->|"get_connection()"| POOL
    T2 -->|"get_connection()"| POOL
    T3 -->|"get_connection()"| POOL
    
    POOL --> C1
    POOL --> C2
    POOL --> C3
    POOL --> CN
    
    C1 --> PG
    C2 --> PG
    C3 --> PG
    CN --> PG

    T1 -.->|"release_connection()"| POOL
    T2 -.->|"release_connection()"| POOL
    T3 -.->|"release_connection()"| POOL
```

## Key Queries

### Fetch Latest Cash
```sql
SELECT notional FROM cash_equity_book 
WHERE portfolio_id = %s 
ORDER BY timestamp DESC LIMIT 1;
```

### Fetch Latest Positions
```sql
SELECT DISTINCT ON (ticker) 
    position_id, portfolio_id, ticker, quantity, updated_at
FROM positions_book
WHERE portfolio_id = %s
ORDER BY ticker, updated_at DESC;
```

### Fetch Market Data
```sql
SELECT * FROM market_data
WHERE ticker IN ({placeholders})
  AND timestamp BETWEEN %s AND %s;
```

### Bulk Insert with Conflict Handling
```sql
INSERT INTO {table} ({columns}) VALUES %s
ON CONFLICT ({conflict_columns}) DO NOTHING;
```
