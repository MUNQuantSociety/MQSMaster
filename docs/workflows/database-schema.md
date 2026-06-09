# Database Schema

PostgreSQL schema as defined in `src/common/database/schemaDefinitions.py`. Tables are created idempotently by `python -m src.common.database.create_all_tables`.

## Tables

```mermaid
erDiagram
    USER_CREDS {
        serial user_id PK
        varchar username UK
        varchar password
    }

    MARKET_DATA {
        serial id PK
        varchar ticker
        timestamptz timestamp
        date date
        varchar exchange
        numeric open_price
        numeric high_price
        numeric low_price
        numeric close_price
        bigint volume
        numeric avg_sentiment
        timestamp created_at
    }

    CASH_EQUITY_BOOK {
        serial id PK
        timestamptz timestamp
        date date
        varchar portfolio_id
        varchar currency
        numeric notional
        timestamp created_at
    }

    POSITIONS_BOOK {
        serial position_id PK
        varchar portfolio_id
        varchar ticker
        numeric quantity
        timestamp updated_at
    }

    TRADE_EXECUTION_LOGS {
        serial trade_id PK
        varchar portfolio_id
        varchar ticker
        timestamptz exec_timestamp
        varchar side
        numeric quantity
        numeric arrival_price
        numeric exec_price
        numeric slippage_bps
        numeric notional
        numeric notional_local
        varchar currency
        numeric fx_rate
        timestamp created_at
    }

    PNL_BOOK {
        serial pnl_id PK
        varchar portfolio_id
        timestamptz timestamp
        date date
        numeric realized_pnl
        numeric unrealized_pnl
        numeric fx_rate
        varchar currency
        numeric notional
        timestamp created_at
    }

    RISK_BOOK {
        serial risk_id PK
        varchar portfolio_id
        date date
        timestamp timestamp
        varchar risk_metric
        numeric value
        timestamp created_at
    }

    PORTFOLIO_WEIGHTS {
        serial weights_id PK
        varchar portfolio_id
        varchar ticker
        numeric weight
        varchar model
        date date
        timestamp updated_at
    }

    NEWS_SENTIMENT {
        serial id PK
        varchar ticker
        text article_url
        timestamp published_at
        float sentiment_score
        text content_summary
        timestamp created_at
    }

    CASH_EQUITY_BOOK ||--o{ POSITIONS_BOOK         : "portfolio_id"
    CASH_EQUITY_BOOK ||--o{ TRADE_EXECUTION_LOGS   : "portfolio_id"
    CASH_EQUITY_BOOK ||--o{ PNL_BOOK               : "portfolio_id"
    CASH_EQUITY_BOOK ||--o{ RISK_BOOK              : "portfolio_id"
    CASH_EQUITY_BOOK ||--o{ PORTFOLIO_WEIGHTS      : "portfolio_id"
    MARKET_DATA      ||--o{ TRADE_EXECUTION_LOGS   : "ticker"
    MARKET_DATA      ||--o{ POSITIONS_BOOK         : "ticker"
    MARKET_DATA      ||--o{ NEWS_SENTIMENT         : "ticker"
    POSITIONS_BOOK   ||--o{ PORTFOLIO_WEIGHTS      : "ticker"
```

Constraints worth highlighting:
- `positions_book` has `UNIQUE (portfolio_id, ticker)` — at most one row per portfolio × ticker.
- `portfolio_weights` has `UNIQUE (portfolio_id, ticker, date, model)` — one weight per portfolio × ticker × day × model.
- `market_data` should have a unique index on `(ticker, timestamp)` so the backfill CLI's `--on-conflict ignore` mode works as intended.

## Read / Write Paths

```mermaid
flowchart TD
    subgraph Producers["Producers"]
        BF["Backfill CLI"]
        RTI["realtimeDataIngestor"]
        LIVE["tradeExecutor"]
        BTEXEC["BacktestExecutor"]
        ALLOC["DailyAllocator"]
        CAP["manage_capital"]
        NLP["NLP pipeline"]
    end

    subgraph Tables["Tables"]
        MD[(market_data)]
        CASH[(cash_equity_book)]
        POS[(positions_book)]
        TLOG[(trade_execution_logs)]
        PNL[(pnl_book)]
        WTS[(portfolio_weights)]
        RISK[(risk_book)]
        NS[(news_sentiment)]
    end

    subgraph Consumers["Consumers"]
        STRAT["Strategy.get_data()"]
        REPORT["Backtest reporting"]
        ALLOCR["DailyAllocator reads"]
    end

    BF --> MD
    RTI --> MD
    NLP --> NS
    NLP -.->|"avg_sentiment overlay"| MD

    LIVE --> CASH
    LIVE --> POS
    LIVE --> TLOG
    BTEXEC -. simulated .-> TLOG

    ALLOC --> CASH
    ALLOC --> TLOG
    CAP --> CASH
    CAP --> TLOG

    MD --> STRAT
    CASH --> STRAT
    POS --> STRAT
    PNL --> STRAT

    MD --> REPORT
    TLOG --> REPORT

    CASH --> ALLOCR
    POS --> ALLOCR
```

`portfolio_weights`, `risk_book`, and `news_sentiment` are produced by research/operational paths but are not yet read by the live trading or backtest engine — they exist for reporting and future strategy use.

## Atomic State Query

`BasePortfolio.get_data()` issues a single CTE-based query so that cash and positions are read in one consistent snapshot, instead of two queries that could straddle a write:

```mermaid
flowchart LR
    subgraph Q["ATOMIC_STATE_QUERY"]
        C1["latest_cash CTE:<br/>cash_equity_book<br/>WHERE portfolio_id = %s<br/>ORDER BY timestamp DESC, id DESC<br/>LIMIT 1"]
        C2["latest_positions CTE:<br/>DISTINCT ON (ticker)<br/>ORDER BY ticker, updated_at DESC"]
        SEL["SELECT row_to_json(latest_cash),<br/>       json_agg(latest_positions)"]
    end
    CASH[(cash_equity_book)] --> C1
    POS[(positions_book)]  --> C2
    C1 --> SEL
    C2 --> SEL
```

The single statement returns both halves of the snapshot, eliminating the read-skew window.

## Connection Pool

`MQSDBConnector` wraps `psycopg2.pool.ThreadedConnectionPool`. Threads (live trading) and processes (backtest) acquire and release connections through the same API:

```mermaid
flowchart TD
    subgraph Threads["Application threads / processes"]
        T1["Portfolio 1 thread"]
        T2["Portfolio 2 thread"]
        TN["Portfolio N thread"]
    end

    subgraph Pool["ThreadedConnectionPool<br/>minconn=1, maxconn=6"]
        C1["conn 1"]
        C2["conn 2"]
        CN["conn N"]
    end

    T1 -->|"get_connection()"| Pool
    T2 -->|"get_connection()"| Pool
    TN -->|"get_connection()"| Pool

    Pool --> C1 --> DB[(PostgreSQL)]
    Pool --> C2 --> DB
    Pool --> CN --> DB

    T1 -.->|"release_connection()"| Pool
    T2 -.->|"release_connection()"| Pool
    TN -.->|"release_connection()"| Pool
```

When tuning concurrency (live thread count, backfill `--threads`, multiprocess backtest workers), keep the working set under `maxconn` to avoid acquisition stalls.

## Auto-seeding

When a portfolio runs for the first time and has no `cash_equity_book` row, `BasePortfolio._seed_initial_cash()` inserts a starter row of `DEFAULT_INITIAL_CAPITAL` (currently `1,000,000` USD). Similarly, missing tickers in `positions_book` are seeded at `quantity = 0`. Both paths log a warning so the first-run behavior is auditable.

## Common Queries

```sql
-- Latest cash balance for a portfolio
SELECT notional FROM cash_equity_book
WHERE portfolio_id = %s
ORDER BY timestamp DESC, id DESC
LIMIT 1;

-- Latest position per ticker for a portfolio
SELECT DISTINCT ON (ticker)
    position_id, portfolio_id, ticker, quantity, updated_at
FROM positions_book
WHERE portfolio_id = %s
ORDER BY ticker, updated_at DESC;

-- Market-data window for a ticker basket
SELECT *
FROM market_data
WHERE ticker IN ({placeholders})
  AND timestamp BETWEEN %s AND %s;

-- Idempotent bulk insert
INSERT INTO {table} ({columns}) VALUES %s
ON CONFLICT ({conflict_columns}) DO NOTHING;
```
