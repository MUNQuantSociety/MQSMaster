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
    ROOT(["Application operation"]) --> MODE{"Workload"}
    MODE --> DATA["Backfill or real-time ingestion"]
    DATA --> MD[("market_data")]
    MODE --> LIVE["Live executor"]
    LIVE --> BOOKS[("Cash, positions, execution logs")]
    MODE --> PNLWORK["PnL worker"]
    PNLWORK --> PNL[("pnl_book")]
    MODE --> CAPITAL["Funding or daily allocation"]
    CAPITAL --> BOOKS
    MODE --> NLP["NLP pipeline"]
    NLP --> NS[("news_sentiment")]
    NS --> SYNC["Recent sentiment sync<br/>Extra sentiment_score column required"]
    SYNC --> MD
    MODE --> RBP["RBP forecast runner"]
    RBP --> RF[("rbp_forecasts")]
    MODE --> BT["Backtest executor"]
    BT --> SIM["Simulated books and local reports"]
    MD --> STRAT["Subsequent strategy data reads"]
    BOOKS --> STRAT
    PNL --> STRAT
    NS --> SENT["Sentiment-aware portfolio reads"]
    RF --> OVERLAY["Optional RBP sizing overlay"]
    BOOKS --> ALLOC["Subsequent capital-allocation reads"]
    style ROOT fill:#dbeafe,stroke:#2563eb,stroke-width:2px
```

Portfolio 7 reads `news_sentiment` when selected. The NLP writer also updates `market_data.sentiment_score`, while the base schema above defines `avg_sentiment`; see the [NLP database contract](../../NLP/WORKFLOW.md#database-contract) and [setup prerequisite](../../README.md#3-configure-credentials-and-initialize-the-database). The RBP service writes `rbp_forecasts`, consumed by the optional executor confidence overlay; see the [RBP workflow](../../RBP/README.md).

## Atomic State Query

`BasePortfolio.get_data()` issues a single CTE-based query so that cash and positions are read in one consistent snapshot, instead of two queries that could straddle a write:

```mermaid
flowchart TD
    ROOT(["BasePortfolio.get_data: request portfolio state"]) --> QUERY["Execute ATOMIC_STATE_QUERY"]
    QUERY --> CASH["Read latest cash snapshot<br/>cash_equity_book"]
    QUERY --> POS["Read latest positions<br/>positions_book"]
    CASH --> SNAPSHOT["Return cash and positions<br/>from one SQL statement"]
    POS --> SNAPSHOT
    style ROOT fill:#dbeafe,stroke:#2563eb,stroke-width:2px
```

The single statement returns both halves of the snapshot, eliminating the read-skew window.

## Connection Pool

`MQSDBConnector` wraps `psycopg2.pool.ThreadedConnectionPool`. Threads (live trading) and processes (backtest) acquire and release connections through the same API:

```mermaid
flowchart TD
    ROOT(["Application needs a database connection"]) --> OWNER["Use this process's MQSDBConnector"]
    OWNER --> THREAD["Calling portfolio or worker thread"]
    THREAD --> GET["get_connection"]
    GET --> POOL["ThreadedConnectionPool<br/>minconn=1, maxconn=6"]
    POOL --> CONN["Acquire and health-check a connection"]
    CONN --> DB["Execute PostgreSQL operation"]
    DB --> RESULT["Commit or roll back as required"]
    RESULT --> RELEASE["release_connection"]
    RELEASE --> READY["Connection available for a later caller"]
    style ROOT fill:#dbeafe,stroke:#2563eb,stroke-width:2px
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
