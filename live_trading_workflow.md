# Live Trading Bot — Workflow

End-to-end loop from market data ingestion to order execution.

```mermaid
graph TD
    subgraph Market Data Ingestion
        FMP[FMP API<br/>quotes · OHLCV · fundamentals] -->|"fetch latest bars"| UPD[Update Database]
        UPD -->|"INSERT / UPSERT"| DB[(PostgreSQL<br/>market_data)]
    end

    subgraph Strategy Layer
        DB -->|"fetch features + history"| FETCH[Fetch Database]
        FETCH -->|"feed inputs"| STRAT[Strategy.OnData]
        STRAT -->|"signal + sizing"| SIG[Compute Signal]
    end

    subgraph Execution Layer
        SIG -->|"context.buy / context.sell"| EXEC[Execute Order]
        EXEC -->|"broker submit"| BROKER[Broker API]
        EXEC -->|"trade log + fills"| DB
    end

    BROKER -.->|"next schedule tick"| FMP

    classDef api fill:#22d3ee,stroke:#0e7490,color:#0f172a;
    classDef db fill:#fbbf24,stroke:#b45309,color:#0f172a;
    classDef strat fill:#a78bfa,stroke:#6d28d9,color:#0f172a;
    classDef exec fill:#f87171,stroke:#b91c1c,color:#ffffff;
    classDef signal fill:#34d399,stroke:#047857,color:#0f172a;

    class FMP,BROKER api;
    class DB db;
    class STRAT,FETCH strat;
    class SIG signal;
    class UPD,EXEC exec;
```

## Stages

1. **FMP API** — pull quotes / OHLCV / fundamentals on schedule tick.
2. **Update Database** — ingest, normalize, write to PostgreSQL.
3. **Strategy** — scheduled run loads parameters and pulls features.
4. **Fetch Database** — read latest bars + history needed by the model.
5. **Compute Signal** — model + rules produce target position and sizing.
6. **Execute Order** — submit via broker API, log fills back to DB.

Loop repeats on next schedule tick.
