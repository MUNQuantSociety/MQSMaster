# NLP Pipeline — Workflow

Multi-source text ingestion → sentiment scoring → database → portfolio consumption.

```mermaid
graph TD
    subgraph Sources
        FMP[FMP API<br/>news · press · transcripts]
        OTHER[Other Sources<br/>Reddit · X · RSS · SEC filings]
    end

    subgraph Ingestion
        FMP -->|"pull raw text"| FETCH[Fetch &amp; Clean]
        OTHER -->|"pull raw text"| FETCH
        FETCH -->|"dedupe · tag ticker · language"| SCORE[Compute Score]
    end

    subgraph Scoring
        SCORE -->|"LLM / FinBERT inference"| SENT[Per-Ticker Sentiment]
        SENT -->|"aggregate window"| UPD[Update Database]
    end

    subgraph Persistence
        UPD -->|"INSERT scores + meta"| DB[(PostgreSQL<br/>nlp_scores · fetch_state)]
    end

    subgraph Consumption
        DB -->|"read latest scores"| PORT[Portfolio]
        PORT -->|"weight + rebalance"| STRAT[Strategy Inputs]
    end

    DB -.->|"scheduled re-fetch"| FETCH

    classDef src fill:#22d3ee,stroke:#0e7490,color:#0f172a;
    classDef src2 fill:#60a5fa,stroke:#1d4ed8,color:#ffffff;
    classDef ingest fill:#f472b6,stroke:#be185d,color:#ffffff;
    classDef score fill:#a78bfa,stroke:#6d28d9,color:#ffffff;
    classDef db fill:#fbbf24,stroke:#b45309,color:#0f172a;
    classDef port fill:#34d399,stroke:#047857,color:#0f172a;

    class FMP src;
    class OTHER src2;
    class FETCH ingest;
    class SCORE,SENT score;
    class UPD,DB db;
    class PORT,STRAT port;
```

## Stages

1. **Sources** — FMP API for news/press/transcripts, plus Reddit, X, RSS, SEC filings.
2. **Fetch & Clean** — dedupe, normalize, tag tickers and language.
3. **Compute Score** — LLM / FinBERT inference produces per-ticker sentiment.
4. **Update Database** — write scores and state snapshots to PostgreSQL.
5. **Portfolio** — reads latest scores, applies weighting and rebalance rules.
6. **Strategy Inputs** — sentiment feeds downstream strategies.

Scheduled re-fetch closes the loop.
