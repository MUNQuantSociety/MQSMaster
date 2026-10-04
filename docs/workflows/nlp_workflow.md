# NLP sentiment: high-level workflow

[Setup and commands](../../NLP/README.md) · [Detailed NLP diagrams](../../NLP/WORKFLOW.md) · [Full live stack](live_trading_workflow.md)

```mermaid
flowchart TD
    START(["NLP/main_NLP.py<br/>Directly or through start.sh"]) --> RUN["NLPRunner: load FinBERT once"]
    RUN --> CYCLE["Reload ticker universe and select rotating batch"]
    CYCLE --> FETCH["Fetch FMP news for every ticker<br/>Yahoo, Finviz, Alpha Vantage for selected batch"]
    FETCH --> CSV["Merge and deduplicate per-ticker article CSVs"]
    CSV --> NEW{"New rows detected?"}
    NEW -->|Yes| SCORE["Fine-tuned FinBERT: article scores"]
    SCORE --> SCORES["Save article and daily score CSVs"]
    SCORES --> DB[("news_sentiment")]
    DB --> SYNC["Recent daily aggregate"]
    SYNC --> MD[("market_data.sentiment_score")]
    MD --> CONSUMER["Available to sentiment-aware strategies<br/>P7 also reads news_sentiment directly"]
    CONSUMER --> FINISH["Finish ticker sweep"]
    NEW -->|No| FINISH
    FINISH --> WAIT["Wait remaining part of 300-second interval"]
    WAIT --> NEXT["Begin next cycle"]
    style START fill:#dbeafe,stroke:#2563eb,stroke-width:2px
```

The live loop processes tickers sequentially, rotating alternative-source work through up to four batches. It uses a latest-page FMP fetch for other batches. The historical backfill command uses paginated fetches and blocks startup during weekday cash-session hours. A fetch-only CLI and a local two-model notebook are also available.

The score is positive probability minus negative probability. Persistence uses `news_sentiment`, while the market-data sync updates only the last seven days of matching sentiment. The root setup documents the required `market_data.sentiment_score` column, which differs from the base schema's `avg_sentiment`.

NLP enriches data; it does not place trades. The current live entrypoint runs P1/P2, not P7. See the [detailed workflow](../../NLP/WORKFLOW.md) for exact source selection, model label assumptions, positional CSV alignment, retry behavior, and database requirements.
