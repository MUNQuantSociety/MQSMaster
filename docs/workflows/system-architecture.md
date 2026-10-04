# System architecture

[Setup](../../README.md) · [Workflow index](README.md)

```mermaid
flowchart TB
    START["start.sh: Linux supervisor"] --> BOT["Live bot"]
    START --> ING["Real-time ingestor"]
    START --> PNL["PnL worker"]
    START --> RF["RBP forecast runner"]
    START --> NLP["NLP persistent worker"]
    START --> RET["Retention persistent worker"]

    FMP["FMP market quotes"] --> ING
    ING --> MD[("market_data")]
    HIST["Historical backfill CLI"] --> MD
    NEWS["News providers"] --> NLP
    NLP --> NS[("news_sentiment")]
    NLP -->|"Recent sentiment sync"| MD
    MD --> BOT
    MD --> PNL
    MD --> RF
    RF --> FC[("rbp_forecasts")]
    FC -.->|"Config-gated confidence blend"| BOT
    NS -.->|"P7 if selected"| BOT
    BOT --> BOOKS[("Cash, positions, execution logs")]
    BOOKS --> PNL
    PNL --> PB[("pnl_book")]
    PB --> BOT
    CAPITAL["Separate funding and allocator commands"] --> BOOKS
    RET -->|"Deletes old rows; SSM run gate"| MD

    MD --> BT["Backtest engine<br/>Event or vectorized fast mode"]
    STRAT["Portfolio strategies and configs"] --> BOT
    STRAT --> BT
    BT --> REPORT["Local reports, CSVs, cache"]
    REPORT --> ANALYSIS["Backtest analysis tools"]
    MD --> RR["RBP research CLI"]
    RR --> CSV["Prediction and RBI CSVs"]
    CFA["CFA calculator<br/>Independent interactive utility"]
```

## Ownership and execution boundaries

| Component | Main responsibility |
|---|---|
| `src/live_trading/` | Threaded strategy execution with a shared DB-backed executor |
| `src/backtest/` | Per-portfolio simulation and fast vector adapters |
| `src/portfolios/` | Signals, indicators, strategy-specific target construction |
| `src/oms/` | Config-gated, per-portfolio in-memory parent/child order scheduling |
| `src/orchestrator/` | Price ingestion, historical backfill, RBP refresh, retention |
| `NLP/` | News scraping, FinBERT inference, sentiment persistence, local experiments |
| `RBP/` | Relevance-based research prediction and forecast service |
| `src/risk_manager/` | Funding, capital allocation, drawdown controls, optional RBP sizing overlay |
| `scripts/` | Analysis, API utilities, and CFA calculator |

The live executor settles fills in PostgreSQL. Backtests do not prove the live path ran. P1/P2 are selected by the live entrypoint; other strategies and optional overlays require explicit selection/configuration. The price, NLP, and RBP workers have different universe-selection rules, described in their guides.

See [launcher lifecycle](live_trading_workflow.md) for which workers survive market close and how container lifetime limits persistent processes. The production deployment must be checked separately from this code map.
