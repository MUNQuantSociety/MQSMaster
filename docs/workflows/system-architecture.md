# System architecture

[Setup](../../README.md) · [Workflow index](README.md)

Read this hierarchy from the root downward. The live stack starts at `start.sh`; the other project entrypoints form a separate branch because the launcher does not run them. Detailed workflow pages show the data exchanged between these components.

```mermaid
flowchart TD
    ROOT(["MQS Trading System: choose an entrypoint"]) --> LIVE["start.sh: supervised live stack"]
    ROOT --> MANUAL["Separately invoked projects"]
    LIVE --> PRE["Environment, script validation, DB preflight"]
    PRE --> PERSIST["Launch persistent workers"]
    PERSIST --> NLP["NLP/main_NLP.py<br/>News providers to FinBERT to sentiment"]
    PERSIST --> RET["Retention pruner<br/>SSM-gated deletion of old price rows"]
    PERSIST --> MARKET["Launch market workers"]
    MARKET --> BOT["src/main.py<br/>Strategies, sizing, optional OMS"]
    MARKET --> ING["Real-time ingestor<br/>FMP quotes to market_data"]
    MARKET --> PNL["PnL worker<br/>Books and prices to pnl_book"]
    MARKET --> RF["RBP runner<br/>Market history to rbp_forecasts"]
    BOT --> BOOKS["Cash, positions, and execution logs"]
    NLP --> NS["news_sentiment and recent market sentiment"]
    MARKET --> WATCH["Market watchdog<br/>Stop market workers at close"]
    MANUAL --> BF["Backfill CLI<br/>Historical prices to market_data"]
    MANUAL --> CAPITAL["Funding and daily allocator<br/>Book balances and internal transfers"]
    MANUAL --> BT["Backtest engine<br/>Selected portfolios and historical data"]
    BT --> REPORT["Reports, CSVs, and local cache"]
    REPORT --> ANALYSIS["Backtest analysis tools"]
    MANUAL --> RR["RBP research CLI"]
    RR --> CSV["Prediction and RBI CSVs"]
    MANUAL --> CFA["CFA calculator"]
    style ROOT fill:#dbeafe,stroke:#2563eb,stroke-width:2px
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
