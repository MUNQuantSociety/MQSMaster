# MQS Trading System — Documentation

This is the entry point for all internal documentation. The repo is split into a live-trading and backtesting core (`src/`), supporting research/analysis modules (`NLP/`, `RBP/`, `CFA/`), and operational tooling (`scripts/`, `.github/workflows`).

## Map at a Glance

```mermaid
flowchart LR
    subgraph Core["Core trading platform — src/"]
        BT[Backtest engine]
        LIVE[Live trading engine]
        STRAT[Strategies P1..P5]
        DATA[Backfill / market data]
        RISK[Risk + capital allocation]
    end

    subgraph Research["Research / signals"]
        NLP[NLP sentiment pipeline]
        RBP[RBP model<br/>(loose research scripts)]
    end

    subgraph Side["Side library"]
        CFA[CFA finance calculator]
    end

    subgraph Ops["Operations"]
        CICD[CI / CD pipeline]
        TESTS[Test suite + markers]
        SCRIPTS[Backtest tooling]
    end

    NLP -->|"news_sentiment"| DATA
    RBP -->|"feeds Portfolio 5"| STRAT
    DATA --> BT
    DATA --> LIVE
    STRAT --> BT
    STRAT --> LIVE
    RISK --> LIVE
    SCRIPTS --> BT
```

## Documentation Index

### Architecture & Workflows (`workflows/`)
- [System architecture](workflows/system-architecture.md) — components and their relationships
- [Backtest flow](workflows/backtest-flow.md) — multiprocess driver, event-mode, fast-mode (vectorized + Monte Carlo)
- [Live trading flow](workflows/live-trading-flow.md) — concurrent portfolio threads, circuit breakers
- [Portfolio / strategy flow](workflows/portfolio-strategy-flow.md) — `BasePortfolio`, indicators, `StrategyContext`
- [Data pipeline](workflows/data-pipeline.md) — FMP ingestion, backfill CLI, parquet cache, real-time ingestor
- [Capital management](workflows/capital-management.md) — master portfolio, daily rebalancing
- [Database schema](workflows/database-schema.md) — tables, ER, atomic state queries

### Subsystems
- [NLP sentiment pipeline](NLP/README.md) — daemon, FinBERT scoring, `news_sentiment` table. **Requires the fine-tuned FinBERT model** to be downloaded into `NLP/finbert-combined-final/` before use — see [NLP setup](NLP/README.md#prerequisite-download-the-finbert-model).
- [Backfill CLI](BackFill/Readme.md) — `backfill_cli` commands and arguments
- [Backfill / refresh script](BackFill/refresh_README.md) — `refresh.py` ticker universe updater
- [Order Management System (proposed)](OMS/OMS_DESIGN.md) — VWAP/TWAP design doc, not yet implemented

### Operations
- [CI / CD](CICD/CICD.md) — pipeline stages, secrets, coverage gate
- [Test modes](TEST_MODES.md) — pytest markers, smoke vs slow vs db tiers

## Key Entry Points

| Task | Command |
|------|---------|
| Run backtest (multi-portfolio, parallel) | `python -m src.main_backtest` |
| Run live trading | `python -m src.main` |
| Backfill historical bars | `python -m src.orchestrator.backfill.backfill_cli concurrent --start ... --end ... --tickers ...` |
| Refresh ticker universe + backfill | `python src/orchestrator/backfill/update/refresh.py` |
| Real-time price ingestor | `python -m src.orchestrator.realTime.realtimeDataIngestor` |
| Daily capital rebalance | `python -m src.risk_manager.daily_allocator` |
| Add / withdraw master capital | `python -m src.risk_manager.manage_capital --action ADD --amount ...` |
| Start NLP sentiment daemon | `python NLP/daemon.py start` |
| Run smoke tests | `pytest -m smoke` |

## Conventions

- Diagrams use Mermaid.js. They render natively in GitHub, in VS Code with the Mermaid extension, and at [mermaid.live](https://mermaid.live).
- All timestamps stored in `America/New_York`; SQL queries normalize through that zone.
- Strategies live in `src/portfolios/portfolio_<n>/` with a paired `config.json`. Configs are loaded dynamically by file location, not by import.
- Indicators are loaded by name via `importlib`; new indicators just need a new file under `src/portfolios/indicators/` whose class name matches the snake_case filename.
