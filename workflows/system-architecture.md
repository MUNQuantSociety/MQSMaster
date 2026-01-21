# System Architecture Overview

High-level architecture of the MQS Trading System showing all major components and their relationships.

```mermaid
flowchart TB
    subgraph Entry["Entry Points"]
        MAIN["main.py<br/>(Live Trading)"]
        BACKTEST["main_backtest.py<br/>(Backtesting)"]
    end

    subgraph Core["Core Engines"]
        RUN_ENGINE["RunEngine<br/>(Live Trading Engine)"]
        BACKTEST_ENGINE["BacktestEngine"]
        BACKTEST_RUNNER["BacktestRunner"]
    end

    subgraph Execution["Trade Execution"]
        LIVE_EXEC["tradeExecutor<br/>(Live)"]
        BT_EXEC["BacktestExecutor<br/>(Simulated)"]
    end

    subgraph Portfolios["Portfolio Strategies"]
        BASE["BasePortfolio<br/>(Abstract Base)"]
        P1["VolMomentum"]
        P2["MomentumStrategy"]
        P3["RegimeAdaptiveStrategy"]
        P4["TrendRotateStrategy"]
    end

    subgraph Data["Data Layer"]
        DB["MQSDBConnector<br/>(PostgreSQL Pool)"]
        FMP["FMPMarketData<br/>(Market Data API)"]
        BACKFILL["Backfill Service"]
    end

    subgraph Risk["Risk Management"]
        ALLOCATOR["DailyAllocator"]
        CAPITAL["manage_capital.py"]
    end

    subgraph API["Strategy API"]
        CONTEXT["StrategyContext"]
        MARKET["MarketData"]
        PORTFOLIO["PortfolioManager"]
        INDICATORS["Indicator System"]
    end

    subgraph Reports["Reporting"]
        REPORTING["BacktestReporting"]
    end

    %% Entry to Engines
    MAIN --> RUN_ENGINE
    BACKTEST --> BACKTEST_ENGINE
    BACKTEST_ENGINE --> BACKTEST_RUNNER

    %% Engines to Execution
    RUN_ENGINE --> LIVE_EXEC
    BACKTEST_RUNNER --> BT_EXEC

    %% Portfolios inherit from Base
    BASE --> P1
    BASE --> P2
    BASE --> P3
    BASE --> P4

    %% Engines load portfolios
    RUN_ENGINE --> BASE
    BACKTEST_RUNNER --> BASE

    %% Data connections
    DB --> RUN_ENGINE
    DB --> BACKTEST_ENGINE
    DB --> LIVE_EXEC
    FMP --> LIVE_EXEC
    FMP --> BACKFILL
    BACKFILL --> DB

    %% Risk Management
    DB --> ALLOCATOR
    DB --> CAPITAL
    FMP --> ALLOCATOR

    %% Strategy API
    BASE --> CONTEXT
    CONTEXT --> MARKET
    CONTEXT --> PORTFOLIO
    BASE --> INDICATORS

    %% Reporting
    BACKTEST_RUNNER --> REPORTING
    BT_EXEC --> REPORTING

    %% Styling
    classDef entry fill:#e1f5fe,stroke:#01579b
    classDef engine fill:#fff3e0,stroke:#e65100
    classDef exec fill:#fce4ec,stroke:#880e4f
    classDef portfolio fill:#e8f5e9,stroke:#1b5e20
    classDef data fill:#f3e5f5,stroke:#4a148c
    classDef risk fill:#fff8e1,stroke:#ff6f00
    classDef api fill:#e0f2f1,stroke:#004d40

    class MAIN,BACKTEST entry
    class RUN_ENGINE,BACKTEST_ENGINE,BACKTEST_RUNNER engine
    class LIVE_EXEC,BT_EXEC exec
    class BASE,P1,P2,P3,P4 portfolio
    class DB,FMP,BACKFILL data
    class ALLOCATOR,CAPITAL risk
    class CONTEXT,MARKET,PORTFOLIO,INDICATORS api
```

## Component Descriptions

| Component | Location | Purpose |
|-----------|----------|---------|
| RunEngine | `src/live_trading/engine.py` | Manages concurrent portfolio execution for live trading |
| BacktestEngine | `src/backtest/backtest_engine.py` | Orchestrates backtest configuration and execution |
| BacktestRunner | `src/backtest/runner.py` | Runs event-driven backtest simulation |
| tradeExecutor | `src/live_trading/executor.py` | Executes real trades via API and updates database |
| BacktestExecutor | `src/backtest/executor.py` | Simulates trade execution with margin model |
| BasePortfolio | `src/portfolios/portfolio_BASE/strategy.py` | Abstract base class for all strategies |
| MQSDBConnector | `src/common/database/MQSDBConnector.py` | Thread-safe PostgreSQL connection pool |
| FMPMarketData | `src/orchestrator/marketData/fmpMarketData.py` | Financial Modeling Prep API client |
