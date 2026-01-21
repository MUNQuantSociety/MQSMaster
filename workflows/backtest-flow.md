# Backtesting Workflow

Complete flow of the backtesting engine from setup to report generation.

```mermaid
flowchart TD
    subgraph Setup["Backtest Setup"]
        START([main_backtest.py]) --> DB["Initialize MQSDBConnector"]
        DB --> ENGINE["Create BacktestEngine"]
        ENGINE --> CONFIG["setup()<br/>- portfolio_classes<br/>- start_date, end_date<br/>- initial_capital<br/>- slippage"]
    end

    subgraph EngineRun["BacktestEngine.run()"]
        CONFIG --> LOOP_PORT["For each portfolio class"]
        LOOP_PORT --> LOAD_CFG["Load config.json<br/>dynamically"]
        LOAD_CFG --> INST_PORT["Instantiate portfolio<br/>with config_dict"]
        INST_PORT --> CREATE_RUNNER["Create BacktestRunner"]
        CREATE_RUNNER --> RUN_RUNNER["runner.run()"]
    end

    subgraph Runner["BacktestRunner Execution"]
        RUN_RUNNER --> PREP_DATA["_prepare_data()"]
        PREP_DATA --> SETUP_EXEC["_setup_executor()"]
        SETUP_EXEC --> EVENT_LOOP["_run_event_loop()"]
        EVENT_LOOP --> CALC_RESULTS["_calculate_results()"]
        CALC_RESULTS --> GEN_REPORT["generate_backtest_report()"]
        GEN_REPORT --> RESTORE["_restore_executor()"]
    end

    subgraph DataPrep["Data Preparation"]
        PREP_DATA --> ADJ_START["Adjust start date<br/>for lookback_days"]
        ADJ_START --> FETCH["fetch_historical_data()"]
        FETCH --> SORT["Sort by timestamp"]
        SORT --> VALIDATE{"Data valid?"}
        VALIDATE -->|No| ABORT["Abort backtest"]
        VALIDATE -->|Yes| CONTINUE["Continue"]
    end

    subgraph EventLoop["Event Loop Simulation"]
        EVENT_LOOP --> GROUP_DATA["Group data by timestamp"]
        GROUP_DATA --> FILTER_TS["Filter timestamps >= start_date"]
        FILTER_TS --> PROGRESS["tqdm progress bar"]
        PROGRESS --> ITER["For each timestamp"]
        
        ITER --> POLL_CHECK{"Poll interval<br/>elapsed?"}
        POLL_CHECK -->|No| SKIP["Skip"]
        POLL_CHECK -->|Yes| UPDATE_PRICES["Update executor prices"]
        
        UPDATE_PRICES --> SLICE["Create historical slice<br/>(lookback window)"]
        SLICE --> BUILD_DATA["Build data_dict"]
        BUILD_DATA --> GEN_SIGNALS["generate_signals_and_trade()"]
        GEN_SIGNALS --> RECORD["Record performance"]
        RECORD --> NEXT{"More timestamps?"}
        NEXT -->|Yes| ITER
        NEXT -->|No| DONE["Loop complete"]
        SKIP --> NEXT
    end

    subgraph Executor["BacktestExecutor"]
        GEN_SIGNALS --> EXEC_TRADE["execute_trade()"]
        EXEC_TRADE --> SLIPPAGE["Apply slippage"]
        SLIPPAGE --> SIZE["Calculate trade size"]
        SIZE --> MARGIN["Check margin/buying power"]
        MARGIN --> UPDATE_STATE["Update cash & positions"]
        UPDATE_STATE --> LOG_TRADE["Log trade"]
    end

    subgraph Reporting["Report Generation"]
        GEN_REPORT --> TRADE_LOG["trade_log.csv"]
        GEN_REPORT --> PERF_ABS["performance_timeseries_absolute.csv"]
        GEN_REPORT --> PERF_PCT["performance_timeseries_percentage.csv"]
        GEN_REPORT --> SUMMARY["summary_metrics.csv"]
        GEN_REPORT --> MINUTE["performance_timeseries_minute_by_minute.csv"]
        GEN_REPORT --> BENCHMARK["benchmark_buy_and_hold_performance.csv"]
        GEN_REPORT --> ROLLING["Rolling statistics (30D, 90D, 180D)"]
        GEN_REPORT --> MONTHLY["monthly_returns.csv"]
        GEN_REPORT --> RISK["Portfolio risk analytics"]
    end

    %% Styling
    classDef setup fill:#e3f2fd,stroke:#1565c0
    classDef engine fill:#fff3e0,stroke:#ef6c00
    classDef runner fill:#e8f5e9,stroke:#2e7d32
    classDef data fill:#f3e5f5,stroke:#7b1fa2
    classDef loop fill:#fce4ec,stroke:#c2185b
    classDef exec fill:#fff8e1,stroke:#f9a825
    classDef report fill:#e0f2f1,stroke:#00695c

    class START,DB,ENGINE,CONFIG setup
    class LOOP_PORT,LOAD_CFG,INST_PORT,CREATE_RUNNER,RUN_RUNNER engine
    class PREP_DATA,SETUP_EXEC,EVENT_LOOP,CALC_RESULTS,GEN_REPORT,RESTORE runner
    class ADJ_START,FETCH,SORT,VALIDATE,ABORT,CONTINUE data
    class GROUP_DATA,FILTER_TS,PROGRESS,ITER,POLL_CHECK,SKIP,UPDATE_PRICES,SLICE,BUILD_DATA,RECORD,NEXT,DONE loop
    class EXEC_TRADE,SLIPPAGE,SIZE,MARGIN,UPDATE_STATE,LOG_TRADE exec
    class TRADE_LOG,PERF_ABS,PERF_PCT,SUMMARY,MINUTE,BENCHMARK,ROLLING,MONTHLY,RISK report
```

## Backtest Output Files

| File | Description |
|------|-------------|
| `trade_log.csv` | All executed trades with timestamps, prices, quantities |
| `performance_timeseries_absolute.csv` | Portfolio value over time |
| `performance_timeseries_percentage.csv` | Returns as percentages |
| `summary_metrics.csv` | Final value, max drawdown, Sharpe ratio |
| `performance_timeseries_minute_by_minute.csv` | High-frequency performance |
| `benchmark_buy_and_hold_performance.csv` | Buy-and-hold comparison |
| `30D_Rolling.csv`, `90D_Rolling.csv`, `180D_Rolling.csv` | Rolling statistics |
| `monthly_returns.csv` | Monthly return breakdown |
| `portfolio_risk_components.csv` | Individual asset volatilities |
| `annualized_correlation_matrix.csv` | Asset correlations |
| `rolling_portfolio_risk.csv` | Rolling portfolio volatility |

## Key Metrics Calculated

- **Max Drawdown**: Peak-to-trough decline
- **Sharpe Ratio**: Annualized risk-adjusted return (√252 factor)
- **Rolling Statistics**: Mean return and volatility over windows
- **Monthly Returns**: Resampled end-of-month returns
