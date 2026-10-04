# Portfolio / strategy flow

The detailed guide is maintained beside the strategy code: [src/portfolios/README.md](../../src/portfolios/README.md).

It includes configuration and activation, all available strategies, indicator warmup, the shared live/event lifecycle, context-to-OMS routing, direct-executor exceptions, and separate P5 and P6/P7/P8 diagrams.

```mermaid
flowchart LR
    CFG["Selected class and sibling config.json"] --> INIT["Instantiate strategy<br/>Warm indicators"]
    INIT --> DATA["Live DB poll or historical event bar"]
    DATA --> CTX["Update indicators<br/>Build StrategyContext"]
    CTX --> ONDATA["OnData"]
    ONDATA --> ORDERS["Context buy/sell<br/>or explicit target-weight call"]
    ORDERS --> EXEC["Direct fill or optional OMS schedule"]
    EXEC --> BOOKS["Live DB books<br/>or simulated backtest books"]
    BOOKS --> DATA
```

Live portfolios share one executor across threads; event backtests have separate executors per portfolio. Fast backtests use vector adapters. Strategies with custom direct-executor calls do not automatically use OMS merely because their config enables it.

Related: [live engine detail](live-trading-workflow_detailed.md), [backtest flow](backtest-flow.md), [NLP detail](../../NLP/WORKFLOW.md), [RBP detail](../../RBP/README.md).
