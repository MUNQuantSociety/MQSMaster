# Live Trading Workflow

Detailed flow of the live trading system from initialization to trade execution.

```mermaid
flowchart TD
    subgraph Init["Initialization"]
        START([main.py]) --> DB_INIT["Initialize MQSDBConnector"]
        DB_INIT --> EXEC_INIT["Initialize tradeExecutor"]
        EXEC_INIT --> ENGINE_INIT["Initialize RunEngine"]
        ENGINE_INIT --> SETUP["setup(portfolios_to_run)"]
    end

    subgraph Load["Portfolio Loading"]
        SETUP --> LOAD_CONFIG["Load config.json<br/>for each portfolio"]
        LOAD_CONFIG --> INSTANTIATE["Instantiate Portfolio<br/>with config_dict"]
        INSTANTIATE --> REGISTER["Register portfolio<br/>in engine"]
    end

    subgraph Run["Concurrent Execution"]
        REGISTER --> RUN["engine.run()"]
        RUN --> THREAD1["Thread: Portfolio 1"]
        RUN --> THREAD2["Thread: Portfolio 2"]
        RUN --> THREADN["Thread: Portfolio N"]
    end

    subgraph Loop["Portfolio Run Loop (per thread)"]
        THREAD1 --> POLL_LOOP{"Running?"}
        POLL_LOOP -->|Yes| GET_DATA["get_data(data_feeds)"]
        GET_DATA --> GEN_SIGNALS["generate_signals_and_trade()"]
        GEN_SIGNALS --> CHECK_FAIL{"Failure?"}
        CHECK_FAIL -->|No| SLEEP["Sleep(poll_interval)"]
        CHECK_FAIL -->|Yes| INC_FAIL["Increment failure count"]
        INC_FAIL --> CIRCUIT{"Circuit breaker<br/>tripped?"}
        CIRCUIT -->|No| SLEEP
        CIRCUIT -->|Yes| STOP_THREAD["Stop thread"]
        SLEEP --> POLL_LOOP
        POLL_LOOP -->|No| STOP_THREAD
    end

    subgraph DataFetch["Data Fetching"]
        GET_DATA --> ATOMIC["Atomic State Query"]
        ATOMIC --> CASH["Fetch CASH_EQUITY"]
        ATOMIC --> POS["Fetch POSITIONS"]
        GET_DATA --> MARKET["Fetch MARKET_DATA"]
        GET_DATA --> NOTIONAL["Fetch PORT_NOTIONAL"]
    end

    subgraph Signal["Signal Generation"]
        GEN_SIGNALS --> UPDATE_IND["Update Indicators"]
        UPDATE_IND --> BUILD_CTX["Build StrategyContext"]
        BUILD_CTX --> ON_DATA["Call OnData(context)"]
        ON_DATA --> TRADE_DECISION{"Trade Signal?"}
    end

    subgraph Execution["Trade Execution"]
        TRADE_DECISION -->|BUY/SELL| CALC_SIZE["Calculate position size"]
        CALC_SIZE --> BUYING_POWER["Check buying power"]
        BUYING_POWER --> GET_PRICE["Get current price<br/>(FMP API)"]
        GET_PRICE --> EXECUTE["Execute trade"]
        EXECUTE --> UPDATE_DB["Update database"]
        TRADE_DECISION -->|HOLD| NO_TRADE["No action"]
    end

    subgraph DBUpdate["Database Updates"]
        UPDATE_DB --> CASH_BOOK["cash_equity_book"]
        UPDATE_DB --> POS_BOOK["positions_book"]
        UPDATE_DB --> TRADE_LOG["trade_execution_logs"]
    end

    %% Styling
    classDef init fill:#e3f2fd,stroke:#1565c0
    classDef load fill:#fff3e0,stroke:#ef6c00
    classDef run fill:#e8f5e9,stroke:#2e7d32
    classDef loop fill:#fce4ec,stroke:#c2185b
    classDef data fill:#f3e5f5,stroke:#7b1fa2
    classDef signal fill:#e0f7fa,stroke:#00838f
    classDef exec fill:#fff8e1,stroke:#f9a825
    classDef db fill:#efebe9,stroke:#5d4037

    class START,DB_INIT,EXEC_INIT,ENGINE_INIT,SETUP init
    class LOAD_CONFIG,INSTANTIATE,REGISTER load
    class RUN,THREAD1,THREAD2,THREADN run
    class POLL_LOOP,GET_DATA,GEN_SIGNALS,CHECK_FAIL,INC_FAIL,CIRCUIT,SLEEP,STOP_THREAD loop
    class ATOMIC,CASH,POS,MARKET,NOTIONAL data
    class UPDATE_IND,BUILD_CTX,ON_DATA,TRADE_DECISION signal
    class CALC_SIZE,BUYING_POWER,GET_PRICE,EXECUTE,UPDATE_DB,NO_TRADE exec
    class CASH_BOOK,POS_BOOK,TRADE_LOG db
```

## Key Features

### Circuit Breaker Pattern
- Tracks consecutive failures per portfolio
- Stops portfolio thread after `max_consecutive_failures` (default: 5)
- Prevents cascading failures from affecting other portfolios

### Thread Safety
- Each portfolio runs in its own thread
- Database connector uses connection pooling
- FMPMarketData has thread-safe rate limiting

### Margin Model
- Supports long and short positions
- Configurable leverage (default: 2.0x)
- Buying power calculation mirrors live trading constraints
