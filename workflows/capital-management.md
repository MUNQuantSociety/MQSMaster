# Capital Management Workflow

Risk management, capital allocation, and fund transfers between portfolios.

```mermaid
flowchart TD
    subgraph External["External Capital"]
        ADD["Add Capital"]
        WITHDRAW["Withdraw Capital"]
    end

    subgraph ManageCapital["manage_capital.py"]
        MC_START["update_capital()"]
        MC_VALIDATE["Validate action<br/>(ADD/WITHDRAW)"]
        MC_BALANCE["Get current balance"]
        MC_CHECK{"Sufficient<br/>funds?"}
        MC_CALC["Calculate new balance"]
        MC_LOG["Log to trade_execution_logs"]
        MC_UPDATE["Update cash_equity_book"]
    end

    subgraph Master["Master Portfolio (ID: 0)"]
        MASTER_CASH["Cash Balance"]
    end

    subgraph Allocator["DailyAllocator"]
        DA_START["run_allocation()"]
        DA_INIT["initialize_new_portfolios()"]
        DA_MASTER["Get master cash"]
        DA_LOOP["For each strategy portfolio"]
        DA_VALUE["Calculate total value<br/>(cash + positions)"]
        DA_TARGET["Calculate target value<br/>(total_equity × weight)"]
        DA_ADJUST["Calculate adjustment"]
        DA_TRANSFER["Execute internal transfer"]
    end

    subgraph Strategies["Strategy Portfolios"]
        P1["Portfolio 1<br/>(weight: 0.3)"]
        P2["Portfolio 2<br/>(weight: 0.3)"]
        P3["Portfolio 3<br/>(weight: 0.4)"]
    end

    subgraph Database["Database Tables"]
        CASH_BOOK["cash_equity_book"]
        TRADE_LOG["trade_execution_logs"]
        POS_BOOK["positions_book"]
    end

    %% External capital flow
    ADD --> MC_START
    WITHDRAW --> MC_START
    MC_START --> MC_VALIDATE
    MC_VALIDATE --> MC_BALANCE
    MC_BALANCE --> MC_CHECK
    MC_CHECK -->|No| REJECT["Reject withdrawal"]
    MC_CHECK -->|Yes| MC_CALC
    MC_CALC --> MC_LOG
    MC_LOG --> MC_UPDATE
    MC_UPDATE --> MASTER_CASH

    %% Allocator flow
    DA_START --> DA_INIT
    DA_INIT --> DA_MASTER
    DA_MASTER --> MASTER_CASH
    DA_MASTER --> DA_LOOP
    DA_LOOP --> DA_VALUE
    DA_VALUE --> DA_TARGET
    DA_TARGET --> DA_ADJUST
    DA_ADJUST --> DA_TRANSFER

    %% Transfer to strategies
    DA_TRANSFER --> P1
    DA_TRANSFER --> P2
    DA_TRANSFER --> P3

    %% Database connections
    MC_LOG --> TRADE_LOG
    MC_UPDATE --> CASH_BOOK
    DA_TRANSFER --> CASH_BOOK
    DA_TRANSFER --> TRADE_LOG
    DA_VALUE --> POS_BOOK
    DA_VALUE --> CASH_BOOK

    %% Styling
    classDef external fill:#e3f2fd,stroke:#1565c0
    classDef manage fill:#fff3e0,stroke:#ef6c00
    classDef master fill:#e8f5e9,stroke:#2e7d32
    classDef allocator fill:#fce4ec,stroke:#c2185b
    classDef strategy fill:#f3e5f5,stroke:#7b1fa2
    classDef db fill:#e0f7fa,stroke:#00838f

    class ADD,WITHDRAW external
    class MC_START,MC_VALIDATE,MC_BALANCE,MC_CHECK,MC_CALC,MC_LOG,MC_UPDATE,REJECT manage
    class MASTER_CASH master
    class DA_START,DA_INIT,DA_MASTER,DA_LOOP,DA_VALUE,DA_TARGET,DA_ADJUST,DA_TRANSFER allocator
    class P1,P2,P3 strategy
    class CASH_BOOK,TRADE_LOG,POS_BOOK db
```

## Daily Allocation Sequence

```mermaid
sequenceDiagram
    participant Scheduler
    participant Allocator
    participant DB
    participant FMP

    Scheduler->>Allocator: run_allocation()
    
    Allocator->>DB: Get master portfolio cash
    DB-->>Allocator: master_cash
    
    loop For each strategy portfolio
        Allocator->>DB: Get cash balance
        DB-->>Allocator: cash
        
        Allocator->>DB: Get positions
        DB-->>Allocator: positions
        
        loop For each position
            Allocator->>FMP: get_current_price(ticker)
            FMP-->>Allocator: price
        end
        
        Allocator->>Allocator: Calculate total value
        Allocator->>Allocator: Calculate target (equity × weight)
        Allocator->>Allocator: Calculate adjustment
    end
    
    Allocator->>DB: Begin transaction
    
    loop For each adjustment
        alt Positive adjustment (fund strategy)
            Allocator->>DB: Debit master, credit strategy
        else Negative adjustment (withdraw from strategy)
            Allocator->>DB: Debit strategy, credit master
        end
    end
    
    Allocator->>DB: Commit transaction
```

## Configuration Schema

```json
{
  "master_portfolio_id": "0",
  "currency": "USD",
  "portfolio_weights": {
    "1": 0.30,
    "2": 0.30,
    "3": 0.40
  }
}
```

## Internal Transfer Recording

Each internal transfer creates two entries in `trade_execution_logs`:

| Field | From Portfolio | To Portfolio |
|-------|----------------|--------------|
| ticker | USD_CASH | USD_CASH |
| side | SELL | BUY |
| quantity | amount | amount |
| arrival_price | 1.0 | 1.0 |
| exec_price | 1.0 | 1.0 |

## Capital Flow Diagram

```mermaid
flowchart LR
    subgraph External
        INVESTOR["Investor"]
    end

    subgraph System
        MASTER["Master Portfolio<br/>(ID: 0)"]
        
        subgraph Strategies
            S1["Strategy 1<br/>30%"]
            S2["Strategy 2<br/>30%"]
            S3["Strategy 3<br/>40%"]
        end
    end

    INVESTOR -->|"Add Capital"| MASTER
    MASTER -->|"Withdraw"| INVESTOR
    
    MASTER <-->|"Daily Rebalance"| S1
    MASTER <-->|"Daily Rebalance"| S2
    MASTER <-->|"Daily Rebalance"| S3

    style MASTER fill:#4caf50,color:#fff
    style S1 fill:#2196f3,color:#fff
    style S2 fill:#2196f3,color:#fff
    style S3 fill:#2196f3,color:#fff
```

## CLI Usage

```bash
# Add capital to master portfolio
python -m src.risk_manager.manage_capital --action ADD --amount 100000

# Withdraw capital from master portfolio
python -m src.risk_manager.manage_capital --action WITHDRAW --amount 50000

# Run daily allocation (typically scheduled)
python -m src.risk_manager.daily_allocator
```
