# RBP: detailed workflow

[Setup and commands](README.md) · [NLP workflow](../NLP/WORKFLOW.md) · [Repository setup](../README.md)

RBP has two entrypoints: [rbp_runner.py](../src/orchestrator/rbp_runner.py) continuously refreshes database forecasts for the live stack; [main_rbp.py](main_rbp.py) runs a research experiment and exports CSVs. Both use the feature engineer and relevance-based predictor.

## Live workflow hierarchy

Read from the highlighted root downward: launcher, worker initialization, inputs, processing, storage, and consumers. Each arrow leads to the next stage or a branch within it. Repeated work ends at a next-cycle leaf to keep the hierarchy readable. The worker command can also be run directly from the repository root after activating the venv.

```mermaid
flowchart TD
    START(["bash start.sh"]) --> LAUNCH["Market-worker launch: src/orchestrator/rbp_runner.py"]
    LAUNCH --> ENTRY["Worker entrypoint<br/>python -m src.orchestrator.rbp_runner"]
    ENTRY --> INIT["1. Initialize runner<br/>Register shutdown handlers; attempt schema bootstrap"]
    INIT --> INPUT["2. Load ticker universe<br/>Positive manager weights and portfolio configs"]
    INPUT --> SERVICE["Connect DB; create RBPForecastService"]
    SERVICE --> READY{"Universe nonempty?"}
    READY -->|Yes| DATA["3. Load market_data history<br/>Aggregate daily bars; engineer features and targets"]
    DATA --> PREDICT["4. For each usable ticker<br/>Build training matrix; predict and calculate RBI"]
    PREDICT --> STORE["5. Persist rbp_forecasts<br/>Ignore conflicting forecast keys"]
    STORE -.-> CONSUMER["Optional executor RBPOverlay<br/>Blend forecast agreement into sizing confidence"]
    STORE --> WAIT["Sleep 300 seconds; respond to shutdown signals"]
    READY -->|No| IDLE["Idle until the next universe check"]
    IDLE --> WAIT
    WAIT --> CYCLE["Next cycle: reload universe and refresh forecasts"]
    style START fill:#dbeafe,stroke:#2563eb,stroke-width:2px
```

Solid arrows show worker execution; the dashed branch shows a downstream consumer, not another worker step. `start.sh` supervises RBP as a market worker and terminates it when the market watchdog stops the market workers. It does not restart a crashed RBP worker. Direct invocation has no market-hours watchdog. See the [launcher lifecycle](../docs/workflows/live_trading_workflow.md#market-watchdog-and-persistent-worker-lifecycle).

The sections below expand the [research command](#research-workflow), [prediction internals](#prediction-internals), [forecast service](#forecast-service-and-live-consumption), and [portfolio integrations](#three-distinct-portfolio-integrations). Review the [forecast freshness limitation](#current-forecast-freshness-limitation) before interpreting service output.

## Research workflow

```mermaid
flowchart TD
    CLI(["python -m RBP.main_rbp"]) --> CFG["RBPConfig"]
    CFG --> LOAD["MarketDataLoader<br/>Query market_data by ticker and date"]
    LOAD --> DAILY["Coerce numeric fields<br/>Aggregate daily OHLCV per ticker"]
    DAILY --> FEATURES["FeatureEngineer<br/>21, 63, 252-observation returns<br/>21 and 63-observation volatility<br/>Seven technical indicators"]
    FEATURES --> TARGET["Target: close 21 observations ahead / close - 1"]
    TARGET --> CLEAN["Drop rows missing any required feature or target"]
    CLEAN --> SPLIT["Split at train_test_split_date<br/>Cap test tasks if configured"]
    SPLIT --> WORK["Parallel prediction for each test row"]
    WORK --> PRED["RBPPredictor: feature-subset and quantile grid"]
    PRED --> RBI["RBICalculator: feature importance per task"]
    RBI --> OUT["Return prediction and RBI DataFrames"]
    OUT --> CSV["Write rbp_predictions.csv<br/>and rbp_rbi_scores.csv"]
    style CLI fill:#dbeafe,stroke:#2563eb,stroke-width:2px
```

The loader converts timestamps to UTC and removes the timezone before daily grouping. Its actual grouping is by those normalized dates; do not assume a separate New York exchange-session resampler. Train/test rows from all configured tickers enter the research training matrix together. Empty data, empty splits, or failure of all prediction tasks abort the experiment.

### Prediction internals

```mermaid
flowchart TD
    TASK(["Task features and training X/Y"]) --> SUBSET["For each configured feature subset"]
    SUBSET --> DIST["Mahalanobis distance statistics"]
    DIST --> REL["Relevance scores for training observations"]
    REL --> Q["For each censoring quantile<br/>Retain relevant observations and form weights"]
    Q --> CELL["Weighted target prediction<br/>Fit, asymmetry, adjusted fit"]
    CELL --> GRID["Grid of cell results"]
    GRID --> COMPOSITE["Clip negative adjusted fits to zero<br/>Normalize positive fits; combine predictions"]
    GRID --> RBI["RBI per feature:<br/>mean fit with feature minus mean fit without"]
    style TASK fill:#dbeafe,stroke:#2563eb,stroke-width:2px
```

If all adjusted fits are zero, the predictor returns 0.0 with an unreliable-prediction warning. Increasing subset size expands the grid combinatorially. Core calculations live in [core/](core/), with prediction and importance in [models/](models/).

## Forecast service and live consumption

```mermaid
flowchart TD
    START(["python -m src.orchestrator.rbp_runner"]) --> SCHEMA["Attempt schema bootstrap; connect DB"]
    SCHEMA --> UNIVERSE["collect_universe<br/>Manager portfolio_weights greater than zero<br/>Union each config's TICKERS"]
    UNIVERSE --> P678["If P6, P7, or P8 enabled:<br/>also union portfolio_6/universe.json"]
    P678 --> SERVICE["RBPForecastService with five-year lookback"]
    SERVICE --> REFRESH["Load and engineer history for current universe"]
    REFRESH --> EACH["For each ticker: sort engineered rows"]
    EACH --> TRAIN["Training rows before asof minus one day<br/>Cache training matrix by latest training date"]
    TRAIN --> TASK["Task = latest surviving engineered row"]
    TASK --> GRID["Predict plus RBI top five features"]
    GRID --> RECORD["ticker, asof, horizon_days=21, y_pred,<br/>rbi_top, model_version, generated_at"]
    RECORD --> DB[("rbp_forecasts<br/>Conflict: ticker, asof, horizon_days, model_version")]
    DB --> SLEEP["Sleep 300 seconds in interruptible steps<br/>Reload universe; repeat"]
    SLEEP --> NEXT["Begin next refresh with updated universe"]
    DB -.-> OVERLAY["Optional RBPOverlay in live executor sizing"]
    OVERLAY --> CONF["Blend forecast agreement into confidence"]
    style START fill:#dbeafe,stroke:#2563eb,stroke-width:2px
```

The service uses the same feature engineer and predictor as research, but constructs training data per ticker and writes forecasts rather than research CSVs. Conflicts are ignored (`DO NOTHING`), not updated. Per-ticker exceptions are logged and skipped; refresh-level errors are retried next cycle. An empty universe idles and is rechecked. The runner handles SIGINT/SIGTERM and closes its DB pool.

The runner's positive-weight universe is **not** the live strategy class selection. `src/main.py` selects live threads explicitly, whereas the manager weights choose the forecast universe and capital shares. The optional executor overlay reads recent forecasts with a short cache; missing/stale/error results leave the strategy confidence unchanged. The checked-in manager config has no `rbp_overlay` section, so the entrypoint leaves it disabled.

### Current forecast freshness limitation

[FeatureEngineer.engineer](features/engineer.py) drops rows without the forward 21-observation target, including the newest 21 daily rows. [RBPForecastService](service.py) then predicts from the **latest surviving engineered row** and labels the output with the current `asof`. That task can also occur in its training set. Consequently, a fresh `generated_at` is not proof of a fresh, out-of-sample 21-day forecast.

The research split also does not purge training rows whose forward-target windows cross the split. These diagrams describe the implemented algorithm; neither command establishes point-in-time research validity by itself.

## Three distinct portfolio integrations

| Integration | Implementation and behavior |
|---|---|
| Portfolio 5 | [RBPModel](../src/portfolios/portfolio_5/rbp_model.py) is a separate implementation, lazily fitted inside [RBPStrategy](../src/portfolios/portfolio_5/strategy.py); its rolling windows operate on supplied bars |
| Portfolio 8 | [Portfolio8Strategy](../src/portfolios/portfolio_8/strategy.py) calls the research pipeline to blend an RBP rank into P6 screening; failures fall back to P6 |
| Live executor overlay | [RBPOverlay](../src/risk_manager/rbp_overlay.py) reads `rbp_forecasts` produced by this service and blends sizing confidence when enabled |

The current research prediction output lacks a ticker column. P8 therefore takes its fallback branch that assigns the test-window average prediction to each capped ticker; do not describe that path as distinct current per-ticker forecasts.
