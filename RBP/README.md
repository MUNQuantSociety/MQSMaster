# Relevance-Based Prediction (RBP)

RBP compares a prediction task with historical observations, forms relevance-weighted predictions over feature subsets and censoring thresholds, and combines them by adjusted fit. RBI (relevance-based importance) summarizes how including each feature changes that fit.

[Detailed workflow diagrams](WORKFLOW.md) · [Repository setup](../README.md) · [Portfolio integrations](../src/portfolios/README.md) · [Live stack](../docs/workflows/live_trading_workflow.md)

## Setup and commands

Use the shared Windows/macOS venv and requirements from the root README. Run from the repository root. Configure the PostgreSQL variables in `.env`, create tables, and backfill the selected tickers. RBP reads `market_data`; it does not download prices itself or require a pretrained model file.

The longest feature needs 252 daily observations; the target needs another 21 future observations, with usable rows on both sides of the research split. A short recent backfill will produce no usable experiment. The loader defaults to daily aggregation of intraday rows; quote-only data with NULL high/low/open is not equivalent to complete historical OHLCV.

```bash
# One research experiment; exports two CSVs
python -m RBP.main_rbp

# Continuous forecasts written to PostgreSQL
python -m src.orchestrator.rbp_runner
```

The research CLI reads [config.py](config.py); it has no argument parser. Change the dataclass defaults or instantiate `RBPConfig` from your own Python entrypoint. It writes `rbp_predictions.csv` and `rbp_rbi_scores.csv` in the current working directory. Stop the continuous runner with Ctrl+C.

| Research setting | Checked-in default |
|---|---|
| Tickers | AAPL, TSLA, AMD, MSFT, NVDA |
| History | Five calendar years backward from execution time |
| Train/test split | 2023-01-01 |
| Features | Five return/volatility features plus SMA, RSI, RMI, ROC, ATR, DMA, VWAP |
| Target | `target_return_21d` |
| Maximum feature subset | 1 feature |
| Censoring quantiles | 0.0, 0.2, 0.5, 0.8 |
| Maximum test tasks | 200, deterministic sampled subset |
| Parallelism | `n_jobs=-1`, joblib/loky |

The rolling history start moves as time passes; adjust the fixed split date so training data still exists. The service constructs its own config with a universe from manager weights, so the research default ticker list is not its live universe.

## Workflow diagrams

The [detailed workflow](WORKFLOW.md) uses the same top-down hierarchy as NLP, with `start.sh` at the root of the live overview:

- [Live workflow hierarchy](WORKFLOW.md#live-workflow-hierarchy): launcher, initialization, inputs, prediction, storage, and optional consumer.
- [Research workflow](WORKFLOW.md#research-workflow): configured experiment through CSV export.
- [Prediction internals](WORKFLOW.md#prediction-internals): feature subsets, relevance, censoring, adjusted fit, and RBI.
- [Forecast service](WORKFLOW.md#forecast-service-and-live-consumption): continuous database refresh and live consumption.
- [Portfolio integrations](WORKFLOW.md#three-distinct-portfolio-integrations): Portfolio 5, Portfolio 8, and the executor overlay.

Read the [forecast freshness limitation](WORKFLOW.md#current-forecast-freshness-limitation): a newly written forecast does not necessarily use the latest market row or an out-of-sample task.

## Verification and source map

A useful run has nonempty loaded and engineered row counts, a nonempty train/test split (research), and saved CSVs or actual DB forecast rows. Inspect timestamps and the [forecast freshness limitation](WORKFLOW.md#current-forecast-freshness-limitation) when interpreting service output. No external API key is needed solely to read previously populated RBP history.

| File | Responsibility |
|---|---|
| [main_rbp.py](main_rbp.py) | Research entrypoint and CSV exports |
| [config.py](config.py) | Experiment defaults |
| [database/loader.py](database/loader.py) | SQL and daily aggregation |
| [features/engineer.py](features/engineer.py) | Features, target, row filtering |
| [pipeline.py](pipeline.py) | Research split and parallel tasks |
| [models/predictor.py](models/predictor.py) | Grid and composite prediction |
| [models/importance.py](models/importance.py) | RBI calculation |
| [service.py](service.py) | Per-ticker forecast generation and persistence |
| [rbp_runner.py](../src/orchestrator/rbp_runner.py) | Continuous orchestration and universe refresh |
