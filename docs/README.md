# MQS Trading System documentation

For installation, cloning, Windows/macOS venv setup, credentials, database initialization, and commands for each project, start with the [root README](../README.md).

## Runtime workflows

| Guide | What it explains |
|---|---|
| [System architecture](workflows/system-architecture.md) | Components and their data dependencies |
| [Full live stack from start.sh](workflows/live_trading_workflow.md) | Preflight, all workers, data flow, watchdog, restart and shutdown behavior |
| [Inside the live bot](workflows/live-trading-workflow_detailed.md) | Portfolio threads, circuit breaker, sizing, OMS, transactional settlement |
| [Real-time ingestor](workflows/realtime-ingestor-workflow.md) | Feed routing, transformation, volume state, bulk inserts |
| [NLP high-level overview](workflows/nlp_workflow.md) | News to sentiment to strategy inputs |
| [Detailed NLP workflow](../NLP/WORKFLOW.md) | Live rotation, providers, model inference, CSVs, DB contract, backfill |
| [Detailed RBP workflow](../RBP/WORKFLOW.md) | Research pipeline, forecast service, predictor, portfolio integrations |
| [Portfolio workflow and setup](../src/portfolios/README.md) | Configuration, indicators, context routing, P5 and P6/P7/P8 |
| [Backtest flow](workflows/backtest-flow.md) | Process batches, event simulation, vector/Monte Carlo paths |
| [Data pipeline](workflows/data-pipeline.md) | Historical backfill, cache, ticker refresh |
| [Capital management](workflows/capital-management.md) | Master funding, allocation, internal transfers |
| [Database schema](workflows/database-schema.md) | Tables and relationships; NLP writer differences are documented in the NLP guide |

## Subsystem commands and reference

- [NLP setup and local experiments](../NLP/README.md)
- [RBP setup and configuration](../RBP/README.md)
- [Price/PnL worker setup](../src/orchestrator/realTime/README.md)
- [Backfill CLI](../src/orchestrator/backfill/Readme.md) and [ticker refresh](../src/orchestrator/backfill/update/refresh_README.md)
- [OMS design and implementation status](OMS/OMS_DESIGN.md): parent/child scheduling and pumps are implemented; durable order persistence, volume-profile integration, and LIMIT/STOP remain separate work
- [CFA calculator](../scripts/CFA/README.md) and [backtest analysis tools](../scripts/Backtest_Analysis/README.md)
- [CI/CD](CICD/CICD.md) and [test modes](TEST_MODES.md)

## Reading the diagrams

Mermaid blocks render on GitHub and in compatible Markdown viewers. Start with the full stack, then follow the subsystem links for detail. Solid arrows describe implemented control/data flow; dashed edges denote optional consumers or integrations where labeled.

The guides describe the checked-in code, not a live deployment audit. Operational timestamps are generally normalized to New York, but each loader's documented conversion matters (for example, RBP's daily grouping uses UTC-normalized timestamps). Runtime configuration and selected classes determine what actually runs.
