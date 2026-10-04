# Workflow diagrams

[Repository setup](../../README.md) · [All documentation](../README.md)

| Start here | Scope |
|---|---|
| [Full live stack](live_trading_workflow.md) | Everything launched from start.sh, including lifecycle and supporting workers |
| [Live bot detail](live-trading-workflow_detailed.md) | Threads, portfolio state, signals, sizing, direct/OMS fills |
| [Real-time ingestor](realtime-ingestor-workflow.md) | Separate complete ingestion workflow |
| [NLP overview](nlp_workflow.md) | High-level news-to-sentiment flow |
| [Detailed NLP](../../NLP/WORKFLOW.md) | Module-level workflow in the NLP folder |
| [Detailed RBP](../../RBP/WORKFLOW.md) | Research and service diagrams in the RBP folder |
| [Detailed portfolios](../../src/portfolios/README.md) | Shared lifecycle and strategy-family diagrams in the portfolio folder |
| [Portfolio overview](portfolio-strategy-flow.md) | Compact shared flow |
| [System architecture](system-architecture.md) | Relationship between projects |
| [Backtests](backtest-flow.md) | Event and fast paths |
| [Data pipeline](data-pipeline.md) | Backfill, cache, and ticker maintenance |
| [Capital management](capital-management.md) | Funding and allocation |
| [Database schema](database-schema.md) | Book/data table relationships |

The live, ingestion, NLP, RBP, portfolio, and database-operation flowcharts read **top to bottom**. Each has one highlighted root: the entrypoint or operation that starts that workflow. Branches sit beneath it; a labelled next-cycle or retry step describes repetition without drawing an arrow back above the root. The database ER diagram remains a table-relationship reference.

Mermaid code fences render directly on GitHub. When behavior changes, update the subsystem diagram beside its code and the high-level overview if connections or lifecycle change. Include optional gates, failure behavior, and the actual storage destination; avoid presenting proposed features as implemented.
