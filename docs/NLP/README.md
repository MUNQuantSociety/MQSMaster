# NLP documentation

The maintained NLP guide lives beside the implementation:

- [Setup, model download, and commands](../../NLP/README.md)
- [Detailed workflow: live loop, scraping, scoring, persistence, backfill](../../NLP/WORKFLOW.md)
- [High-level diagram](../workflows/nlp_workflow.md)
- [Windows/macOS environment and database setup](../../README.md)

## Prerequisite: Download the FinBERT Model

Follow [NLP model setup](../../NLP/README.md#setup-download-the-finbert-model). The required directory is `NLP/finbert-combined-final/`; the production scorer expects positive/neutral/negative indices 0/1/2 and has no automatic model fallback.

## Run from the repository root

```bash
python -m NLP.main_NLP
```

For one historical batch instead:

```bash
python -m NLP.backfill_NLP --start 2025-01-01 --end 2025-12-31 --tickers AAPL MSFT
```

Configure the database column prerequisite before either DB-backed path. The [detailed workflow](../../NLP/WORKFLOW.md#database-contract) explains why article persistence and market-data sentiment synchronization must be checked separately.
