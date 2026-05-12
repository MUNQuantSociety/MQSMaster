# NLP Sentiment Pipeline

This guide documents the NLP article scraping and sentiment pipeline in detail. The matching quick command reference lives in [NLP/README.md](../../NLP/README.md).

## Prerequisite: Download the FinBERT Model

Before running any sentiment scoring path, download the fine-tuned FinBERT model and place it inside the `NLP/` directory. The model is **not** committed to the repo.

1. Download from: <https://drive.google.com/drive/folders/1v7NjSuyFq4CTIctrw1bSv13JzkkMg1l8?usp=sharing>
2. Place the folder so the final path is `NLP/finbert-combined-final/`.
3. The directory must contain the model weights and tokenizer files (`config.json`, `tokenizer.json`, `model.safetensors`, etc.).

`NLP/sentiment/scorer.py` (re-exported from `NLP/sentiment_processor.py`) resolves the default model directory as `NLP/finbert-combined-final`. If that folder is missing, the pipeline falls back to `ProsusAI/finbert` on HuggingFace, which is **not** the production fine-tuned model and will produce different scores.

## Overview

The NLP system has three main entrypoints:

- `NLP/daemon.py` for continuous batched scraping and processing
- `NLP/fetch_articles.py` for one-off article fetches (multi-source merge)
- `NLP/process_sentiment_pipeline.py` for sentiment scoring and database writes

A separate synthetic health-check path lives in `NLP/monitor_daemon.py`.

> The five top-level CLI modules above (`fetch_articles`, `fetch_alt_articles`,
> `process_sentiment_pipeline`, `update_database`, `sentiment_processor`,
> `test_pipeline`) are thin **backwards-compat shims**. The implementation
> lives in subpackages - see [Package Layout](#package-layout). External
> imports such as `from NLP import fetch_articles` and the daemon's
> `subprocess.run([..., "-m", "NLP.fetch_articles", ...])` calls keep working.

## Runtime Model

The daemon runs continuously with a repeating cycle:

- Main scrape interval: 5 minutes
- Delay between batches: 2 minutes
- Batches are generated dynamically from portfolio config files
- Unsupported tickers are filtered before batching
- Recent activity is tracked to skip repeated sentiment processing when there are no new articles

## Key Commands

### Start the daemon

```bash
python NLP/daemon.py start
```

### Fetch articles manually

```bash
python -m NLP.fetch_articles AAPL 2025-12-01 2025-12-31
```

### Run sentiment processing

```bash
python -m NLP.process_sentiment_pipeline AAPL MSFT
```

### Test the pipeline

```bash
python -m NLP.test_pipeline
```

### Query the database

```bash
python -m NLP.update_database --query AAPL MSFT
```

### Run synthetic monitoring

```bash
python NLP/monitor_daemon.py --synthetic --max-log-age-hours 72
```

## Package Layout

The package is organised by responsibility. Top-level modules are thin
back-compat shims; the canonical implementations live in subpackages.

```
NLP/
├── daemon.py                    # entrypoint - NLPDaemon + TickerBatchProcessor
├── monitor_daemon.py            # entrypoint - DaemonMonitor / DaemonHealthCheck
├── fetch_articles.py            # shim -> scrapers.fmp + scrapers.aggregator
├── fetch_alt_articles.py        # shim -> ArticleScraper facade
├── sentiment_processor.py       # shim -> sentiment.scorer (SentimentProcessor)
├── process_sentiment_pipeline.py# shim -> sentiment.pipeline (SentimentPipeline)
├── update_database.py           # shim -> persistence.repository
├── test_pipeline.py             # shim -> sentiment.pipeline smoke run
├── core/                        # shared helpers
│   ├── paths.py                 # NLP_DIR, ARTICLES_DIR, SCORES_DIR, MODEL_DIR
│   ├── timestamps.py            # normalize_timestamp / normalize_published_date_column
│   └── logging_config.py        # get_logger + rotating-file logger
├── scrapers/
│   ├── base.py                  # BaseNewsScraper ABC, ArticleRecord
│   ├── fmp.py                   # FmpNewsScraper, FmpFetchStateStore
│   ├── yahoo.py                 # YahooNewsScraper (yfinance + retry)
│   ├── finviz.py                # FinvizNewsScraper (async aiohttp)
│   ├── alpha_vantage.py         # AlphaVantageNewsScraper
│   ├── truth_social.py          # TruthSocialScraper (Apify)
│   └── aggregator.py            # ArticleAggregator (multi-source merge + dedup)
├── sentiment/
│   ├── scorer.py                # FinBertSentimentScorer
│   └── pipeline.py              # SentimentPipeline
├── persistence/
│   └── repository.py            # NewsSentimentRepository
└── orchestration/
    └── ticker_universe.py       # TickerUniverse (load + batch tickers)
```

### Class map (canonical names)

| Concern              | Class                          | Module                              | Legacy alias                |
|----------------------|--------------------------------|-------------------------------------|-----------------------------|
| FMP paged scraper    | `FmpNewsScraper`               | `NLP.scrapers.fmp`                  | -                           |
| FMP fetch state      | `FmpFetchStateStore`           | `NLP.scrapers.fmp`                  | -                           |
| Yahoo scraper        | `YahooNewsScraper`             | `NLP.scrapers.yahoo`                | `ArticleScraper.scrape_yahoo`  |
| Finviz scraper       | `FinvizNewsScraper`            | `NLP.scrapers.finviz`               | `ArticleScraper.scrape_finviz` |
| Alpha Vantage        | `AlphaVantageNewsScraper`      | `NLP.scrapers.alpha_vantage`        | `ArticleScraper.scrape_alpha`  |
| Truth Social         | `TruthSocialScraper`           | `NLP.scrapers.truth_social`         | `ArticleScraper.trump_tracker` |
| Multi-source merge   | `ArticleAggregator`            | `NLP.scrapers.aggregator`           | `merge_all_sources`         |
| Sentiment scorer     | `FinBertSentimentScorer`       | `NLP.sentiment.scorer`              | `SentimentProcessor`        |
| Pipeline orchestrator| `SentimentPipeline`            | `NLP.sentiment.pipeline`            | -                           |
| DB repository        | `NewsSentimentRepository`      | `NLP.persistence.repository`        | `SentimentDatabaseUpdater`  |
| Ticker loader        | `TickerUniverse`               | `NLP.orchestration.ticker_universe` | -                           |
| Daemon loop          | `NLPDaemon`                    | `NLP.daemon`                        | `start_daemon` / `run_scraping_cycle` |
| Daemon health check  | `DaemonHealthCheck`            | `NLP.monitor_daemon`                | `run_synthetic_check`       |

## Data Layout

### Articles

Stored under `NLP/articles/` as per-ticker CSVs.

Examples:

- `AAPL.csv`
- `MSFT.csv`
- `GOOGL.csv`

### Sentiment scores

Stored under `NLP/sentiment_scores/` as per-ticker outputs.

Examples:

- `AAPL_article_scores.csv`
- `AAPL_daily_scores.csv`

### Model assets

The FinBERT model lives in `NLP/finbert-combined-final/`. This folder is **not** in the repo — download it from the Google Drive link in [Prerequisite: Download the FinBERT Model](#prerequisite-download-the-finbert-model) and place it under `NLP/` before running the pipeline.

## Portfolio-Driven Batching

The daemon loads tickers from portfolio config files under `src/portfolios/` and splits them into roughly equal batches.

Behavior:

- Reads tickers from portfolio configs in order
- Deduplicates tickers while preserving first-seen order
- Excludes tickers in the configured skip set, such as `^VIX`
- Builds up to four batches from the resulting list
- Processes each batch with a pause between batches

This means batch contents are data-driven and may change when portfolio configs change.

## Processing Flow

```text
Article scraping
      ↓
Per-ticker CSV merge
      ↓
Sentiment scoring with FinBERT
      ↓
Database update
      ↓
Trading strategy access
```

## Database Schema

The `news_sentiment` table stores:

- `ticker` - stock symbol
- `article_url` - unique article identifier
- `published_at` - article publication timestamp
- `sentiment_score` - FinBERT score in the range `[-1.0, 1.0]`
- `content_summary` - article summary or excerpt

## Monitoring and Logs

The daemon writes logs to `NLP/daemon.log` and reports:

- batch startup and completion
- fetch status per ticker
- sentiment processing status
- skip decisions for low-activity tickers
- memory cleanup activity
- errors and retries

## Configuration Notes

### Timing

- Scrape cycle interval: 300 seconds
- Delay between batches: 120 seconds
- Memory cleanup interval: every 10 cycles
- Log rotation size: 10 MB

### Environment

Typical runtime requirements include:

- Python dependencies from `requirements.txt`
- portfolio config files under `src/portfolios/`
- database connectivity for sentiment persistence
- optional market-data and API credentials for source fetches

## Legacy and Manual Workflows

Use the one-off commands above when you want to run a single pipeline step. Use the daemon only when you want continuous scheduled batching.

## Troubleshooting

### No tickers loaded

Check the portfolio config files under `src/portfolios/` and confirm they contain `TICKERS` entries.

### No new articles found

This is usually expected when the source has already been processed recently or no fresh articles exist for the selected date range.

### Sentiment step skipped

The daemon skips sentiment work when recent cycles show no new articles for a ticker.

### Monitoring failures

Use `NLP/monitor_daemon.py` for synthetic checks instead of treating external service probing as part of the normal pytest suite.

## Related Docs

- [Quick NLP commands](../../NLP/README.md)
- [Test modes](../TEST_MODES.md)
