# NLP Sentiment Pipeline

This guide documents the NLP article scraping and sentiment pipeline in detail. The matching quick command reference lives in [NLP/README.md](../../NLP/README.md).

## Prerequisite: Download the FinBERT Model

Before running any sentiment scoring path, download the fine-tuned FinBERT model and place it inside the `NLP/` directory. The model is **not** committed to the repo.

1. Download from: <https://drive.google.com/drive/folders/1v7NjSuyFq4CTIctrw1bSv13JzkkMg1l8?usp=sharing>
2. Place the folder so the final path is `NLP/finbert-finetuned-final/`.
3. The directory must contain the model weights and tokenizer files (`config.json`, `tokenizer.json`, `model.safetensors`, etc.).

`NLP/sentiment/scorer.py` (re-exported from `NLP/sentiment_processor.py`) resolves the default model directory as `NLP/finbert-finetuned-final`. **There is no HuggingFace fallback**: a missing model directory now raises `FileNotFoundError` at scorer init. The earlier `ProsusAI/finbert` fallback used a different label index ordering (positive/negative/neutral) than our fine-tuned model (positive/neutral/negative), so `probs[:, 0] - probs[:, 2]` produced silently wrong scores on the fallback path. Download the fine-tuned model before running any sentiment path.

## Overview

The NLP system has four main entrypoints:

- `NLP/main_NLP.py` for the **live** continuous pipeline (ticker discovery → fetch → score → database, every 5 min)
- `NLP/backfill_NLP.py` for **historical** date-range backfill (multi-source per ticker; refuses to run during market hours)
- `NLP/fetch_articles.py` for one-off article fetches (multi-source merge)
- `NLP/process_sentiment_pipeline.py` for sentiment scoring and database writes

A separate synthetic health-check path lives in `NLP/monitor_daemon.py`.

> `main_NLP.py` is the NLP counterpart of `src/main.py`: a thin entrypoint
> that instantiates `NLP.runner.NLPRunner` and calls `.run()`. All
> orchestration logic (`NLPRunner`, `TickerBatchProcessor`,
> `SkipStatsTracker`) lives in `NLP/runner.py`.
>
> The five top-level CLI modules (`fetch_articles`, `fetch_alt_articles`,
> `process_sentiment_pipeline`, `update_database`, `sentiment_processor`,
> `test_pipeline`) are thin **backwards-compat shims**. The implementation
> lives in subpackages - see [Package Layout](#package-layout). External
> imports such as `from NLP import fetch_articles` and the runner's
> `subprocess.run([..., "-m", "NLP.fetch_articles", ...])` calls keep working.

## Runtime Model

### Live mode (`NLP/main_NLP.py`)

`NLPRunner` runs continuously with a repeating cycle:

- Main scrape interval: 5 minutes
- Tickers loaded from `src/orchestrator/backfill/tickers.json` (`^VIX` excluded)
- Per cycle: every ticker gets a fast **FMP page-0** check (newest articles)
- One of 4 batches per cycle additionally pulls Yahoo + Finviz + Alpha Vantage (round-robin), so each ticker gets a multi-source refresh roughly every ~20 minutes
- FinBERT model loads **once** at process startup, shared across all tickers in-process (no subprocess fan-out)
- `SkipStatsTracker` suppresses the sentiment step for tickers that produced no new articles in the last 5 cycles
- Always allowed during market hours (the live ingestor needs continuous news flow)

### Backfill mode (`NLP/backfill_NLP.py`)

One-shot CLI for a historical date range:

- Iterates tickers serially; per ticker runs `ArticleAggregator.run(start, end)` (FMP all pages + Yahoo + Finviz + Alpha Vantage) followed by `SentimentPipeline.process_ticker_complete`
- **Hard rule**: refuses to run while `is_market_open()` returns True. Pass `--wait` to block until 16:00 ET then resume; otherwise the runtime exits with code 2

## Rate Limit / Budget

FMP enforces a 3000 req/min account cap, shared between the realtime ingestor and the NLP pipeline. They split it statically:

| Process              | Cap         | Source                                        |
|----------------------|-------------|-----------------------------------------------|
| Realtime ingestor    | 1000/min    | `FMPRateLimiter.for_realtime()`               |
| NLP (live + backfill)| 2000/min    | `FMPRateLimiter.for_nlp()`                    |

Realtime currently consumes ~1 req/min (one batch-quote per minute), so the 1000 cap is effectively reserved headroom. NLP's 2000/min comfortably covers the 1919-ticker live sweep (≈384 req/min).

Limiter lives at `src/common/fmp_rate_limiter.py`. `ArticlesGateway` and `FMPMarketData` both acquire from the appropriate per-process singleton before each HTTP call.

## Key Commands

### Start the live pipeline

```bash
python NLP/main_NLP.py
```

### Backfill a historical date range

```bash
# All tickers between Jan 1 and Dec 31, 2025
python NLP/backfill_NLP.py --start 2025-01-01 --end 2025-12-31

# Subset of tickers
python NLP/backfill_NLP.py --start 2025-12-01 --end 2025-12-31 --tickers AAPL MSFT

# Block until market close instead of aborting
python NLP/backfill_NLP.py --start 2025-01-01 --end 2025-12-31 --wait
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
├── main_NLP.py                  # entrypoint - live pipeline runner, mirrors src/main.py
├── backfill_NLP.py              # entrypoint - historical date-range backfill
├── runner.py                    # NLPRunner + TickerProcessor + SkipStatsTracker
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
    └── ticker_universe.py       # TickerUniverse (load tickers.json + batch)
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
| Pipeline runner      | `NLPRunner`                    | `NLP.runner`                        | `NLPDaemon` (former name)   |
| Runner health check  | `DaemonHealthCheck`            | `NLP.monitor_daemon`                | `run_synthetic_check`       |

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

The FinBERT model lives in `NLP/finbert-finetuned-final/`. This folder is **not** in the repo — download it from the Google Drive link in [Prerequisite: Download the FinBERT Model](#prerequisite-download-the-finbert-model) and place it under `NLP/` before running the pipeline.

## Ticker Universe and Batching

`NLPRunner` loads tickers from `src/orchestrator/backfill/tickers.json` — the same universe the backfill pipeline ingests — and splits them into roughly equal batches.

Behavior:

- Reads the JSON list directly (no portfolio config indirection)
- Deduplicates tickers while preserving first-seen order
- Excludes tickers in the configured skip set, default `^VIX`
- Builds up to four batches from the resulting list
- Processes each batch with a pause between batches

This means batch contents track `tickers.json` and will change when the backfill universe changes.

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
- `content_summary` - article summary or excerpt (truncated to 1000 chars)
- `content_length` - word count of `title + content` before truncation (drives length weighting downstream)

## Sentiment Score Computation

### Per-article (FinBERT)

```python
score = P(positive) - P(negative)   # range [-1, +1]
```

Label index mapping comes from the fine-tuned model's `config.json` (`positive=0, neutral=1, negative=2`). The neutral probability is discarded entirely.

### Daily aggregate (length-weighted, freshness-decayed)

For each (ticker, date) the daily mean is a weighted average:

```
weight  = content_length * 0.5 ** (age_seconds / 23400)
score_d = SUM(score * weight) / SUM(weight)
```

- `content_length` = word count of `title + content` at insert time. Backfilled rows use `LENGTH(content_summary)` as a proxy.
- `age_seconds = NOW() - published_at`
- `23400 s = 6.5 h = one trading session` — the decay half-life.

The same formula is implemented in two sites:

| Output                                       | Source                                                                       |
|----------------------------------------------|------------------------------------------------------------------------------|
| `NLP/sentiment_scores/<T>_daily_scores.csv`  | `FinBertSentimentScorer._aggregate_daily_means` (pandas)                     |
| `market_data.sentiment_score`                | `NewsSentimentRepository.update_market_data_sentiment` (PostgreSQL `POWER`/`SUM`) |

The CSV side falls back to an unweighted arithmetic mean when the article-scores CSV pre-dates the (`content_length`, `published_at`) schema, so legacy data still aggregates.

### Implications of freshness decay

Because the weight depends on `NOW()`, the per-day score is **time-dependent**: an article that was the dominant signal yesterday weighs less today as it ages. `update_market_data_sentiment` therefore rewrites the full sentiment history for the ticker on every invocation. An `ABS(old - new) > 1e-4` guard suppresses no-op writes from float-precision noise but every cycle still touches every (ticker, date) row that has news.

If this cost becomes a problem at full ~1900-ticker scale, options are:
- Cap the lookback window in the SQL (`WHERE published_at >= NOW() - INTERVAL '14 days'`)
- Switch to a within-day-anchored decay (stable historical values)

## Monitoring and Logs

`NLPRunner` writes logs to `NLP/daemon.log` and reports:

- batch startup and completion
- fetch status per ticker
- sentiment processing status
- skip decisions for low-activity tickers
- memory cleanup activity
- errors and retries

## Configuration Notes

### Timing

- Live scrape cycle interval: 300 seconds
- Memory cleanup interval: every 10 cycles
- Log rotation size: 10 MB
- Live mode lookback window (FMP date filter): 7 days

### Environment

Typical runtime requirements include:

- Python dependencies from `requirements.txt`
- `src/orchestrator/backfill/tickers.json` (the active ticker universe)
- database connectivity for sentiment persistence
- `FMP_API_KEY`, `ALPHA_KEY` (and optionally `APIFY_KEY` for Truth Social)

## Legacy and Manual Workflows

Use the one-off commands above when you want to run a single pipeline step. Use `python NLP/main_NLP.py` only when you want continuous scheduled batching.

## Troubleshooting

### No tickers loaded

Confirm `src/orchestrator/backfill/tickers.json` exists, is valid JSON, and contains a non-empty list of ticker symbols.

### No new articles found

This is usually expected when the source has already been processed recently or no fresh articles exist for the selected date range.

### Sentiment step skipped

The runner skips sentiment work when recent cycles show no new articles for a ticker.

### Monitoring failures

Use `NLP/monitor_daemon.py` for synthetic checks instead of treating external service probing as part of the normal pytest suite.

## Related Docs

- [Quick NLP commands](../../NLP/README.md)
- [Test modes](../TEST_MODES.md)
