# NLP Commands

## Setup: Download the FinBERT Model

The NLP pipeline requires the fine-tuned FinBERT model. It is **not** committed to the repo and must be downloaded before running any sentiment scoring.

1. Download the model folder from this Google Drive link:
   <https://drive.google.com/drive/folders/1v7NjSuyFq4CTIctrw1bSv13JzkkMg1l8?usp=sharing>
2. Place the downloaded folder inside `NLP/` so the final path is:
   ```
   NLP/finbert-finetuned-final/
   ```
3. Confirm the directory contains the model weights, tokenizer, and config files (e.g. `config.json`, `tokenizer.json`, `model.safetensors` or equivalent).

The default model path is resolved by `NLP/sentiment/scorer.py` (re-exported from `NLP/sentiment_processor.py`) as `NLP/finbert-finetuned-final`. **There is no HuggingFace fallback**: if the directory is missing the scorer raises `FileNotFoundError` at startup with a pointer back to this README. The previous fallback (`ProsusAI/finbert`) used a different label ordering (positive=0, negative=1, neutral=2 vs. our fine-tuned positive=0, neutral=1, negative=2) and would have produced silently incorrect scores.

## Quick Commands

```bash
# Live polling loop (every 5 min, all tickers in tickers.json)
python NLP/main_NLP.py

# Historical backfill for a date range (refuses to run during market hours)
python NLP/backfill_NLP.py --start 2025-01-01 --end 2025-12-31
python NLP/backfill_NLP.py --start 2025-12-01 --end 2025-12-31 --tickers AAPL MSFT
python NLP/backfill_NLP.py --start 2025-01-01 --end 2025-12-31 --wait

# Individual subprocesses / smoke checks (back-compat shims)
python -m NLP.fetch_articles AAPL 2025-12-01 2025-12-31
python -m NLP.process_sentiment_pipeline AAPL MSFT
python -m NLP.test_pipeline
python -m NLP.update_database --query AAPL MSFT
python NLP/monitor_daemon.py --synthetic --max-log-age-hours 72
```

## Live vs Backfill

| Mode      | Entrypoint               | Cadence            | Sources per cycle                                              | Market-hours rule       |
|-----------|--------------------------|--------------------|----------------------------------------------------------------|-------------------------|
| Live      | `NLP/main_NLP.py`        | every 5 minutes    | FMP page 0 for **all** tickers; alt sources round-robin'd per batch | Always allowed         |
| Backfill  | `NLP/backfill_NLP.py`    | one-shot CLI       | FMP all pages + Yahoo + Finviz + Alpha Vantage per ticker      | **Blocked while open**  |

### Live mode design

- FinBERT loads **once** at process startup, shared across all tickers (vs. the old subprocess-per-ticker pattern that reloaded ~1900 times per cycle).
- Each cycle: every ticker gets a fast FMP page-0 check (newest articles). One of the 4 batches additionally pulls Yahoo / Finviz / Alpha Vantage that cycle, so each ticker gets a full multi-source refresh roughly every 4 cycles (~20 min).
- Articles bypassing FinBERT scoring when no new rows are added; sentiment runs only for tickers whose article CSV grew.
- `SkipStatsTracker` skips the sentiment step for tickers that have produced no new articles in the last 5 cycles (cuts wasted scoring work).

### Backfill mode design

- Iterates tickers serially. Per ticker: `ArticleAggregator.run(start, end)` (all sources, full pagination) → `SentimentPipeline.process_ticker_complete`.
- Hard guard before startup: refuses to run if `is_market_open()` returns True. Override with `--wait` to block until 16:00 ET then resume.

## Sentiment Score: How it's Computed

**Per-article (FinBERT)**: `score = P(positive) - P(negative)`, range `[-1, +1]`. Neutral probability is discarded. Label mapping is read from the fine-tuned model's config (`positive=0, neutral=1, negative=2`).

**Daily aggregate**: length-weighted, freshness-decayed mean of per-article scores. Per-article weight:

```
weight = content_length * 0.5 ^ (age_seconds / 23400)
```

where `content_length` is the word count of `title + content` (stored on `news_sentiment` at insert time, backfilled with `LENGTH(content_summary)` for legacy rows) and `23400 = 6.5h = one trading session`. The same formula runs in two places:

| Output                                  | Site                                              |
|-----------------------------------------|---------------------------------------------------|
| `NLP/sentiment_scores/<T>_daily_scores.csv` | `FinBertSentimentScorer._aggregate_daily_means`  |
| `market_data.sentiment_score`           | `NewsSentimentRepository.update_market_data_sentiment` (SQL) |

**Important implication — freshness decay vs. NOW()**: because the decay uses the current clock, *historical days' scores shift every cycle as older articles' weights drop*. `update_market_data_sentiment` rewrites the full sentiment history for the ticker on each invocation. A no-op-write guard (`ABS(diff) > 1e-4`) suppresses float-precision-only updates.

## Rate Limit / Budget

FMP enforces a 3000 req/min account cap. The two NLP+market processes split it statically:

| Process              | Cap         | Source                                  |
|----------------------|-------------|-----------------------------------------|
| Realtime ingestor    | 1000/min    | `FMPRateLimiter.for_realtime()`         |
| NLP (live + backfill)| 2000/min    | `FMPRateLimiter.for_nlp()`              |

Realtime ingestor currently consumes ~1 req/min (one batch-quote per minute), so the 1000 cap is effectively reserved headroom. NLP's 2000/min comfortably covers the 1919-ticker live sweep (≈384 req/min).

Limiter lives at `src/common/fmp_rate_limiter.py`. `ArticlesGateway` and `FMPMarketData` both acquire from the appropriate per-process singleton before each HTTP call.

## Ticker Universe

The pipeline scrapes the same universe the backfill pipeline ingests:
`src/orchestrator/backfill/tickers.json`. `^VIX` is excluded by default.
Override by passing a different path to `TickerUniverse(tickers_path=...)`.

## Package Layout

```
NLP/
├── main_NLP.py                # entrypoint - live pipeline runner, mirrors src/main.py
├── backfill_NLP.py            # entrypoint - historical date-range backfill
├── runner.py                  # NLPRunner + TickerProcessor + SkipStatsTracker
├── monitor_daemon.py          # entrypoint - runner health / synthetic checks
├── fetch_articles.py          # back-compat shim -> scrapers.fmp + scrapers.aggregator
├── fetch_alt_articles.py      # back-compat shim -> ArticleScraper facade over scrapers/
├── sentiment_processor.py     # back-compat shim -> sentiment.scorer
├── process_sentiment_pipeline.py # back-compat shim -> sentiment.pipeline
├── update_database.py         # back-compat shim -> persistence.repository
├── test_pipeline.py           # back-compat smoke entrypoint
├── core/                      # shared helpers (paths, timestamps, logging)
├── scrapers/                  # one class per news source + ArticleAggregator
│   ├── base.py                # BaseNewsScraper ABC + ArticleRecord schema
│   ├── fmp.py                 # FmpNewsScraper (+ fetch_latest_and_save), FmpFetchStateStore
│   ├── yahoo.py               # YahooNewsScraper (yfinance)
│   ├── finviz.py              # FinvizNewsScraper (async aiohttp)
│   ├── alpha_vantage.py       # AlphaVantageNewsScraper
│   ├── truth_social.py        # TruthSocialScraper (Apify)
│   └── aggregator.py          # ArticleAggregator (.run / .run_fmp_only)
├── sentiment/                 # FinBERT scoring + end-to-end pipeline
│   ├── scorer.py              # FinBertSentimentScorer (legacy alias: SentimentProcessor)
│   └── pipeline.py            # SentimentPipeline
├── persistence/
│   └── repository.py          # NewsSentimentRepository
├── orchestration/
│   └── ticker_universe.py     # TickerUniverse (load tickers.json + batch)
├── articles/                  # per-ticker article CSVs (gitignored)
├── sentiment_scores/          # per-ticker score CSVs (gitignored)
├── fetch_state/               # FMP paged-fetch cursors (gitignored)
└── finbert-finetuned-final/   # downloaded model directory (gitignored)
```

Related shared modules:

- `src/common/market_hours.py` — `is_market_open()`, `seconds_until_market_close()`
- `src/common/fmp_rate_limiter.py` — process-singleton rate limiter (`for_realtime` / `for_nlp`)

## More Details

See [docs/NLP/README.md](../docs/NLP/README.md) for the full NLP pipeline guide, runtime behavior, data layout, and monitoring notes.
