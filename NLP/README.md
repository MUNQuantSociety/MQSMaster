# NLP Commands

## Setup: Download the FinBERT Model

The NLP pipeline requires the fine-tuned FinBERT model. It is **not** committed to the repo and must be downloaded before running any sentiment scoring.

1. Download the model folder from this Google Drive link:
   <https://drive.google.com/drive/folders/1v7NjSuyFq4CTIctrw1bSv13JzkkMg1l8?usp=sharing>
2. Place the downloaded folder inside `NLP/` so the final path is:
   ```
   NLP/finbert-combined-final/
   ```
3. Confirm the directory contains the model weights, tokenizer, and config files (e.g. `config.json`, `tokenizer.json`, `model.safetensors` or equivalent).

The default model path is resolved by `NLP/sentiment/scorer.py` (re-exported from `NLP/sentiment_processor.py`) as `NLP/finbert-combined-final`. Without this folder, the pipeline falls back to `ProsusAI/finbert` from HuggingFace, which is **not** the fine-tuned model used in production.

## Quick Commands

```bash
python NLP/daemon.py start
python -m NLP.fetch_articles AAPL 2025-12-01 2025-12-31
python -m NLP.process_sentiment_pipeline AAPL MSFT
python -m NLP.test_pipeline
python -m NLP.update_database --query AAPL MSFT
python NLP/monitor_daemon.py --synthetic --max-log-age-hours 72
```

> The `python -m NLP.<module>` paths above are kept stable for the daemon's
> `subprocess.run(...)` calls and for any external automation. Each one is a
> thin back-compat shim that re-exports the canonical implementation from a
> subpackage (see [Package Layout](#package-layout) below).

## Package Layout

```
NLP/
├── daemon.py                  # entrypoint - long-running scraper + sentiment loop
├── monitor_daemon.py          # entrypoint - daemon health / synthetic checks
├── fetch_articles.py          # back-compat shim -> scrapers.fmp + scrapers.aggregator
├── fetch_alt_articles.py      # back-compat shim -> ArticleScraper facade over scrapers/
├── sentiment_processor.py     # back-compat shim -> sentiment.scorer
├── process_sentiment_pipeline.py # back-compat shim -> sentiment.pipeline
├── update_database.py         # back-compat shim -> persistence.repository
├── test_pipeline.py           # back-compat smoke entrypoint
├── core/                      # shared helpers (paths, timestamps, logging)
├── scrapers/                  # one class per news source + ArticleAggregator
│   ├── base.py                # BaseNewsScraper ABC + ArticleRecord schema
│   ├── fmp.py                 # FmpNewsScraper, FmpFetchStateStore
│   ├── yahoo.py               # YahooNewsScraper (yfinance)
│   ├── finviz.py              # FinvizNewsScraper (async aiohttp)
│   ├── alpha_vantage.py       # AlphaVantageNewsScraper
│   ├── truth_social.py        # TruthSocialScraper (Apify)
│   └── aggregator.py          # ArticleAggregator (merges + dedupes sources)
├── sentiment/                 # FinBERT scoring + end-to-end pipeline
│   ├── scorer.py              # FinBertSentimentScorer (legacy alias: SentimentProcessor)
│   └── pipeline.py            # SentimentPipeline
├── persistence/
│   └── repository.py          # NewsSentimentRepository (legacy alias: SentimentDatabaseUpdater)
├── orchestration/
│   └── ticker_universe.py     # TickerUniverse (load + batch portfolio tickers)
├── articles/                  # per-ticker article CSVs (gitignored)
├── sentiment_scores/          # per-ticker score CSVs (gitignored)
├── fetch_state/               # FMP paged-fetch cursors (gitignored)
└── finbert-combined-final/    # downloaded model directory (gitignored)
```

## More Details

See [docs/NLP/README.md](../docs/NLP/README.md) for the full NLP pipeline guide, runtime behavior, data layout, and monitoring notes.
