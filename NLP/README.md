# NLP Sentiment Analysis Pipeline

This directory contains the NLP sentiment analysis pipeline that scrapes financial news articles, processes them for sentiment using FinBERT, and stores the results in the database.

## System Overview

The system now runs every **5 minutes** with the following improvements:

### 1. Batched Processing
- **Batch 1**: AAPL, MSFT, GOOGL
- **Batch 2**: AMZN, TSLA  
- **Batch 3**: NVDA, AMD
- **2-minute intervals** between batches to avoid rate limiting

### 2. Integrated Pipeline
Each ticker goes through:
1. **Article Fetching**: `fetch_articles.py` scrapes from multiple sources and merges into single CSV
2. **Sentiment Processing**: FinBERT model computes sentiment scores
3. **Database Storage**: Results stored in `news_sentiment` table

### 3. Consolidated Data Storage
- **Single CSV per ticker**: All sources (FMP, Yahoo, Finviz, Alpha Vantage) merged into `TICKER.csv`
- **No separate files**: Eliminates multiple CSV files per ticker
- **Automatic deduplication**: Removes duplicate articles across sources

### 4. Database Integration
- Sentiment scores automatically stored in PostgreSQL
- Real-time access for trading strategies
- Duplicate prevention and data validation

## Key Files

### Core Pipeline
- `scraper_daemon.sh` - Main daemon script (runs every 5 minutes)
- `fetch_articles.py` - Article scraping from multiple sources, merged into single CSV per ticker
- `sentiment_processor.py` - FinBERT sentiment analysis
- `update_database.py` - Database integration
- `process_sentiment_pipeline.py` - Integrated pipeline orchestrator

### Utilities
- `cleanup_csv_files.py` - Merge existing separate CSV files into single files per ticker
- `test_pipeline.py` - End-to-end pipeline testing

## Usage

### Start the Daemon
```bash
# Start the scraper daemon
./scraper_daemon.sh start

# Monitor with auto-restart
./scraper_daemon.sh monitor

# Check status
./scraper_daemon.sh status

# Stop the daemon
./scraper_daemon.sh stop
```

### Manual Processing
```bash
# Process specific tickers
python -m NLP.process_sentiment_pipeline AAPL MSFT

# Test the pipeline
python -m NLP.test_pipeline

# Query database
python -m NLP.update_database --query AAPL MSFT

# Cleanup existing separate CSV files (merge into single files)
python -m NLP.cleanup_csv_files

# Cleanup specific tickers only
python -m NLP.cleanup_csv_files --tickers AAPL MSFT

# Force remove all separate files (if merging fails)
python -m NLP.cleanup_csv_files --force
```

## Database Schema

The `news_sentiment` table stores:
- `ticker` - Stock symbol
- `article_url` - Unique article identifier
- `published_at` - Article publication timestamp
- `sentiment_score` - FinBERT score (-1.0 to 1.0)
- `content_summary` - Article summary (first 500 chars)

## System Flow

```
Article Scraping (fetch_articles.py)
         ↓
   Merge All Sources into Single CSV per Ticker
         ↓
Sentiment Analysis (sentiment_processor.py)
         ↓
Database Storage (update_database.py)
         ↓
Trading Strategy Access
```

## File Structure

### Articles Directory (`NLP/articles/`)
- `AAPL.csv` - All Apple articles from all sources
- `MSFT.csv` - All Microsoft articles from all sources  
- `GOOGL.csv` - All Google articles from all sources
- etc.

### Sentiment Scores Directory (`NLP/sentiment_scores/`)
- `AAPL_article_scores.csv` - Per-article sentiment scores
- `AAPL_daily_scores.csv` - Daily averaged sentiment scores
- etc.

## Timing Configuration

- **Main Loop**: Every 5 minutes
- **Batch 1**: AAPL, MSFT, GOOGL (starts immediately)
- **Batch 2**: AMZN, TSLA (starts after 2 minutes)
- **Batch 3**: NVDA, AMD (starts after 4 minutes)
- **Next Cycle**: Starts at 5-minute mark

## Monitoring

Logs are written to `scraper.log` with:
- Article fetching status
- Sentiment processing progress
- Database update results
- Error handling and retries

## Dependencies

- **FinBERT Model**: `finbert-combined-final/` directory
- **Database**: PostgreSQL with `news_sentiment` table
- **Python Libraries**: transformers, torch, pandas, tqdm
- **System**: CUDA support recommended for faster processing

---

## Legacy Manual Usage

### Fetching Articles Manually

```bash
python -m NLP.fetch_articles AAPL 2025-12-01 2025-12-31
```

### Adding Models

1. Download the required model `.zip` file from [this link](https://drive.google.com/drive/u/4/folders/1v7NjSuyFq4CTIctrw1bSv13JzkkMg1l8).
2. Extract the contents into the `NLP/models` folder.

> Contact one of the DataInfra Members if you are unable to access the link.

### Manual Notebook Execution

1. Open `visualise_NLP.ipynb` in a Jupyter Notebook environment.
2. Run each cell in order to process the data and visualize the results.

> **Note**: The automated pipeline now handles sentiment processing. The notebook is primarily for visualization and analysis.
