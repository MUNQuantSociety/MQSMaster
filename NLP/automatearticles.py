import os
import sys
import time
import logging
import pandas as pd
import subprocess
from datetime import datetime
import pytz

from common.database.MQSDBConnector import MQSDBConnector
from nlp.sentiment_model import SentimentAnalyzer  # <- add the model here (work in progress)

# --- CONFIG ---
LOG_FILE = "/var/log/sentiment_ingestor.log"
TIMEZONE = pytz.timezone("America/New_York")
FETCH_INTERVAL_SECONDS = 1800  # every 30 minutes
DB_TABLE_NAME = "news_sentiment"

# --- LOGGING ---
def setup_logging():
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s"
    )

# --- HELPERS ---
def run_fetch_articles(ticker, start_date, end_date):
    """Call fetch_articles.py and return path to CSV"""
    base_dir = os.path.join(os.path.dirname(__file__), "NLP")
    csv_path = os.path.join(base_dir, "articles", f"{ticker}.csv")

    try:
        subprocess.run(
            [sys.executable, os.path.join(base_dir, "fetch_articles.py"),
             ticker, start_date, end_date],
            check=True
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"fetch_articles.py failed: {e}")
        return None
    
    return csv_path if os.path.exists(csv_path) else None

def process_articles(csv_path, sentiment_model):
    """Load CSV and compute sentiment scores"""
    try:
        df = pd.read_csv(csv_path, parse_dates=["publishedDate"])
    except Exception as e:
        logging.error(f"Failed reading CSV: {e}")
        return []

    if df.empty:
        logging.info("No new articles found.")
        return []

    logging.info(f"Scoring {len(df)} articles with sentiment model...")
    df["sentiment"] = sentiment_model.predict(df["content"].fillna(""))

    records = df[["publishedDate", "title", "content", "site", "sentiment"]].to_dict("records")
    return records

def run_ingestion_cycle(db, sentiment_model, ticker, start_date, end_date):
    """Fetch articles, score sentiment, and insert into DB"""
    csv_path = run_fetch_articles(ticker, start_date, end_date)
    if not csv_path:
        return

    records = process_articles(csv_path, sentiment_model)
    if not records:
        return

    result = db.bulk_inject_to_db(DB_TABLE_NAME, records,
                                  conflict_columns=["publishedDate", "title"])
    if result["status"] == "success":
        logging.info(f"Inserted {len(records)} sentiment rows into {DB_TABLE_NAME}.")
    else:
        logging.error(f"DB insert failed: {result['message']}")

# --- MAIN LOOP ---
def main():
    setup_logging()
    logging.info("======= Starting Real-Time Sentiment Ingestor =======")

    db = MQSDBConnector()
    sentiment_model = SentimentAnalyzer()  # load your FinBERT wrapper

    ticker = os.getenv("TICKER", "AAPL")
    start_date = os.getenv("START_DATE", "2023-01-01")
    end_date = os.getenv("END_DATE", datetime.now().strftime("%Y-%m-%d"))

    try:
        while True:
            run_ingestion_cycle(db, sentiment_model, ticker, start_date, end_date)
            logging.info(f"Sleeping {FETCH_INTERVAL_SECONDS}s before next cycle.")
            time.sleep(FETCH_INTERVAL_SECONDS)
    except (KeyboardInterrupt, SystemExit):
        logging.info("Sentiment ingestor stopped by user.")
    finally:
        db.close_all_connections()
        logging.info("======= Sentiment Ingestor Stopped =======")

if __name__ == "__main__":
    main()
