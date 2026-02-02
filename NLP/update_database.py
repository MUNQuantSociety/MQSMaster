#!/usr/bin/env python3
"""
update_database.py
Updates the news_sentiment table with computed sentiment scores.
Integrates with the existing MQS database system.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from datetime import datetime
import logging

# Add project root to path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

try:
    from src.common.database.MQSDBConnector import MQSDBConnector
except ImportError:
    logging.error("Could not import MQSDBConnector. Make sure the database module is available.")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SentimentDatabaseUpdater:
    """
    Updates the news_sentiment table with computed sentiment scores.
    """
    
    def __init__(self):
        """Initialize the database updater."""
        self.db = MQSDBConnector()
        self._ensure_table_exists()
        self._create_indexes()
    
    def _ensure_table_exists(self) -> None:
        """Ensure the news_sentiment table exists."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10),
            article_url TEXT,
            published_at TIMESTAMP,
            sentiment_score FLOAT,
            content_summary TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        result = self.db.execute_query(create_table_sql)
        if result["status"] == "error":
            logger.error(f"Failed to create news_sentiment table: {result['message']}")
            raise Exception("Database table creation failed")
        
        logger.info("news_sentiment table verified/created")
    
    def _create_indexes(self) -> None:
        """Create indexes for efficient querying."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_news_sentiment_ticker_date ON news_sentiment(ticker, published_at);",
            "CREATE INDEX IF NOT EXISTS idx_news_sentiment_published_at ON news_sentiment(published_at);",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_news_sentiment_url ON news_sentiment(article_url);"
        ]
        
        for index_sql in indexes:
            result = self.db.execute_query(index_sql)
            if result["status"] == "error":
                logger.warning(f"Failed to create index: {result['message']}")
    
    def check_article_exists(self, article_url: str) -> bool:
        """
        Check if an article already exists in the database.
        
        Args:
            article_url: URL of the article to check
            
        Returns:
            True if article exists, False otherwise
        """
        query = "SELECT COUNT(*) as count FROM news_sentiment WHERE article_url = %s"
        result = self.db.execute_query(query, (article_url,), fetch=True)
        
        if result["status"] == "error":
            logger.error(f"Error checking article existence: {result['message']}")
            return False
        
        # With RealDictCursor, result["data"] is a list of dictionaries
        if result["data"]:
            return result["data"][0]["count"] > 0
        
        return False
    
    def insert_sentiment_record(self, ticker: str, article_url: str, published_at: datetime,
                              sentiment_score: float, content_summary: str) -> bool:
        """
        Insert a single sentiment record into the database.
        
        Args:
            ticker: Stock ticker symbol
            article_url: URL of the article
            published_at: Publication timestamp
            sentiment_score: Sentiment score (-1.0 to 1.0)
            content_summary: Summary of article content
            
        Returns:
            True if insertion was successful, False otherwise
        """
        # Validate sentiment score range
        if not -1.0 <= sentiment_score <= 1.0:
            logger.warning(f"Invalid sentiment score {sentiment_score} for {ticker}. Skipping.")
            return False
        
        # Check for duplicates
        if self.check_article_exists(article_url):
            logger.debug(f"Article already exists: {article_url}")
            return True  # Not an error, just already exists
        
        # Truncate content summary if too long
        if len(content_summary) > 1000:
            content_summary = content_summary[:1000]
        
        insert_sql = """
        INSERT INTO news_sentiment (ticker, article_url, published_at, sentiment_score, content_summary)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        try:
            result = self.db.execute_query(
                insert_sql,
                (ticker, article_url, published_at, sentiment_score, content_summary)
            )
            
            if result["status"] == "error":
                logger.error(f"Failed to insert sentiment record for {ticker}: {result['message']}")
                logger.error(f"Data: URL={article_url[:100]}..., score={sentiment_score}, date={published_at}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Exception during sentiment record insertion for {ticker}: {str(e)}")
            logger.error(f"Data: URL={article_url[:100]}..., score={sentiment_score}, date={published_at}")
            return False
    
    def process_articles_with_sentiment(self, ticker: str, articles_df: pd.DataFrame, 
                                      sentiment_df: pd.DataFrame) -> Dict[str, int]:
        """
        Process articles and their sentiment scores, inserting into database.
        
        Args:
            ticker: Stock ticker symbol
            articles_df: DataFrame with article data (publishedDate, title, content, site)
            sentiment_df: DataFrame with sentiment scores (date, sentiment)
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {"processed": 0, "inserted": 0, "skipped": 0, "errors": 0}
        
        # Validate that we have matching number of articles and sentiment scores
        if len(articles_df) != len(sentiment_df):
            logger.error(f"Mismatch: {len(articles_df)} articles vs {len(sentiment_df)} sentiment scores for {ticker}")
            return stats
        
        # Reset indices to ensure proper alignment
        articles_df = articles_df.reset_index(drop=True)
        sentiment_df = sentiment_df.reset_index(drop=True)
        
        logger.info(f"Processing {len(articles_df)} articles with sentiment for {ticker}")
        
        # Process articles and sentiment scores in parallel (1:1 correspondence)
        for i in range(len(articles_df)):
            stats["processed"] += 1
            
            try:
                article_row = articles_df.iloc[i]
                sentiment_score = sentiment_df.iloc[i]["sentiment"]
                
                # Create content summary (first 500 characters)
                content = str(article_row.get("content", ""))
                title = str(article_row.get("title", ""))
                content_summary = (title + " " + content)[:500].strip()
                
                # Insert record
                success = self.insert_sentiment_record(
                    ticker=ticker,
                    article_url=str(article_row.get("site", "")),
                    published_at=pd.to_datetime(article_row["publishedDate"]),
                    sentiment_score=float(sentiment_score),
                    content_summary=content_summary
                )
                
                if success:
                    stats["inserted"] += 1
                else:
                    stats["skipped"] += 1
                    
            except Exception as e:
                logger.error(f"Error processing article {i} for {ticker}: {str(e)}")
                stats["errors"] += 1
        
        logger.info(f"Completed {ticker}: {stats['inserted']} inserted, {stats['skipped']} skipped, {stats['errors']} errors")
        return stats
    
    def update_from_csv_files(self, ticker: str, articles_dir: str = "NLP/articles",
                            sentiment_dir: str = "NLP/sentiment_scores") -> bool:
        """
        Update database from CSV files for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            articles_dir: Directory containing article CSV files
            sentiment_dir: Directory containing sentiment CSV files
            
        Returns:
            True if update was successful, False otherwise
        """
        articles_path = Path(articles_dir) / f"{ticker}.csv"
        sentiment_path = Path(sentiment_dir) / f"{ticker}_article_scores.csv"
        
        try:
            # Load articles
            if not articles_path.exists():
                logger.warning(f"Articles file not found: {articles_path}")
                return False
            
            articles_df = pd.read_csv(articles_path, parse_dates=["publishedDate"])
            logger.info(f"Loaded {len(articles_df)} articles for {ticker}")
            
            # Load sentiment scores
            if not sentiment_path.exists():
                logger.warning(f"Sentiment file not found: {sentiment_path}")
                return False
            
            sentiment_df = pd.read_csv(sentiment_path, parse_dates=["date"])
            logger.info(f"Loaded {len(sentiment_df)} sentiment scores for {ticker}")
            
            # Process and insert
            stats = self.process_articles_with_sentiment(ticker, articles_df, sentiment_df)
            
            return stats["inserted"] > 0 or stats["skipped"] > 0
            
        except Exception as e:
            logger.error(f"Error updating database for {ticker}: {e}")
            return False
    
    def update_multiple_tickers(self, tickers: List[str], articles_dir: str = "NLP/articles",
                              sentiment_dir: str = "NLP/sentiment_scores") -> Dict[str, bool]:
        """
        Update database for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            articles_dir: Directory containing article CSV files
            sentiment_dir: Directory containing sentiment CSV files
            
        Returns:
            Dictionary mapping ticker to success status
        """
        results = {}
        total_stats = {"processed": 0, "inserted": 0, "skipped": 0, "errors": 0}
        
        for ticker in tickers:
            logger.info(f"Updating database for ticker: {ticker}")
            results[ticker] = self.update_from_csv_files(ticker, articles_dir, sentiment_dir)
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        logger.info(f"Database update complete: {successful}/{total} tickers successful")
        
        return results
    
    def get_sentiment_data(self, ticker: str, start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Retrieve sentiment data from database.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            
        Returns:
            DataFrame with sentiment data
        """
        query = "SELECT * FROM news_sentiment WHERE ticker = %s"
        params = [ticker]
        
        if start_date:
            query += " AND published_at >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND published_at <= %s"
            params.append(end_date)
        
        query += " ORDER BY published_at DESC"
        
        result = self.db.execute_query(query, params, fetch=True)
        
        if result["status"] == "error":
            logger.error(f"Error retrieving sentiment data: {result['message']}")
            return pd.DataFrame()
        
        if not result["data"]:
            return pd.DataFrame()
        
        # Convert to DataFrame
        columns = ["id", "ticker", "article_url", "published_at", "sentiment_score", "content_summary", "created_at"]
        df = pd.DataFrame(result["data"], columns=columns)
        df["published_at"] = pd.to_datetime(df["published_at"])
        df["created_at"] = pd.to_datetime(df["created_at"])
        
        return df


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update news_sentiment database table")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols to update")
    parser.add_argument("--articles-dir", default="NLP/articles",
                       help="Directory containing article CSV files")
    parser.add_argument("--sentiment-dir", default="NLP/sentiment_scores",
                       help="Directory containing sentiment CSV files")
    parser.add_argument("--query", action="store_true",
                       help="Query and display sentiment data instead of updating")
    
    args = parser.parse_args()
    
    # Initialize updater
    updater = SentimentDatabaseUpdater()
    
    if args.query:
        # Query mode
        for ticker in args.tickers:
            logger.info(f"Querying sentiment data for {ticker}")
            df = updater.get_sentiment_data(ticker)
            if not df.empty:
                print(f"\n{ticker} - {len(df)} records:")
                print(df[["published_at", "sentiment_score", "content_summary"]].head())
            else:
                print(f"\n{ticker} - No data found")
    else:
        # Update mode
        results = updater.update_multiple_tickers(
            tickers=args.tickers,
            articles_dir=args.articles_dir,
            sentiment_dir=args.sentiment_dir
        )
        
        # Exit with error code if any updates failed
        if not all(results.values()):
            sys.exit(1)


if __name__ == "__main__":
    main()