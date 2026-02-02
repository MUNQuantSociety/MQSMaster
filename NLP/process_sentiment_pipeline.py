#!/usr/bin/env python3
"""
process_sentiment_pipeline.py
Integrated pipeline that processes articles for sentiment and updates the database.
This script is called by the scraper daemon after articles are fetched.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import logging

# Add project root to path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from NLP.sentiment_processor import SentimentProcessor
from NLP.update_database import SentimentDatabaseUpdater

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SentimentPipeline:
    """
    Integrated pipeline for sentiment processing and database updates.
    """
    
    def __init__(self, model_dir: str = "ProsusAI/finbert", chunk_size: int = 32):
        """
        Initialize the sentiment pipeline.
        
        Args:
            model_dir: Directory containing the FinBERT model or HuggingFace model name
            chunk_size: Batch size for processing articles
        """
        self.processor = SentimentProcessor(model_dir=model_dir, chunk_size=chunk_size)
        self.db_updater = SentimentDatabaseUpdater()
        
        logger.info("Initialized SentimentPipeline")
    
    def process_ticker_complete(self, ticker: str, articles_dir: str = "NLP/articles",
                              sentiment_dir: str = "NLP/sentiment_scores") -> bool:
        """
        Complete processing pipeline for a single ticker:
        1. Process articles for sentiment
        2. Update database with results
        
        Args:
            ticker: Stock ticker symbol
            articles_dir: Directory containing article CSV files
            sentiment_dir: Directory to save sentiment CSV files
            
        Returns:
            True if processing was successful, False otherwise
        """
        logger.info(f"Starting complete processing pipeline for {ticker}")
        
        try:
            # Step 1: Process articles for sentiment
            logger.info(f"Step 1: Processing sentiment for {ticker}")
            sentiment_success = self.processor.process_ticker(
                ticker=ticker,
                articles_dir=articles_dir,
                output_dir=sentiment_dir
            )
            
            if not sentiment_success:
                logger.error(f"Sentiment processing failed for {ticker}")
                return False
            
            # Step 2: Update database
            logger.info(f"Step 2: Updating database for {ticker}")
            db_success = self.db_updater.update_from_csv_files(
                ticker=ticker,
                articles_dir=articles_dir,
                sentiment_dir=sentiment_dir
            )
            
            if not db_success:
                logger.error(f"Database update failed for {ticker}")
                return False
            
            logger.info(f"Complete processing pipeline successful for {ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Error in complete processing pipeline for {ticker}: {e}")
            return False
    
    def process_multiple_tickers_complete(self, tickers: List[str], 
                                        articles_dir: str = "NLP/articles",
                                        sentiment_dir: str = "NLP/sentiment_scores") -> Dict[str, bool]:
        """
        Complete processing pipeline for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            articles_dir: Directory containing article CSV files
            sentiment_dir: Directory to save sentiment CSV files
            
        Returns:
            Dictionary mapping ticker to success status
        """
        results = {}
        
        for ticker in tickers:
            results[ticker] = self.process_ticker_complete(
                ticker=ticker,
                articles_dir=articles_dir,
                sentiment_dir=sentiment_dir
            )
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        logger.info(f"Pipeline processing complete: {successful}/{total} tickers successful")
        
        return results


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run complete sentiment processing pipeline")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols to process")
    parser.add_argument("--articles-dir", default="NLP/articles",
                       help="Directory containing article CSV files")
    parser.add_argument("--sentiment-dir", default="NLP/sentiment_scores",
                       help="Directory to save sentiment CSV files")
    parser.add_argument("--model-dir", default="ProsusAI/finbert",
                       help="Directory containing the FinBERT model or HuggingFace model name")
    parser.add_argument("--chunk-size", type=int, default=32,
                       help="Batch size for processing articles")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SentimentPipeline(
        model_dir=args.model_dir,
        chunk_size=args.chunk_size
    )
    
    # Process tickers
    results = pipeline.process_multiple_tickers_complete(
        tickers=args.tickers,
        articles_dir=args.articles_dir,
        sentiment_dir=args.sentiment_dir
    )
    
    # Exit with error code if any processing failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()