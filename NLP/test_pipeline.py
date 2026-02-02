#!/usr/bin/env python3
"""
test_pipeline.py
Test script to verify the sentiment processing pipeline works correctly.
"""

import os
import sys
from pathlib import Path
import logging

# Add project root to path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from NLP.process_sentiment_pipeline import SentimentPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_pipeline():
    """Test the sentiment processing pipeline with a single ticker."""
    
    # Test configuration
    test_ticker = "AAPL"
    articles_dir = "NLP/articles"
    sentiment_dir = "NLP/sentiment_scores"
    
    logger.info("Starting pipeline test")
    
    # Check if test data exists
    articles_path = Path(articles_dir) / f"{test_ticker}.csv"
    if not articles_path.exists():
        logger.error(f"Test articles file not found: {articles_path}")
        logger.info("Please run the article scraper first to generate test data")
        return False
    
    try:
        # Initialize pipeline
        pipeline = SentimentPipeline()
        
        # Test processing
        logger.info(f"Testing pipeline with {test_ticker}")
        success = pipeline.process_ticker_complete(
            ticker=test_ticker,
            articles_dir=articles_dir,
            sentiment_dir=sentiment_dir
        )
        
        if success:
            logger.info("Pipeline test PASSED")
            
            # Verify outputs
            sentiment_path = Path(sentiment_dir) / f"{test_ticker}_article_scores.csv"
            daily_path = Path(sentiment_dir) / f"{test_ticker}_daily_scores.csv"
            
            if sentiment_path.exists() and daily_path.exists():
                logger.info("Sentiment CSV files created successfully")
                
                # Test database query
                db_data = pipeline.db_updater.get_sentiment_data(test_ticker)
                if not db_data.empty:
                    logger.info(f"Database contains {len(db_data)} sentiment records for {test_ticker}")
                    logger.info("Database integration test PASSED")
                else:
                    logger.warning("No data found in database")
                    
            return True
        else:
            logger.error("Pipeline test FAILED")
            return False
            
    except Exception as e:
        logger.error(f"Pipeline test error: {e}")
        return False


def main():
    """Main test function."""
    success = test_pipeline()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()