#!/usr/bin/env python3
"""
sentiment_processor.py
Extracts sentiment calculation logic from visualise_NLP.ipynb Cell 1
Processes articles and computes sentiment scores using FinBERT model.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import logging

# Add project root to path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SentimentProcessor:
    """
    Processes articles and computes sentiment scores using FinBERT model.
    Extracted from visualise_NLP.ipynb Cell 1 for production use.
    """
    
    def __init__(self, model_dir: str = "ProsusAI/finbert", chunk_size: int = 32):
        """
        Initialize the sentiment processor.
        
        Args:
            model_dir: Directory containing the FinBERT model or HuggingFace model name
            chunk_size: Batch size for processing articles
        """
        self.model_dir = model_dir
        self.chunk_size = chunk_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
        logger.info(f"Initialized SentimentProcessor with device: {self.device}")
        logger.info(f"Using model: {self.model_dir}")
    
    def load_model(self) -> None:
        """Load the FinBERT model and tokenizer."""
        try:
            logger.info(f"Loading model from {self.model_dir}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    self.model_dir, 
                    torch_dtype=(torch.float16 if self.device.type == "cuda" else torch.float32)
                )
                .to(self.device)
                .eval()
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def process_articles_from_csv(self, csv_path: str, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process articles from merged CSV file and compute sentiment scores.
        
        Args:
            csv_path: Path to the merged CSV file containing articles from all sources
            ticker: Ticker symbol for the articles
            
        Returns:
            Tuple of (article_scores_df, daily_scores_df)
        """
        try:
            # Load articles from merged CSV
            df = pd.read_csv(csv_path, parse_dates=["publishedDate"])
            logger.info(f"Loaded {len(df)} articles for {ticker} from merged CSV: {csv_path}")
            
        except FileNotFoundError:
            logger.warning(f"Article file not found for {ticker}: {csv_path}")
            return pd.DataFrame(), pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_path}: {e}")
            return pd.DataFrame(), pd.DataFrame()
        
        if df.empty:
            logger.warning(f"No articles found for {ticker}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Remove any duplicate articles that might have slipped through
        initial_count = len(df)
        df = df.drop_duplicates(subset=["publishedDate", "title"], keep="first")
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} duplicate articles for {ticker}")
        
        # Prepare data for processing
        df["date"] = df["publishedDate"].dt.date
        texts = (df["content"].fillna("") + " " + df["title"].fillna("")).tolist()
        dates = pd.to_datetime(df["date"]).tolist()
        
        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Process articles in chunks
        records = []
        logger.info(f"Processing {len(texts)} articles for {ticker} in chunks of {self.chunk_size}")
        
        for i in tqdm(range(0, len(texts), self.chunk_size), desc=f"Processing {ticker}"):
            batch_texts = texts[i : i + self.chunk_size]
            batch_dates = dates[i : i + self.chunk_size]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Run inference
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                
                # Calculate sentiment scores
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                scores = probs[:, 0] - probs[:, 2]  # Positive probability - Negative probability
                
                # Store results
                records.extend(zip(batch_dates, scores))
                
                # Cleanup GPU memory
                del inputs, logits, probs
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error processing batch {i//self.chunk_size + 1} for {ticker}: {e}")
                # Continue with next batch
                continue
        
        if not records:
            logger.warning(f"No sentiment scores computed for {ticker}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Create DataFrames
        article_scores_df = pd.DataFrame(records, columns=["date", "sentiment"])
        daily_scores_df = article_scores_df.groupby("date", as_index=False)["sentiment"].mean()
        
        logger.info(f"Computed sentiment for {ticker}: {len(article_scores_df)} article scores, {len(daily_scores_df)} daily scores")
        
        return article_scores_df, daily_scores_df
    
    def process_ticker(self, ticker: str, articles_dir: str = "NLP/articles", 
                      output_dir: str = "NLP/sentiment_scores") -> bool:
        """
        Process all articles for a specific ticker and save sentiment scores.
        
        Args:
            ticker: Ticker symbol to process
            articles_dir: Directory containing article CSV files
            output_dir: Directory to save sentiment score CSV files
            
        Returns:
            True if processing was successful, False otherwise
        """
        articles_path = Path(articles_dir) / f"{ticker}.csv"
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            # Process articles
            article_scores_df, daily_scores_df = self.process_articles_from_csv(
                str(articles_path), ticker
            )
            
            if article_scores_df.empty:
                logger.warning(f"No sentiment scores to save for {ticker}")
                return False
            
            # Save results
            article_output_path = output_path / f"{ticker}_article_scores.csv"
            daily_output_path = output_path / f"{ticker}_daily_scores.csv"
            
            article_scores_df.to_csv(article_output_path, index=False)
            daily_scores_df.to_csv(daily_output_path, index=False)
            
            logger.info(f"Saved sentiment scores for {ticker}:")
            logger.info(f"  - Article scores: {article_output_path}")
            logger.info(f"  - Daily scores: {daily_output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {e}")
            return False
    
    def process_multiple_tickers(self, tickers: List[str], articles_dir: str = "NLP/articles",
                               output_dir: str = "NLP/sentiment_scores") -> Dict[str, bool]:
        """
        Process multiple tickers and return results.
        
        Args:
            tickers: List of ticker symbols to process
            articles_dir: Directory containing article CSV files
            output_dir: Directory to save sentiment score CSV files
            
        Returns:
            Dictionary mapping ticker to success status
        """
        results = {}
        
        for ticker in tickers:
            logger.info(f"Processing ticker: {ticker}")
            results[ticker] = self.process_ticker(ticker, articles_dir, output_dir)
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        logger.info(f"Processing complete: {successful}/{total} tickers successful")
        
        return results


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process articles for sentiment analysis")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols to process")
    parser.add_argument("--articles-dir", default="NLP/articles", 
                       help="Directory containing article CSV files")
    parser.add_argument("--output-dir", default="NLP/sentiment_scores",
                       help="Directory to save sentiment score CSV files")
    parser.add_argument("--model-dir", default="ProsusAI/finbert",
                       help="Directory containing the FinBERT model or HuggingFace model name")
    parser.add_argument("--chunk-size", type=int, default=32,
                       help="Batch size for processing articles")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = SentimentProcessor(
        model_dir=args.model_dir,
        chunk_size=args.chunk_size
    )
    
    # Process tickers
    results = processor.process_multiple_tickers(
        tickers=args.tickers,
        articles_dir=args.articles_dir,
        output_dir=args.output_dir
    )
    
    # Exit with error code if any processing failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()