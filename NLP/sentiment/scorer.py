"""FinBERT sentiment scorer.

Loads a FinBERT (or compatible HuggingFace) model and computes a per-article
sentiment score in ``[-1, 1]`` (positive prob - negative prob). Designed
for batched CSV-driven processing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from NLP.core import (
    ARTICLES_DIR,
    MODEL_DIR,
    SCORES_DIR,
    ensure_project_root_on_path,
    get_logger,
)

ensure_project_root_on_path()

logger = get_logger(__name__)

DEFAULT_LOCAL_MODEL_DIR = str(MODEL_DIR)
HUGGINGFACE_FALLBACK = "ProsusAI/finbert"


class FinBertSentimentScorer:
    """Score articles with a FinBERT model.

    Picks a local safetensors checkpoint when available (avoids the
    ``torch.load()`` vulnerability in ``torch < 2.6``) and falls back to
    the HuggingFace hub.
    """

    DEFAULT_CHUNK_SIZE: int = 32

    def __init__(
        self,
        model_dir: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        self.model_dir = self._resolve_model_dir(model_dir)
        self.chunk_size = chunk_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

        logger.info(f"Initialized {self.__class__.__name__} with device: {self.device}")
        logger.info(f"Using model: {self.model_dir}")

    @staticmethod
    def _resolve_model_dir(explicit: Optional[str]) -> str:
        if explicit is not None:
            return explicit
        if os.path.exists(DEFAULT_LOCAL_MODEL_DIR):
            logger.info(f"Using local safetensors model: {DEFAULT_LOCAL_MODEL_DIR}")
            return DEFAULT_LOCAL_MODEL_DIR
        logger.warning(
            f"Local model not found at {DEFAULT_LOCAL_MODEL_DIR}, "
            f"falling back to HuggingFace: {HUGGINGFACE_FALLBACK}"
        )
        return HUGGINGFACE_FALLBACK

    def load_model(self) -> None:
        """Load the FinBERT model and tokenizer using safetensors when available."""
        try:
            logger.info(f"Loading model from {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(
                    self.model_dir,
                    trust_remote_code=False,
                    torch_dtype=(
                        torch.float16 if self.device.type == "cuda" else torch.float32
                    ),
                )
                .to(self.device)
                .eval()
            )
            logger.info("Model loaded successfully")
        except Exception as exc:
            logger.error(f"Failed to load model from {self.model_dir}: {exc}")
            raise

    def _ensure_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            self.load_model()

    @staticmethod
    def _aggregate_daily_means(article_scores_df: pd.DataFrame) -> pd.DataFrame:
        """Group per-article scores into a (date, sentiment) daily mean frame.

        The grouper is materialized as a real ``date`` column before the
        ``groupby`` so pandas keeps it in the result. Using an external
        Series grouper drops the column under pandas >= 2.2 and emits a
        FutureWarning.
        """
        df = article_scores_df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.groupby("date", as_index=False)["sentiment"].mean()

    def _score_batch(self, texts: List[str]) -> List[float]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        # positive prob - negative prob, clipped naturally to [-1, 1].
        scores = (probs[:, 0] - probs[:, 2]).tolist()

        del inputs, logits, probs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return scores

    def process_articles_from_csv(
        self, csv_path: str, ticker: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Score every article in ``csv_path`` for ``ticker``.

        Returns ``(article_scores_df, daily_scores_df)`` where
        ``article_scores_df`` carries per-article rows and
        ``daily_scores_df`` is the daily mean.
        """
        try:
            df = pd.read_csv(csv_path, parse_dates=["publishedDate"])
            logger.info(
                f"Loaded {len(df)} articles for {ticker} from merged CSV: {csv_path}"
            )
        except FileNotFoundError:
            logger.warning(f"Article file not found for {ticker}: {csv_path}")
            return pd.DataFrame(), pd.DataFrame()
        except Exception as exc:
            logger.error(f"Error loading CSV file {csv_path}: {exc}")
            return pd.DataFrame(), pd.DataFrame()

        if df.empty:
            logger.warning(f"No articles found for {ticker}")
            return pd.DataFrame(), pd.DataFrame()

        initial_count = len(df)
        df = df.drop_duplicates(subset=["publishedDate", "title"], keep="first")
        if len(df) < initial_count:
            logger.info(
                f"Removed {initial_count - len(df)} duplicate articles for {ticker}"
            )

        scores_path = os.path.normpath(
            csv_path.replace(
                os.path.basename(csv_path),
                f"../sentiment_scores/{ticker}_article_scores.csv",
            )
        )
        existing_scores_df = pd.DataFrame()
        if os.path.exists(scores_path):
            try:
                existing_scores_df = pd.read_csv(scores_path, parse_dates=["date"])
                already_scored = len(existing_scores_df)
                df = df.iloc[already_scored:]
                logger.info(
                    f"Skipping {already_scored} already-scored articles for {ticker}, "
                    f"{len(df)} new to process"
                )
            except Exception:
                pass

        if df.empty:
            logger.info(f"No new articles to score for {ticker}")
            if not existing_scores_df.empty:
                daily_scores_df = self._aggregate_daily_means(existing_scores_df)
                return existing_scores_df, daily_scores_df
            return pd.DataFrame(), pd.DataFrame()

        df["date"] = df["publishedDate"].dt.date
        texts = (df["content"].fillna("") + " " + df["title"].fillna("")).tolist()
        dates = pd.to_datetime(df["date"]).tolist()

        self._ensure_loaded()

        records: List[Tuple] = []
        logger.info(
            f"Processing {len(texts)} articles for {ticker} in chunks of {self.chunk_size}"
        )

        for i in tqdm(
            range(0, len(texts), self.chunk_size), desc=f"Processing {ticker}"
        ):
            batch_texts = texts[i : i + self.chunk_size]
            batch_dates = dates[i : i + self.chunk_size]
            try:
                scores = self._score_batch(batch_texts)
                records.extend(zip(batch_dates, scores))
            except Exception as exc:
                logger.error(
                    f"Error processing batch {i // self.chunk_size + 1} for {ticker}: {exc}"
                )
                continue

        if not records:
            logger.warning(f"No sentiment scores computed for {ticker}")
            return pd.DataFrame(), pd.DataFrame()

        new_scores_df = pd.DataFrame(records, columns=["date", "sentiment"])

        if not existing_scores_df.empty:
            article_scores_df = pd.concat(
                [existing_scores_df, new_scores_df], ignore_index=True
            )
        else:
            article_scores_df = new_scores_df

        daily_scores_df = self._aggregate_daily_means(article_scores_df)

        logger.info(
            f"Computed sentiment for {ticker}: {len(new_scores_df)} new article scores, "
            f"{len(daily_scores_df)} daily scores total"
        )
        return article_scores_df, daily_scores_df

    def process_ticker(
        self,
        ticker: str,
        articles_dir: str = str(ARTICLES_DIR),
        output_dir: str = str(SCORES_DIR),
    ) -> bool:
        """Score every article CSV for ``ticker`` and persist results."""
        articles_path = Path(articles_dir) / f"{ticker}.csv"
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        try:
            article_scores_df, daily_scores_df = self.process_articles_from_csv(
                str(articles_path), ticker
            )
        except Exception as exc:
            logger.error(f"Error processing ticker {ticker}: {exc}")
            return False

        if article_scores_df.empty:
            logger.warning(f"No sentiment scores to save for {ticker}")
            return False

        article_output_path = output_path / f"{ticker}_article_scores.csv"
        daily_output_path = output_path / f"{ticker}_daily_scores.csv"

        article_scores_df.to_csv(article_output_path, index=False)
        daily_scores_df.to_csv(daily_output_path, index=False)

        logger.info(f"Saved sentiment scores for {ticker}:")
        logger.info(f"  - Article scores: {article_output_path}")
        logger.info(f"  - Daily scores: {daily_output_path}")
        return True

    def compute_average_sentiment(
        self,
        ticker: str,
        start_date,
        end_date,
        scores_dir: str = str(SCORES_DIR),
    ) -> Optional[float]:
        """Average daily sentiment for ``ticker`` over ``[start_date, end_date]``."""
        daily_path = Path(scores_dir) / f"{ticker}_daily_scores.csv"
        if not daily_path.exists():
            logger.warning(f"No daily scores file found for {ticker}: {daily_path}")
            return None

        try:
            df = pd.read_csv(daily_path, parse_dates=["date"])
        except Exception as exc:
            logger.error(f"Failed to read daily scores for {ticker}: {exc}")
            return None

        start = pd.to_datetime(start_date, utc=True).normalize()
        end = pd.to_datetime(end_date, utc=True).normalize()

        mask = (df["date"] >= start) & (df["date"] <= end)
        window = df.loc[mask, "sentiment"]

        if window.empty:
            logger.info(
                f"No sentiment data for {ticker} between {start.date()} and {end.date()}"
            )
            return None

        avg = float(window.mean())
        logger.info(
            f"Average sentiment for {ticker} [{start.date()} -> {end.date()}]: "
            f"{avg:.4f} ({len(window)} days)"
        )
        return avg

    def process_multiple_tickers(
        self,
        tickers: List[str],
        articles_dir: str = str(ARTICLES_DIR),
        output_dir: str = str(SCORES_DIR),
    ) -> Dict[str, bool]:
        """Run :meth:`process_ticker` for each ticker. Returns a status map."""
        results: Dict[str, bool] = {}
        for ticker in tickers:
            logger.info(f"Processing ticker: {ticker}")
            results[ticker] = self.process_ticker(ticker, articles_dir, output_dir)

        successful = sum(1 for ok in results.values() if ok)
        total = len(results)
        logger.info(f"Processing complete: {successful}/{total} tickers successful")
        return results
