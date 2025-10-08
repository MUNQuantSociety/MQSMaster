# agenticwrapper.py
# Inference-only wrapper for the NLP workflow:
# - TextBlob baseline (polarity/subjectivity)
# - Agent model (Hugging Face checkpoint) -> sentiment_score in [-1, 1], sentiment_label ∈ {negative, neutral, positive}
# - Robust CSV loader that maps fetch_articles.py outputs to {date, title, content}
# - Flexible daily aggregation for either engine

from __future__ import annotations

import os
from typing import Optional, List, Dict, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Helpers (numeric + mapping)
# -----------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def _probs_to_score_label(p: np.ndarray, id2label: Dict[int, str]) -> Tuple[float, str]:
    """
    Map class probabilities to a continuous score and a hard label.
    Robust to {id: name} order; expects names like 'negative','neutral','positive'.
    score ∈ [-1,1] emphasizes non-neutral mass:
        score = (P(pos) - P(neg)) * (1 - P(neu))
    """
    by = {id2label[i].lower(): float(p[i]) for i in range(len(p))}
    p_pos, p_neg, p_neu = by.get("positive", 0.0), by.get("negative", 0.0), by.get("neutral", 0.0)
    score = (p_pos - p_neg) * (1.0 - p_neu)
    label = id2label[int(np.argmax(p))].lower()
    return float(score), label

def _device_select(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Agent loader + batched scorer
# -----------------------------
def load_agent(model_dir: str, device: Optional[str] = None):
    """
    Load a HF text classifier checkpoint (e.g. FinBERT fine-tune, or your agent-trained dir).
    Returns (tokenizer, model, device_str, id2label_dict).
    """
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    dev = _device_select(device)
    mdl.to(dev).eval()
    id2label = {int(k): v for k, v in getattr(mdl.config, "id2label", {}).items()} or {
        0: "negative", 1: "neutral", 2: "positive"
    }
    return tok, mdl, dev, id2label

@torch.inference_mode()
def score_batch_with_agent(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    id2label: Dict[int, str],
    max_length: int = 256,
    batch_size: int = 32
) -> Tuple[List[float], List[str]]:
    scores: List[float] = []
    labels: List[str] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(
            chunk, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.detach().cpu().numpy()
        probs = _softmax(logits)
        for p in probs:
            s, lb = _probs_to_score_label(p, id2label)
            scores.append(s)
            labels.append(lb)
    return scores, labels

# -----------------------------
# CSV normalization utilities
# -----------------------------
def normalize_articles_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Map various possible fetch formats to a clean schema: {date, title, content}.
    - date: uses 'publishedDate' if present, else 'date', else today()
    - content: prefer 'text', else 'content', else empty
    - title: optional ('' if missing)
    Drops rows with empty/whitespace-only content.
    """
    out = pd.DataFrame()

    # date
    if "publishedDate" in df_raw.columns:
        out["date"] = pd.to_datetime(df_raw["publishedDate"]).dt.date.astype(str)
    elif "date" in df_raw.columns:
        out["date"] = pd.to_datetime(df_raw["date"]).dt.date.astype(str)
    else:
        out["date"] = pd.Series([datetime.now().date().isoformat()] * len(df_raw))

    # title
    out["title"] = df_raw["title"].astype(str) if "title" in df_raw.columns else ""

    # content
    if "text" in df_raw.columns:
        out["content"] = df_raw["text"].astype(str)
    elif "content" in df_raw.columns:
        out["content"] = df_raw["content"].astype(str)
    else:
        out["content"] = ""

    # keep only non-empty content
    out = out[out["content"].str.strip().ne("")].reset_index(drop=True)
    return out

def to_finbert_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional helper for A/B with FinBERT outputs:
    - If TextBlob polarity exists, add 'sentiment_score' = polarity
    - Optionally add 'confidence' proxy from subjectivity (1 - subjectivity)
    """
    out = df.copy()
    if "polarity" in out.columns and "sentiment_score" not in out.columns:
        out["sentiment_score"] = out["polarity"].astype(float)
    if "subjectivity" in out.columns and "confidence" not in out.columns:
        out["confidence"] = (1.0 - out["subjectivity"].astype(float)).clip(0.0, 1.0)
    return out

# -----------------------------
# Main wrapper class
# -----------------------------
class LocalAgenticNewsAI:
    """
    Lightweight wrapper used by notebooks:
      - load/hold articles in self.memory (columns: date, title, content, +scored cols)
      - analyze via TextBlob (baseline) OR agent checkpoint (HF classifier)
      - summarize to daily aggregates compatible with viz/backtests
    """
    def __init__(self, news_csv_path: Optional[str] = None):
        self.memory = pd.DataFrame(columns=["date", "title", "content"])
        self.news_csv_path = news_csv_path
        if news_csv_path and os.path.exists(news_csv_path):
            self.load_articles(news_csv_path)

    # -----------------------------
    # Memory / IO
    # -----------------------------
    def load_articles(self, path: str):
        raw = pd.read_csv(path)
        self.memory = normalize_articles_df(raw)
        print(f"[AGENT] Loaded {len(self.memory)} normalized articles from {path}")

    def add_article(self, title: str, content: str, date: Optional[str] = None):
        date = date or datetime.now().strftime("%Y-%m-%d")
        new = pd.DataFrame([{"date": date, "title": title, "content": content}])
        self.memory = pd.concat([self.memory, new], ignore_index=True)
        print(f"[AGENT] Added article: {title[:64]}...")

    def save_memory(self, path: str):
        self.memory.to_csv(path, index=False)
        print(f"[AGENT] Memory saved to {path}")

    # -----------------------------
    # Sentiment: TextBlob baseline
    # -----------------------------
    def analyze_sentiment(self):
        if self.memory.empty:
            print("[AGENT] No articles to analyze.")
            return
        self.memory["polarity"] = self.memory["content"].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
        self.memory["subjectivity"] = self.memory["content"].apply(
            lambda x: TextBlob(str(x)).sentiment.subjectivity
        )
        print("[AGENT] TextBlob sentiment analysis completed.")

    # -----------------------------
    # Sentiment: Agent checkpoint
    # -----------------------------
    def analyze_sentiment_agent(
        self,
        model_dir: str,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 256,
    ):
        """
        Score articles with a HF classifier loaded from model_dir.
        Adds columns:
          - sentiment_score ∈ [-1, 1]
          - sentiment_label ∈ {'negative','neutral','positive'}
        """
        if self.memory.empty:
            print("[AGENT] No articles to analyze.")
            return
        tok, mdl, dev, id2label = load_agent(model_dir, device)
        texts = self.memory["content"].astype(str).tolist()
        scores, labels = score_batch_with_agent(
            texts, tok, mdl, dev, id2label,
            max_length=max_length, batch_size=batch_size
        )
        self.memory["sentiment_score"] = scores
        self.memory["sentiment_label"] = labels
        print(f"[AGENT] Agent sentiment analysis completed using {model_dir} on {dev}.")

    # -----------------------------
    # Daily aggregation
    # -----------------------------
    def summarize_daily(self) -> pd.DataFrame:
        """
        Flexible daily summary:
          - If agent columns present: mean_score, n_articles, pos_share
          - Else (TextBlob): mean polarity/subjectivity + n_articles
        """
        if self.memory.empty:
            return pd.DataFrame()

        df = self.memory.copy()
        if "sentiment_score" in df.columns:
            daily = (df.groupby("date", as_index=False)
                       .agg(mean_score=("sentiment_score", "mean"),
                            n_articles=("sentiment_score", "size"),
                            pos_share=("sentiment_label", lambda s: float((s == "positive").mean()))))
        else:
            # TextBlob path
            if "polarity" not in df.columns or "subjectivity" not in df.columns:
                self.analyze_sentiment()
                df = self.memory
            daily = (df.groupby("date", as_index=False)
                       .agg(polarity=("polarity", "mean"),
                            subjectivity=("subjectivity", "mean"),
                            n_articles=("title", "size")))
        return daily

    # -----------------------------
    # Simple autonomous run (baseline)
    # -----------------------------
    def run(self, save_path: str = "agent_memory.csv") -> pd.DataFrame:
        """
        Baseline autonomous workflow (TextBlob):
          1) analyze_sentiment()
          2) summarize_daily()
          3) save_memory(save_path)
        """
        print("[AGENT] Running baseline workflow (TextBlob).")
        self.analyze_sentiment()
        summary = self.summarize_daily()
        self.save_memory(save_path)
        print("[AGENT] Workflow completed.")
        return summary

# -------------------------------
# CLI / example
# -------------------------------
if __name__ == "__main__":
    # Example usage (adjust paths as needed):
    example_csv = "NLP/articles/AAPL.csv"
    agent = LocalAgenticNewsAI(example_csv if os.path.exists(example_csv) else None)

    # Baseline
    agent.analyze_sentiment()
    daily_tb = agent.summarize_daily()
    agent.save_memory("NLP/sentiment_scores_agentic/AAPL__agent_textblob_articles.csv")
    daily_tb.to_csv("NLP/sentiment_scores_agentic/AAPL__agent_textblob_daily.csv", index=False)

    # Agent inference (point to a trained checkpoint directory)
    # model_dir = "NLP/models/trained_sentiment_agent"
    # agent.analyze_sentiment_agent(model_dir, device=None, batch_size=32, max_length=256)
    # daily_ag = agent.summarize_daily()
    # agent.save_memory("NLP/sentiment_scores_agentic/AAPL__agent_articles.csv")
    # daily_ag.to_csv("NLP/sentiment_scores_agentic/AAPL__agent_daily.csv", index=False)
