import sys
from datetime import datetime

import pandas as pd
import pytest

from NLP import fetch_articles
from NLP.fetch_alt_articles import ArticleScraper, normalize_published_date_column

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.workflow_nlp,
]


def test_article_scraper_initialization():
    scraper = ArticleScraper("AAPL")
    assert scraper.symbol == "AAPL"
    assert hasattr(scraper, "scrape_yahoo")
    assert hasattr(scraper, "scrape_finviz")
    assert hasattr(scraper, "scrape_alpha")


def test_article_data_format_validation():
    raw = pd.DataFrame(
        [
            {
                "publishedDate": "2025-01-01 12:00:00",
                "title": "Test title",
                "content": "Test content",
                "site": "https://example.com",
            }
        ]
    )

    normalized = normalize_published_date_column(raw)
    assert not normalized.empty
    assert "publishedDate" in normalized.columns
    assert pd.api.types.is_datetime64_any_dtype(normalized["publishedDate"])
    assert normalized.iloc[0]["title"] == "Test title"


def test_deduplication_logic():
    base_time = datetime(2025, 1, 1, 12, 0, 0)
    df1 = pd.DataFrame(
        [
            {
                "publishedDate": base_time,
                "title": "Duplicate",
                "content": "A",
                "site": "site-a",
            }
        ]
    )
    df2 = pd.DataFrame(
        [
            {
                "publishedDate": base_time,
                "title": "Duplicate",
                "content": "B",
                "site": "site-b",
            },
            {
                "publishedDate": base_time,
                "title": "Unique",
                "content": "C",
                "site": "site-c",
            },
        ]
    )

    combined = fetch_articles.remove_duplicates(df1, df2)
    assert len(combined) == 2

    duplicate_rows = combined[combined["title"] == "Duplicate"]
    assert len(duplicate_rows) == 1
    duplicate_row = duplicate_rows.iloc[0]
    assert duplicate_row["publishedDate"] == base_time
    assert duplicate_row["site"] == "site-a"
    assert duplicate_row["content"] == "A"

    unique_rows = combined[combined["title"] == "Unique"]
    assert len(unique_rows) == 1
    unique_row = unique_rows.iloc[0]
    assert unique_row["publishedDate"] == base_time
    assert unique_row["site"] == "site-c"
    assert unique_row["content"] == "C"


def test_fetch_articles_cli_parsing(monkeypatch):
    argv = ["fetch_articles.py", "AAPL", "2025-01-01", "2025-01-02"]
    monkeypatch.setattr(sys, "argv", argv)

    args = fetch_articles.parse_args()
    assert args.ticker == "AAPL"
    assert args.start_date == "2025-01-01"
    assert args.end_date == "2025-01-02"


def test_sentiment_processor_model_load(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    import NLP.sentiment_processor as sentiment_processor

    class _StubTokenizer:
        @classmethod
        def from_pretrained(cls, _model_dir):
            return cls()

    class _StubModel:
        @classmethod
        def from_pretrained(cls, _model_dir, torch_dtype=None, **_kwargs):
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(sentiment_processor, "AutoTokenizer", _StubTokenizer)
    monkeypatch.setattr(
        sentiment_processor, "AutoModelForSequenceClassification", _StubModel
    )

    processor = sentiment_processor.SentimentProcessor(model_dir=str(tmp_path))
    processor.load_model()

    assert processor.tokenizer is not None
    assert processor.model is not None
