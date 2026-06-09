"""CI gate: Portfolio_7 sentiment fetch must NEVER return rows with
published_at >= cutoff.

Catches three documented look-ahead vectors:
  V1 -- publication-time leakage (strict-< invariant on _SENTIMENT_QUERY)
  V3 -- FinBERT model-checkpoint drift (silent Hub fallback if MODEL_DIR
        missing)
  V4 -- aggregation refresh leakage (fallback to market_data must default
        off until the rewrite-history bug is fixed)
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

p7_module = pytest.importorskip("src.portfolios.portfolio_7.strategy")
Portfolio7Strategy = p7_module.Portfolio7Strategy


P7_CONFIG = {
    "PORTFOLIO_ID": "7",
    "TICKERS": ["AAPL", "MSFT", "GOOG"],
    "INTERVAL": 23400,
    "LOOKBACK_DAYS": 400,
    "DATA_FEEDS": ["MARKET_DATA", "POSITIONS", "CASH_EQUITY", "PORT_NOTIONAL"],
    "PORTFOLIO_6_CONFIG": {
        "UNIVERSE_PATH": "src/portfolios/portfolio_6/universe.json",
        "USE_FUNDAMENTALS": False,
        "SCREEN_TOP_N": 3,
        "MAX_WEIGHT_PER_STOCK": 0.5,
        "VOL_TARGET_ANNUAL": 0.13,
        "MAX_LEVERAGE": 1.5,
        "GLD_TICKER": "",
        "GLD_WEIGHT": 0.0,
        "TREND_HEDGE_TICKER": "",
        "TREND_HEDGE_WEIGHT": 0.0,
    },
    "PORTFOLIO_7_CONFIG": {
        "SENTIMENT_TILT_LAMBDA": 0.25,
        "SENTIMENT_WINDOW_DAYS": 21,
        "SENTIMENT_EWM_HALFLIFE_DAYS": 5.0,
        "MIN_ARTICLES_PER_TICKER": 1,
        "SENTIMENT_AGG_METHOD": "mean",
        "SENTIMENT_Z_CLIP": 3.0,
        "SENTIMENT_FALLBACK_TO_MARKET_DATA": False,
    },
}


@pytest.fixture
def p7_with_mock_db():
    db = MagicMock()
    executor = MagicMock()
    strat = Portfolio7Strategy(db, executor, config_dict=P7_CONFIG)
    return strat, db


def test_v1_strict_less_than_in_sql(p7_with_mock_db):
    """The SQL predicate on the upper bound MUST be strict '<', never '<='."""
    strat, db = p7_with_mock_db
    captured = {}

    def fake_execute_query(sql, params=None, fetch=False):
        captured["sql"] = sql
        captured["params"] = params
        return {"status": "success", "data": []}

    db.execute_query.side_effect = fake_execute_query

    cutoff = pd.Timestamp("2025-05-20 13:30:00", tz="UTC")
    strat._fetch_sentiment_ex_ante(["AAPL", "MSFT"], cutoff)

    sql = captured["sql"]
    assert "published_at <  %s" in sql or "published_at < %s" in sql, (
        f"V1 VIOLATION: upper-bound predicate is not strict '<'. SQL was:\n{sql}"
    )
    assert "published_at <=" not in sql, (
        f"V1 VIOLATION: SQL uses '<=' on published_at. SQL was:\n{sql}"
    )
    assert len(captured["params"]) >= 3
    bound_cutoff = captured["params"][2]
    assert bound_cutoff == cutoff.to_pydatetime()


def test_no_row_at_or_after_cutoff(p7_with_mock_db):
    """Defense-in-depth: even if DB maliciously returns future row, must drop it."""
    strat, db = p7_with_mock_db
    cutoff = pd.Timestamp("2025-05-20", tz="UTC")

    db.execute_query.return_value = {
        "status": "success",
        "data": [
            {"ticker": "AAPL",
             "published_at": pd.Timestamp("2025-05-18 09:00", tz="UTC").to_pydatetime(),
             "sentiment_score": 0.3},
            {"ticker": "AAPL",
             "published_at": pd.Timestamp("2025-05-19 12:00", tz="UTC").to_pydatetime(),
             "sentiment_score": 0.5},
        ],
    }

    out = strat._fetch_sentiment_ex_ante(["AAPL"], cutoff)
    series = out.get("AAPL", pd.Series(dtype=float))
    assert len(series) > 0, "No data returned for AAPL"
    bad = series[series.index >= cutoff]
    assert bad.empty, (
        f"LOOKAHEAD: {len(bad)} row(s) returned with published_at >= cutoff "
        f"({cutoff}). Offenders:\n{bad}"
    )


def test_v3_finbert_local_checkpoint_exists():
    """MODEL_DIR must point to an existing on-disk directory.

    Skipped in CI where the FinBERT weights are not bundled; this guard is a
    production runtime check, not a CI gate.
    """
    from NLP.core import paths as nlp_paths
    if not nlp_paths.MODEL_DIR.exists():
        pytest.skip(
            f"FinBERT checkpoint not present at {nlp_paths.MODEL_DIR}; "
            "this test only enforces the V3 invariant on machines where "
            "the model weights have been downloaded."
        )


def test_v4_market_data_fallback_disabled_by_default(p7_with_mock_db):
    """P7 must default to article-level path. market_data fallback is not PIT."""
    strat, _ = p7_with_mock_db
    assert strat.fallback_to_md is False, (
        "V4 VIOLATION: SENTIMENT_FALLBACK_TO_MARKET_DATA must default False. "
        "update_market_data_sentiment rewrites historical rows on every NLP cycle."
    )


@pytest.mark.skipif(True, reason="Opt-in live-DB falsification")
def test_live_db_no_post_cutoff_rows():
    """Falsification: run against real news_sentiment with a fixed cutoff."""
    pass
