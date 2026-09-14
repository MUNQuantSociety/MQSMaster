"""VolMomentum's risk-off rule: it blocks new longs and nothing else.

The cash test that defines risk-off (cash under 10% of the book) is true on
every bar of a fully invested book, so it must never be a reason to sell —
doing so flattened and re-bought every still-bullish name on alternate bars.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.portfolios.portfolio_1.strategy import VolMomentum


def decide(momentum, *, position=0, cash=100_000):
    """Trade requests OnData makes for one ticker, bypassing __init__'s DB wiring."""
    prices = 100 * np.cumprod(1 + np.resize([0.01, -0.01], 42))
    history = pd.DataFrame({"close_price": prices})
    calls = []
    strategy = VolMomentum.__new__(VolMomentum)
    strategy.logger = logging.getLogger("vol-momentum-risk-off")
    strategy.PERIOD = 20
    strategy.TARGET_WEIGHT = 0.2
    strategy.tickers = ["AAPL"]
    strategy.roc = {"AAPL": SimpleNamespace(IsReady=True, Current=momentum)}
    context = SimpleNamespace(
        Market={
            "AAPL": SimpleNamespace(
                Exists=True, Close=prices[-1], History=lambda window: history
            )
        },
        Portfolio=SimpleNamespace(
            cash=cash,
            total_value=100_000,
            positions={"AAPL": position},
            get_asset_weight=lambda ticker, price: 0.1 if position else 0,
        ),
        buy=lambda ticker, **kw: calls.append(("buy", ticker)),
        sell=lambda ticker, **kw: calls.append(("sell", ticker)),
    )
    strategy.OnData(context)
    return calls


def test_bullish_signal_opens_a_long_with_cash_available():
    assert decide(50) == [("buy", "AAPL")]


def test_risk_off_blocks_a_new_long():
    assert decide(50, cash=5_000) == []


def test_risk_off_keeps_a_bullish_position_rather_than_flattening_it():
    assert decide(50, position=10, cash=5_000) == []


def test_risk_off_still_lets_a_bearish_position_close():
    assert decide(-50, position=10, cash=5_000) == [("sell", "AAPL")]
