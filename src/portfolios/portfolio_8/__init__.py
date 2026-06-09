"""Portfolio 8: Boring + Not-Lottery + RBP overlay.

Composes Portfolio_6's monthly screen with a Relevance-Based Prediction
rank as a fourth signal. Position sizing remains inverse-vol with the
existing P6 vol-target and hedge sleeves.
"""

from src.portfolios.portfolio_8.strategy import Portfolio8Strategy

__all__ = ["Portfolio8Strategy"]
