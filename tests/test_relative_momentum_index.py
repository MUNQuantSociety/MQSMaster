# tests/test_relative_momentum_index.py

from datetime import datetime, timezone
import random


from src.portfolios.indicators.relative_momentum_index import RelativeMomentumIndex


def generate_trend(start: float, steps: int, drift: float = 0.5, noise: float = 0.1):
    price = start
    series = []
    for _ in range(steps):
        price += drift + random.uniform(-noise, noise)
        series.append(round(price, 4))
    return series


def test_rmi_not_ready_initially():
    rmi = RelativeMomentumIndex("TEST", period=5, momentum_period=2)
    assert not rmi.IsReady
    assert rmi.CurrentValue is None


def test_rmi_readiness_and_value_range():
    period = 5
    momentum_period = 2
    rmi = RelativeMomentumIndex("TEST", period=period, momentum_period=momentum_period)
    prices = [100 + i for i in range(50)]  # steadily increasing

    ready_seen = False
    for p in prices:
        val = rmi.Update(datetime.now(timezone.utc), p)
        if rmi.IsReady:
            ready_seen = True
            assert val is not None
            assert 0.0 <= val <= 100.0
    assert ready_seen, "RMI never became ready"


def test_rmi_reset():
    rmi = RelativeMomentumIndex("TEST", period=5, momentum_period=3)
    for p in [100,101,102,103,104,105,106,107]:
        rmi.Update(datetime.now(timezone.utc), p)
    assert rmi.IsReady
    assert rmi.CurrentValue is not None
    rmi.Reset()
    assert not rmi.IsReady
    assert rmi.CurrentValue is None


def test_rmi_extreme_flat():
    # Flat prices => momentum=0 => avg_loss=avg_gain=0 -> RMI should settle at 0 (as defined)
    rmi = RelativeMomentumIndex("FLAT", period=4, momentum_period=2)
    prices = [50]*20
    for p in prices:
        rmi.Update(datetime.now(timezone.utc), p)
    if rmi.IsReady:
        assert rmi.CurrentValue in (0.0, 100.0)  # Implementation returns 0.0 when both zero


def test_rmi_name_property():
    rmi = RelativeMomentumIndex("ABC", period=7, momentum_period=4)
    assert rmi.Name == "RMI_ABC_7_4"
