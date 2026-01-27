from src.live_trading.engine import RunEngine
from src.live_trading.executor import tradeExecutor
from src.portfolios.portfolio_1.strategy import VolMomentum
from src.portfolios.portfolio_2.strategy import MomentumStrategy


def test_trade_executor_and_engine_wiring(db_connection):
    """
    Basic integration-style test:

    - create a tradeExecutor with the DB connector
    - create a RunEngine with the same DB connector + executor
    - verify the engine is wired up correctly

    We intentionally DO NOT call any long-running methods (like a run loop);
    the goal here is just to ensure the objects can be constructed and wired.
    """

    # Create the trade executor with DB connector (constructor requires db_connector)
    executor = tradeExecutor(db_connector=db_connection)
    assert executor is not None, "tradeExecutor() should return a valid instance"

    # Create the live trading engine
    engine = RunEngine(db_connector=db_connection, executor=executor)
    assert engine is not None, (
        "RunEngine should initialize with DB connector and executor"
    )

    # The engine should actually hold the executor instance we passed in
    assert hasattr(engine, "executor"), "RunEngine should have an 'executor' attribute"
    assert engine.executor is executor, (
        "RunEngine.executor should reference the passed tradeExecutor"
    )

    # Optionally check that the engine can at least *see* the strategy classes
    # without blowing up on import level
    _ = [VolMomentum, MomentumStrategy]
