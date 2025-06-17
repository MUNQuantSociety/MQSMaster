This folder refactors the portfolio framework by removing the `run()` and `backtest()` methods from `BasePortfolio` and introducing separate engine classes (`RunEngine` and `BacktestEngine`). The goal is to improve modularity, testability, and flexibility when combining strategies with different execution modes.

## Changes

- **Removed** `run()` and `backtest()` from `portfolio_BASE/strategy.py`.
- **Added** two new engine classes in `engines/`:
  - `RunEngine`: handles the live polling loop and passes fetched data to the strategy.
  - `BacktestEngine`: wraps the historical replay (via `BacktestRunner`) and invokes the strategy with time-sliced data.
- **Updated** `main.py` (or `portfolioManager.py`) to instantiate a strategy and then choose either `RunEngine` or `BacktestEngine` to drive it.

## Benefits over previous structure

- **Separation of concerns**: Strategies focus solely on signal logic (`generate_signals_and_trade`), while engines handle orchestration (looping, timing, data retrieval for live or backtest).
- **Reusability & Flexibility**: Any `BasePortfolio` subclass can be plugged into either engine without altering its code. New execution modes (e.g., batch runs, paper-trade scheduler) can be added by creating new engines.
- **Testability**: Signal logic can be unit-tested in isolation by calling `generate_signals_and_trade` with synthetic data, without dealing with loops or timing. Engines can be tested separately by mocking strategies.
- **Maintainability**: Smaller, single-purpose classes/files are easier to navigate. Changes in loop/backtest logic do not risk breaking strategy code.
- **Readability & Onboarding**: Clear boundaries help new developers understand “how a strategy decides trades” vs. “how we drive it live or in backtest.”
- **Configurability**: Engines can accept different parameters (poll intervals, date ranges, logging hooks) without modifying strategy classes.
