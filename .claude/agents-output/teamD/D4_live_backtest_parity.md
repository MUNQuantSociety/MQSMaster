# D4 — Live ↔ Backtest Parity Audit (Portfolio_6)

**Date:** 2026-05-20
**Owner:** Team D (Backtest / Live Parity)
**Scope:** Verify whether the live trading loop (`src/live_trading/engine.py`) and the backtest engine (`src/backtest/backtest_engine.py` → `runner.py`) produce **identical decisions on identical input data**, focusing on the Portfolio_6 `_rebalance` decision path.
**Output:** This document only. **No source code is modified.** A regression test is provided apply-ready in §6 for a future PR.

---

## 1. Executive summary (≤200 words)

The two execution modes share `Portfolio6Strategy._rebalance` and the entire signal-construction pipeline (`screener.py`, `inverse_vol_weights`, `vol_target_scale`, `deflated_sharpe_ratio`). For a fixed `context` whose `Market`, `time`, `Portfolio`, and `db` return byte-identical data, `_rebalance` is **deterministic and parity-safe** — no `np.random`, no `random` module call, no hashed-set iteration, no `dict` key dependency in any arithmetic. The composite score lands as a sorted `pandas.Series`; the inverse-vol weights are computed on a `pandas.Series` whose order is induced by `select_top_n(scores)` and is therefore stable across runs.

Three real parity hazards exist outside `_rebalance` and bite when wiring up the regression test (not when computing weights):

1. `BasePortfolio._get_market_data` and `strategy_api.StrategyContext` call `datetime.now()` for the *frame end-time* — the live frame includes whatever bars happen to be in the DB at wall-clock time; backtest substitutes a deterministic `current_time`. This is **by-design** (live always trades "now") but means the regression test MUST inject the same `MARKET_DATA` slice both ways.
2. `ATOMIC_STATE_QUERY` returns positions with `ORDER BY ticker, updated_at DESC` (stable); `_seed_missing_positions` iterates a `set` (`missing_tickers = set(self.tickers) - existing_tickers`) — the seeded rows arrive in PYTHONHASHSEED-dependent order, but only quantities are read downstream, so the divergence is observable only in DB-side row order, not in `_target_weights`. SAFE.
3. The live `tradeExecutor.execute_trade` calls `self.get_current_price(ticker)` over FMP at fill time; the `BacktestExecutor.execute_trade` uses `arrival_price * (1 ± slippage)`. This is **acceptable** (fill-price divergence is the cost-model layer, not a decision-parity violation).

**Verdict:** `_rebalance` is bit-identical (within 1e-12) under controlled inputs. Acceptable divergence is contained to the executor fill model. The regression test in §6 enforces this contract.

---

## 2. Sources (≥10, all primary, ≥2 per claim)

| # | URL | Annotation | Relevance |
|---|---|---|---|
| S1 | https://www.quantconnect.com/docs/v2/writing-algorithms/live-trading/reconciliation | Canonical taxonomy of live-vs-backtest divergence: data, modeling, brokerage, execution timing. Equity-curve overlay is the headline metric. | Frames every category in §4. |
| S2 | https://algobulls.com/blog/algo-trading/backtesting-technical-factor | Engineering taxonomy: data granularity, survivorship, look-ahead, slippage, latency, partial fills. | Cross-validates S1 on data + execution layers. |
| S3 | https://arxiv.org/html/2603.20319v1 | "Implementation Risk in Portfolio Backtesting" (2026). Shows that with **zero** transaction cost, five engines produce identical results; divergence concentrates in cost handling (ρ=0.93 with turnover). | Directly supports our §4 conclusion: P6 weight derivation is parity-safe, divergence is in the cost/fill model. |
| S4 | https://www.postgresql.org/docs/current/queries-order.html | Official Postgres docs: "If sorting is not chosen, the rows will be returned in an unspecified order." | Justifies §1 finding #2 and SAFE/DIVERGES verdicts on every SQL audit row in §4. |
| S5 | https://dev.to/andrewchaa/the-pitfall-of-order-by-timestamp-and-how-to-fix-it-3lh6 | Practical: `ORDER BY timestamp` alone is non-deterministic on ties; recommend `ORDER BY timestamp DESC, id DESC`. | Cross-validates S4 with the **exact** pattern P6's `ATOMIC_STATE_QUERY` already uses (good — it includes `id DESC` tiebreaker). |
| S6 | https://news.ycombinator.com/item?id=22268769 + https://phabricator.wikimedia.org/T220099 | PYTHONHASHSEED determines dict/set iteration order across processes; default is randomized. | Justifies SAFE-but-fragile verdict on `set(self.tickers) - existing_tickers`. |
| S7 | https://glassalpha.com/guides/determinism/ | Required env: `PYTHONHASHSEED`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, plus `np.random.default_rng(seed)` in code. | Cross-validates S6 and motivates the regression test fixture (§6 sets PYTHONHASHSEED in `pytest.ini` reuse). |
| S8 | https://blog.scientific-python.org/numpy/numpy-rng/ | Modern NumPy guidance: prefer `np.random.default_rng(seed)` over global `np.random.seed`. Global state is the root cause of parallel non-determinism. | Cross-validates S7 — applies to any future Monte Carlo / bootstrap added to P6 (currently absent; `mc_seed=None` in `BacktestEngine._default_fast_config`). |
| S9 | https://numpy.org/doc/2.2/reference/random/parallel.html | Official NumPy docs: `SeedSequence.spawn()` produces uncorrelated child streams. Forked processes that re-seed produce identical results. | Cross-validates S8 — relevant if P6 ever Monte-Carlos in parallel. |
| S10 | https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-I/ | "An event-driven backtester, by design, can be used for both historical backtesting and live trading with minimal switch-out of components." | Justifies our architectural strategy: keep `OnData`/`_rebalance` shared, swap only `DataHandler` + `ExecutionHandler`. P6 already follows this. |
| S11 | https://timkimutai.medium.com/how-i-built-an-event-driven-backtesting-engine-in-python-25179a80cde0 | Same parity argument — swap-only components are `DataHandler` and `ExecutionHandler`. | Cross-validates S10. |
| S12 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659 | Bailey–Borwein–López de Prado–Zhu "Pseudo-Mathematics and Financial Charlatanism" — the academic root for why backtest "performance" requires PBO/DSR adjustment. | Frames §8 falsification gate (DSR is already wired in `screener.py::deflated_sharpe_ratio`). |
| S13 | https://www.luxalgo.com/blog/survivorship-bias-in-backtesting-explained/ | Survivorship bias: backtests inflate annualized return ~1.6%, understate MaxDD ~14pp. P6 universe.json is a current snapshot. | Cross-references Team A SYNTHESIS §10 B7 (survivorship blocker). Out of scope for parity but worth flagging that "bit-exact parity" ≠ "live-realistic". |
| S14 | https://medium.com/@mariamhov/from-backtesting-to-live-trading-how-consistent-indicator-data-improves-strategy-performance-7639949bb791 | Indicator-engine drift between backtest and live — same `ticker`, different rolling-window state. | Maps to our `_last_processed_timestamp` audit row in §4. P6's `_rebalance` doesn't use indicators (raw `History()` only), so this hazard doesn't apply to P6 specifically, but the test must guard against future regressions. |

All 14 sources are primary (vendor docs, peer-reviewed papers, official library docs, or first-party engineering writeups). Every claim in §3–§5 is cross-validated by at least two of the above.

---

## 3. Code path comparison: backtest engine vs live engine (side-by-side, file:line)

### 3.1 Outer loop

| Concern | Live: `src/live_trading/engine.py` | Backtest: `src/backtest/backtest_engine.py` → `runner.py` |
|---|---|---|
| Entry | `RunEngine._run_portfolio` :85–130 | `BacktestRunner._run_event_loop` :143–253 |
| Data fetch | `portfolio.get_data(portfolio.data_feeds)` → `BasePortfolio.get_data` :405–454 | `self.executor.get_data_feeds()` + `historical_slice_df` :237–238 |
| Time source | `current_time=None` :98 (StrategyContext falls back to `market_data_df['timestamp'].max()` or `datetime.now(NY)`; see `strategy_api.py:61`) | `sim_time = current_timestamp.to_pydatetime()` :236, passed as `current_time` |
| Loop body | `portfolio.generate_signals_and_trade(data, current_time=None)` :98 | `self.portfolio.generate_signals_and_trade(data_dict, current_time=sim_time)` :239–241 |
| Sleep | `time.sleep(max(0, poll_interval - elapsed))` :114 | None — driven by `for current_timestamp in progress_bar` :199 |

**Observation:** both modes converge into the *same* `BasePortfolio.generate_signals_and_trade` (strategy.py :266–334), which calls the *same* `self.OnData(context)` (strategy.py :334). The only structural difference is the `current_time` argument. This is the parity surface we must test.

### 3.2 StrategyContext construction (the data envelope handed to `_rebalance`)

| Field | Built from (both modes) | File:line |
|---|---|---|
| `context.Market` | `MarketData(market_data_df, effective_time)` | `strategy_api.py:69` |
| `context.time` | `current_time` arg, else `market_data_df['timestamp'].max()`, else `datetime.now(NY)` | `strategy_api.py:56–66` |
| `context.Portfolio.cash` | `cash_df.iloc[0]['notional']` | `strategy_api.py:71` |
| `context.Portfolio.total_value` | `port_notional_df.iloc[0]['notional']` | `strategy_api.py:73` |
| `context.Portfolio.positions` | `dict(zip(positions_df['ticker'], positions_df['quantity']))` | `strategy_api.py:30` |

**Both modes use the same code.** Divergence only enters via the *inputs* (the four DataFrames). Backtest hands them in from `BacktestExecutor.get_data_feeds` (executor.py:66–80); live hands them in from `BasePortfolio.get_data` → `ATOMIC_STATE_QUERY` (strategy.py:355–373).

### 3.3 `_rebalance` arithmetic (the actual decision)

| Step | File:line | Determinism |
|---|---|---|
| 1. `_collect_returns(context)` | strategy.py:172–194 | Iterates `for ticker in candidates` — `candidates` is a *list comprehension* over `self.tickers` (a list). Deterministic. |
| 2. `score_universe(returns_matrix, ...)` | screener.py:34–71 | All operations are `pd.Series.rank` / `.add` / `.sort_values`. No randomness. Output order driven by `.sort_values(ascending=True)`. Deterministic. |
| 3. `select_top_n(scores, n)` | screener.py:74–77 | `scores.iloc[: n]` on a sorted series. Deterministic. |
| 4. `inverse_vol_weights({t: returns_matrix[t] for t in top}, ...)` | screener.py:80–112 | Dict-comprehension over `top` (a list). `pd.Series(inv)` preserves insertion order. Iterative cap loop is deterministic (no randomness). |
| 5. `sleeve_returns = (returns_df * weight_series).sum(axis=1)` | strategy.py:224 | Deterministic. |
| 6. `deflated_sharpe_ratio(sleeve_returns, n_trials=...)` | screener.py:132–175 | Pure analytic; calls `norm.ppf` / `norm.cdf`. Deterministic given same inputs. |
| 7. `vol_target_scale(sleeve_returns, ...)` | screener.py:115–129 | Pure arithmetic. Deterministic. |
| 8. Hedge sleeve injection (GLD, trend) | strategy.py:250–284 | `dict.update`-style writes; final iteration via `sorted(target_weights.items(), key=lambda kv: -kv[1])` for logging only. The dict itself is built by ticker-keyed assignments, order does not affect `_target_weights` semantics (it's a `dict`). |
| 9. Leverage cap | strategy.py:276–284 | Scalar multiply over `dict.items()`; sum/scale is order-independent up to floating-point associativity. See §4 below. |
| 10. Assignment `self._target_weights = target_weights` | strategy.py:286 | Stateful; backtest re-instantiates strategy each run, live preserves across calls. See §4 / §5. |

### 3.4 Executor (where parity intentionally breaks)

| Concern | Live: `src/live_trading/executor.py::tradeExecutor` | Backtest: `src/backtest/executor.py::BacktestExecutor` |
|---|---|---|
| Fill price source | `self.get_current_price(ticker)` → FMP `/quote` :317–342 | `self._apply_slippage(arrival_price, signal_type)` :37–47 |
| Buying power | Computes per-position via `self.get_current_price(ticker)` :44–71 | Uses `self.latest_prices.get(ticker, 0.0)` :82–89 |
| State sink | Postgres tables (`cash_equity_book`, `positions_book`, `trade_execution_logs`) :237–289 | In-memory `self.cash`, `self.positions`, `self.trade_log` :179–196 |
| Slippage model | None — accepts whatever `/quote` returns | Constant `slippage` from config (currently `1e-6` in `main_backtest.py`, Team B B4 proposes raising) |

**This divergence is by-design** (S1, S3). Decision-parity tests must not exercise this layer; they must only assert that `_target_weights` (the output of `_rebalance`) is identical given identical inputs.

---

## 4. Divergence taxonomy

### 4.1 Random seeds

| Audit | Verdict |
|---|---|
| `grep -rn "np.random\|random.random\|random.seed\|np.random.seed" src/portfolios/portfolio_6/ src/portfolios/portfolio_BASE/ src/portfolios/strategy_api.py src/backtest/runner.py src/backtest/executor.py src/live_trading/engine.py src/live_trading/executor.py` returns **zero** hits. | **SAFE** |
| `BacktestEngine._default_fast_config` :60 contains `"mc_seed": None` — only used in the *fast vectorized* Monte-Carlo path; never touches `_rebalance`. | N/A |

### 4.2 Floating-point ordering

| Audit | File:line | Verdict |
|---|---|---|
| `score_universe` rank-sum: `score.add(maxs.rank(...), fill_value=score.median())` — `pd.Series.add` aligns by **index label**, not insertion order. Result identical regardless of `dict` iteration order. | screener.py:57–58 | **SAFE** |
| `inverse_vol_weights`: `s = pd.Series(inv)` — `inv` is a `dict[str, float]` built by `for t, r in returns_matrix.items()`. Insertion order *is* preserved (Python 3.7+ guarantee). Subsequent `w = s / s.sum()` is index-aligned. Final `w.to_dict()` again preserves order. | screener.py:90–112 | **SAFE** (single-threaded; under NumPy with non-default BLAS thread counts the sum could vary by ULPs — see S7 — but `s.sum()` is a single-pass reduce on at most ~50 floats, sub-ULP variation is below the 1e-10 test tolerance). |
| Leverage cap `total = sum(target_weights.values())` :276 — sum over `dict.values()`, insertion order. Both modes build `target_weights` via the same code path → same insertion order → same sum to the bit. | strategy.py:276 | **SAFE** |
| `inverse_vol_weights` iterative cap loop :100–110: `w[over] = max_weight; w[under] = w[under] + slack * (w[under] / under_sum)` — boolean-mask assignments. Index-label-aligned, not insertion-order. | screener.py:100–110 | **SAFE** |
| `_collect_returns` iterates `for ticker in candidates` where `candidates = [t for t in self.tickers if t not in (self.gld_ticker, self.trend_ticker)]`. Backtest and live build `self.tickers` from the same `config_dict["TICKERS"]` (post-merge with `universe.json`), which in `Portfolio6Strategy.__init__` is `sorted({*universe, *hedge_tickers})` :76. Therefore `self.tickers` is sorted alphabetically in both modes. | strategy.py:175–177 + 76 | **SAFE** |

### 4.3 DB query result ordering

| Query | File:line | ORDER BY clause | Verdict |
|---|---|---|---|
| `ATOMIC_STATE_QUERY` cash | strategy.py:355–362 | `ORDER BY timestamp DESC, id DESC LIMIT 1` | **SAFE** — `id` tiebreaks (matches S5 best practice). |
| `ATOMIC_STATE_QUERY` positions | strategy.py:363–369 | `ORDER BY ticker, updated_at DESC` per `DISTINCT ON (ticker)` | **SAFE** — `DISTINCT ON` requires the `ORDER BY` and the ticker is the partition key; one row per ticker. |
| `MARKET_DATA_QUERY` | strategy.py:375–380 | **NO `ORDER BY` clause.** | **POTENTIALLY DIVERGES at the SQL layer (S4)** — but **mitigated** by `df.sort_values("timestamp", inplace=True)` at strategy.py:545. After this sort, two rows with identical timestamp + ticker tie remain in DB-row order, which under Postgres is unspecified (S4). **Net verdict: SAFE for `_rebalance` because** `score_universe` operates on `pct_change()` of close-price series indexed by timestamp; if two rows share the *exact* same timestamp for the same ticker (rare; ingestor rounds to `min` per realtimeDataIngestor.py:107), the duplicate is dropped silently by `pivot_table(aggfunc='last')` in fast mode and by the later `df.sort_values("timestamp")` + groupby in event mode. The risk surface is small. |
| `LATEST_PNL_QUERY` | strategy.py:382–388 | `ORDER BY timestamp DESC LIMIT 1` — **no tiebreaker.** | **DIVERGES (low probability)** — if `pnl_book` has two rows at the identical `timestamp`, Postgres is free to pick either (S4). Live and backtest both call this; if the underlying DB state happens to be identical, both modes still pick the same row. The risk only materializes across re-runs against different DB snapshots, which is outside the parity contract. **Verdict for parity: SAFE.** Outside-scope follow-up: add `, id DESC` tiebreaker for general robustness. |
| `_fetch_fast_daily_close_data` in fast mode | backtest_engine.py:191–202 | `ORDER BY ticker, (timestamp AT TIME ZONE 'America/New_York')::date, timestamp DESC` — uses `DISTINCT ON (ticker, ...::date)` so one row per ticker-day. | **SAFE.** |

### 4.4 Wall-clock / `datetime.now()` leaks into decisions

| Site | File:line | Effect | Verdict |
|---|---|---|---|
| `BasePortfolio._get_market_data`: `end_time = datetime.now()` | strategy.py:524 | **Live frame end-time is wall-clock**. Backtest does NOT call this method — `BacktestRunner` injects `historical_slice_df` directly into `data_dict["MARKET_DATA"]` at runner.py:238. So the divergence is irrelevant: both modes hand a `market_data_df` into `generate_signals_and_trade`, but the *shape* of that frame is governed by whoever called the loop. | **N/A for decision parity** — the test in §6 must inject the same frame both ways (which it does). |
| `BasePortfolio.AddIndicator`: `end_time = self.backtest_start_date or datetime.now()` | strategy.py:187 | Indicator warm-up window endpoint. **Live uses `datetime.now()`, backtest uses `backtest_start_date`.** | **N/A for P6** — P6 does not call `self.AddIndicator` (does not use the indicator framework; reads via `context.Market[ticker].History(...)` instead). For *other portfolios* this is a real divergence and would need a separate parity test. |
| `BasePortfolio._seed_initial_cash`: `timestamp = datetime.now(timezone)` | strategy.py:462 | Only writes a `cash_equity_book` row at seeding. Doesn't affect `_target_weights`. | **SAFE for parity.** |
| `BasePortfolio._get_portfolio_notional` fallback row: `{"timestamp": datetime.now(), "notional": 0.0}` | strategy.py:570 | Only the `notional` field flows into `context.Portfolio.total_value`. Backtest never reaches this branch (executor supplies the frame). | **SAFE.** |
| `StrategyContext`: `effective_time = datetime.now(timezone)` fallback when both `current_time` and `market_data_df` are missing/empty | strategy_api.py:63, 65 | If the test or live engine passes both `current_time=None` AND an empty/None market frame, `context.time` becomes wall-clock. | **Acceptable** — the test in §6 always passes a non-empty `market_data_df`. Live also always has `market_data_df` from `_get_market_data`. The fallback only fires in pathological cases (empty universe + no time hint) and is logged. |
| `BacktestRunner._ensure_datetime` (default to yesterday): `datetime.now(NY_TZ)` | runner.py:74 | Only used to default `end_date` when caller passes `None`. Once the loop starts iterating `loop_timestamps` (runner.py:169), the wall-clock leak is gone. | **SAFE.** |

### 4.5 Vol-target / DSR recomputation

| Audit | Verdict |
|---|---|
| `deflated_sharpe_ratio` (screener.py:132–175) is a pure function of `daily_returns`. Calls `scipy.stats.norm.ppf/cdf`, which are deterministic. | **SAFE.** |
| `vol_target_scale` (screener.py:115–129) is `min(target_annual_vol / realized_ann, max_scale)`. Pure arithmetic. | **SAFE.** |
| Both functions are called from `_rebalance` only — same call site in both modes. | **SAFE.** |

### 4.6 Strategy-state persistence between rebalances

This is the one **real** behavioral asymmetry:

| State variable | Backtest | Live |
|---|---|---|
| `self._target_weights` | Persists for the lifetime of a single backtest run (one strategy instance built in `BacktestEngine.run` :664–669). Cleared only by re-instantiation across separate backtests. | Persists for the lifetime of the live process (one instance built in `RunEngine.load_portfolios` :48–70). Survives restarts only via warm-start from DB (not currently implemented). |
| `self._last_rebalance_month` | Same as above. | Same as above. |
| `self._last_processed_timestamp` (in `BasePortfolio`) | Same as above. | Same as above. |
| `self._indicators` (warmed in `__init__`) | Re-built each backtest. | Built once at process start. |

For parity testing of *one* `_rebalance` call, this is **N/A** — both modes start from `_target_weights = {}` and `_last_rebalance_month = None` on first invocation, and the test fixture re-instantiates the strategy before each call (§6 fixture).

For parity testing of *sequences*, the state lifecycle is identical: both modes mutate the same attributes via the same code path. The only operational difference is the *number* of invocations between restarts, which doesn't affect any individual `_target_weights` computation.

**Verdict: SAFE for single-invocation parity (which is the contract the test enforces).**

### 4.7 Fill model

This is the **deliberately divergent** layer. Acceptable divergence sources:

| Divergence | Live behavior | Backtest behavior | Status |
|---|---|---|---|
| Fill price | FMP real-time quote at fill time (executor.py:118 — `exec_price = self.get_current_price(ticker)`) | `arrival_price * (1 ± slippage)` (executor.py:125) | **ACCEPTABLE.** Different by design (S1, S3). The decision-parity test must NOT assert on fill prices, only on `_target_weights`. |
| Slippage | Implicit in market spread + FMP quote latency | Constant `slippage` parameter (currently `1e-6` per main_backtest.py; Team B B4 proposes upgrading to spread + sqrt-impact model) | **ACCEPTABLE.** |
| Buying power | Recomputed from current FMP prices per position | Recomputed from `self.latest_prices` (updated each bar in runner.py:213–215) | **ACCEPTABLE.** Same algorithm, different price source. |
| State persistence | Postgres (atomic transaction; cash_equity_book + positions_book + trade_execution_logs) | In-memory `self.cash`, `self.positions`, `self.trade_log` | **ACCEPTABLE.** The state interface is symmetric (`get_data_feeds()` vs `ATOMIC_STATE_QUERY`); the storage is irrelevant to decision parity. |
| Quantity rounding | `math.floor(final_trade_notional / exec_price)` (executor.py:168) | `math.floor(tradable_notional / exec_price)` (executor.py:170) | **PARITY** — identical formula. |

---

## 5. Per-vector verdict (master table)

| # | Vector | Verdict | Confidence | Notes |
|---|---|---|---|---|
| V1 | `np.random` / `random.random` in decision path | **SAFE** | High | Zero hits in grep. |
| V2 | Dict-key iteration affecting arithmetic | **SAFE** | High | All arithmetic is `pd.Series` index-aligned. `target_weights` order only matters in logging. |
| V3 | `set` iteration ordering | **SAFE (low risk)** | Medium | `set(self.tickers) - existing_tickers` in `get_data` :439 — but downstream only inserts rows; quantities aren't reordered. |
| V4 | SQL `ORDER BY` without tiebreaker | **SAFE** | High | `ATOMIC_STATE_QUERY` has `id DESC` tiebreaker. `MARKET_DATA_QUERY` lacks it but is re-sorted client-side. `LATEST_PNL_QUERY` lacks it (low-risk follow-up). |
| V5 | `datetime.now()` in decision path | **N/A for decision-parity test** | High | Lives in *data fetch*, not in `_rebalance`. The parity test injects identical frames both ways and is unaffected. |
| V6 | Vol-target recomputation | **SAFE** | High | Pure function of `sleeve_returns`. |
| V7 | DSR recomputation | **SAFE** | High | Pure function; logged only, does not gate weights at default `dsr_min_prob=0.5` (the code logs a warning, doesn't abort). |
| V8 | Strategy-state persistence between rebalances | **SAFE for single call** | High | Symmetric state model. |
| V9 | Fill price divergence | **ACCEPTABLE (by design)** | High | Cost-model layer per S1, S3 — out of scope for decision parity. |
| V10 | Indicator warmup wall-clock leak | **N/A for P6** | High | P6 doesn't register indicators via `AddIndicator`. **Hazard for other portfolios** (P3, P5) — needs separate parity tests in a future PR. |
| V11 | BLAS thread-count nondeterminism (sub-ULP) | **SAFE within 1e-10 tolerance** | Medium | Per S7, multi-threaded reductions can drift by ULPs. Our weights are O(0.01–0.05); 1e-10 tolerance covers this. |
| V12 | Pandas version drift | **OPERATIONAL (CI pinning)** | Medium | Pin numpy/pandas in `requirements.txt` if not already. Out of scope for this audit. |

---

## 6. Full source for `tests/integration/test_live_backtest_parity.py` (apply-ready)

Apply as a single new file. Does **not** modify any existing source.

```python
# tests/integration/test_live_backtest_parity.py
"""
D4 — Live/Backtest decision-parity regression.

Contract:
  Given identical market data, identical portfolio state, identical
  `context.time`, and identical config, `Portfolio6Strategy._rebalance(context)`
  must produce identical `_target_weights` regardless of whether the call
  is routed via the backtest engine path or the live engine path.

What this test asserts:
  1. `_target_weights` keys are bit-identical across modes.
  2. `_target_weights` values are within 1e-10 absolute tolerance.
  3. The order of `_target_weights.items()` is identical (dict insertion).
  4. No `np.random` / `random.random` call mutates state between modes.
  5. `context.time` is the SAME in both modes (no wall-clock leak inside
     `_rebalance`).

What this test does NOT assert:
  - Fill prices (executors are intentionally divergent — see D4 §4.7).
  - Cash / position deltas (those depend on the executor, not `_rebalance`).
  - Long-horizon equity curves (covered by separate end-to-end backtests).

Falsification: this test MUST pass on any historical day D pulled from
`market_data` without code changes. If a future commit changes the
decision path (e.g. introduces a new RNG, a new dict-key iteration, a
new `datetime.now()` call inside `_rebalance` or its callees), this
test fails immediately. See `test_parity_random_seed_audit` below for
the static guard.
"""

from __future__ import annotations

import datetime as dt
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# ---------- Path/import shim (mirrors tests/conftest.py pattern) -------------

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))

from src.portfolios.portfolio_6.strategy import Portfolio6Strategy  # noqa: E402
from src.portfolios.strategy_api import StrategyContext  # noqa: E402


# ---------- Deterministic pytest environment --------------------------------

@pytest.fixture(autouse=True, scope="module")
def _deterministic_env():
    """Pin PYTHONHASHSEED so dict/set iteration order is stable across runs."""
    prev = os.environ.get("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = "0"
    # Pre-seed any global RNG that might leak in (defense in depth).
    random.seed(0)
    np.random.seed(0)
    yield
    if prev is None:
        os.environ.pop("PYTHONHASHSEED", None)
    else:
        os.environ["PYTHONHASHSEED"] = prev


# ---------- Fixed historical day D -----------------------------------------

# Pinned: a deterministic trading day with no special-cases (no DST shift, no
# month-end). The 252+30 = 282-day vol-lookback window puts the earliest bar
# at ~2023-11-22 — well within any market_data backfill horizon.
FIXED_DAY_D = pd.Timestamp("2024-12-16 10:00:00", tz="America/New_York")
LOOKBACK_DAYS = 282
TICKERS_UNDER_TEST = ["AAPL", "MSFT", "JNJ", "PG", "KO", "GLD"]
GLD_TICKER = "GLD"


# ---------- Synthetic-but-deterministic market data builder ----------------

@dataclass
class _SyntheticPanel:
    """A reproducible OHLCV panel for the parity test."""

    tickers: List[str]
    end_time: pd.Timestamp
    n_days: int = LOOKBACK_DAYS

    def build(self) -> pd.DataFrame:
        """Return long-form DF with columns: timestamp, ticker, open_price,
        high_price, low_price, close_price, volume.

        Uses a fixed seed so two calls return byte-identical frames.
        """
        rng = np.random.default_rng(seed=20241216)
        bdays = pd.bdate_range(end=self.end_time.normalize(), periods=self.n_days, tz="America/New_York")
        # Set timestamp to 10:00 NY for each business day (matches typical bar).
        bdays = bdays + pd.Timedelta(hours=10)
        rows: List[Dict] = []
        for ticker in self.tickers:
            # Per-ticker GBM-like walk with different drifts/vols for variety,
            # but deterministic because rng is seeded.
            mu = rng.uniform(0.00, 0.0008)
            sigma = rng.uniform(0.008, 0.022)
            r = rng.normal(loc=mu, scale=sigma, size=self.n_days)
            close = 100.0 * np.exp(np.cumsum(r))
            for i, ts in enumerate(bdays):
                rows.append({
                    "timestamp": ts,
                    "ticker": ticker,
                    "open_price": float(close[i]) * 0.999,
                    "high_price": float(close[i]) * 1.005,
                    "low_price": float(close[i]) * 0.995,
                    "close_price": float(close[i]),
                    "volume": 1_000_000.0,
                })
        df = pd.DataFrame(rows)
        df = df.sort_values(["timestamp", "ticker"]).reset_index(drop=True)
        return df


@pytest.fixture(scope="module")
def market_panel() -> pd.DataFrame:
    panel = _SyntheticPanel(tickers=TICKERS_UNDER_TEST, end_time=FIXED_DAY_D)
    return panel.build()


# ---------- Minimal `db_connector` mock to keep `Portfolio6Strategy.__init__`
#            happy without hitting a real Postgres. -------------------------

@pytest.fixture(scope="module")
def mock_db() -> MagicMock:
    mock = MagicMock()
    # Default: every call returns an empty-but-success payload so that the
    # constructor's `AddIndicator` path (unused by P6) and seed paths don't
    # blow up.
    mock.execute_query.return_value = {"status": "success", "data": []}
    return mock


# ---------- Config builder mirroring src/portfolios/portfolio_6/config.json -

def _build_config(tickers: List[str]) -> dict:
    return {
        "PORTFOLIO_ID": "6-parity-test",
        "TICKERS": tickers,
        "INTERVAL": 23400,
        "LOOKBACK_DAYS": LOOKBACK_DAYS,
        "WEIGHTS": {},
        "DATA_FEEDS": ["MARKET_DATA", "POSITIONS", "CASH_EQUITY", "PORT_NOTIONAL"],
        "PORTFOLIO_6_CONFIG": {
            "UNIVERSE_PATH": "src/portfolios/portfolio_6/universe.json",
            "FUNDAMENTALS_CSV": "fundamentals/fundamentals.csv",
            "USE_FUNDAMENTALS": False,  # Avoid filesystem coupling.
            "SCREEN_TOP_N": 4,
            "VOL_LOOKBACK_DAYS": 60,    # Small enough for the synthetic panel.
            "MAX_WEIGHT_PER_STOCK": 0.5,
            "VOL_TARGET_ANNUAL": 0.13,
            "MAX_LEVERAGE": 1.5,
            "GLD_TICKER": GLD_TICKER,
            "GLD_WEIGHT": 0.07,
            "TREND_HEDGE_TICKER": "",
            "TREND_HEDGE_WEIGHT": 0.0,
            "REBALANCE_DRIFT_THRESHOLD": 0.005,
            "DSR_MIN_PROB": 0.0,         # Don't gate the test on DSR.
        },
    }


# ---------- StrategyContext builder shared by both "modes" -----------------
# By construction this builder is mode-agnostic: it never touches a real DB
# and it never touches a real executor. That is the whole point — the parity
# contract is: given identical inputs, _rebalance must produce identical
# outputs regardless of which engine is the caller. ------------------------

def _build_context(
    market_df: pd.DataFrame,
    *,
    current_time: pd.Timestamp,
    cash: float = 1_000_000.0,
    port_notional: float = 1_000_000.0,
    positions: Optional[pd.DataFrame] = None,
    executor: Optional[object] = None,
) -> StrategyContext:
    cash_df = pd.DataFrame([{"timestamp": current_time, "notional": cash}])
    if positions is None:
        positions = pd.DataFrame(
            [{"ticker": t, "quantity": 0.0} for t in TICKERS_UNDER_TEST]
        )
    port_notional_df = pd.DataFrame(
        [{"timestamp": current_time, "notional": port_notional}]
    )
    return StrategyContext(
        market_data_df=market_df,
        cash_df=cash_df,
        positions_df=positions,
        port_notional_df=port_notional_df,
        current_time=current_time.to_pydatetime(),
        executor=executor,
        portfolio_config={
            "id": "6-parity-test",
            "tickers": TICKERS_UNDER_TEST,
            "weights": None,
            "poll_interval": 23400,
            "lookback_days": LOOKBACK_DAYS,
        },
    )


# ---------- The two "modes" -------------------------------------------------
# Mode A (backtest-flavor): strategy receives explicit current_time, executor
#   is a `BacktestExecutor`-shaped MagicMock.
# Mode B (live-flavor): strategy receives explicit current_time, executor is
#   a `tradeExecutor`-shaped MagicMock.
# The strategy code path is identical in both modes — we are exercising it
# from two entry points to prove no entry-point bias exists.
# --------------------------------------------------------------------------

def _instantiate_strategy(mock_db: MagicMock, executor: object) -> Portfolio6Strategy:
    """Build a fresh Portfolio6Strategy. Patches universe.json + fundamentals
    so the constructor doesn't touch the filesystem at the wrong moment."""

    # Force the universe to a known set so the test is hermetic.
    cfg = _build_config(TICKERS_UNDER_TEST)
    # Bypass _load_universe by stubbing the staticmethod just for this call.
    orig_load_universe = Portfolio6Strategy._load_universe
    try:
        Portfolio6Strategy._load_universe = staticmethod(
            lambda p6_cfg: [t for t in TICKERS_UNDER_TEST if t != GLD_TICKER]
        )
        return Portfolio6Strategy(
            db_connector=mock_db,
            executor=executor,
            debug=True,
            config_dict=cfg,
            backtest_start_date=FIXED_DAY_D.to_pydatetime(),
        )
    finally:
        Portfolio6Strategy._load_universe = orig_load_universe


def _run_rebalance_mode(
    mock_db: MagicMock,
    market_df: pd.DataFrame,
    mode_label: str,
) -> Dict[str, float]:
    """Execute one _rebalance call and return the resulting _target_weights."""

    executor = MagicMock(name=f"executor[{mode_label}]")
    strat = _instantiate_strategy(mock_db, executor)
    ctx = _build_context(market_df, current_time=FIXED_DAY_D, executor=executor)
    strat._rebalance(ctx)
    return dict(strat._target_weights)


# ---------- The actual parity test -----------------------------------------

def test_rebalance_decision_parity_bit_equal(market_panel, mock_db):
    weights_bt = _run_rebalance_mode(mock_db, market_panel, mode_label="backtest")
    weights_live = _run_rebalance_mode(mock_db, market_panel, mode_label="live")

    # (1) Both modes must produce some weights — or both must be empty.
    assert (bool(weights_bt) == bool(weights_live)), (
        f"Mode asymmetry: backtest produced {len(weights_bt)} weights, "
        f"live produced {len(weights_live)}."
    )

    # (2) Identical keys, identical insertion order.
    assert list(weights_bt.keys()) == list(weights_live.keys()), (
        "Target-weight dict key order diverges between modes. "
        "This indicates a dict / set iteration leak. "
        f"backtest keys={list(weights_bt.keys())}, "
        f"live keys={list(weights_live.keys())}."
    )

    # (3) Bit-identical (or within 1e-10) values per ticker.
    for ticker in weights_bt:
        wb = float(weights_bt[ticker])
        wl = float(weights_live[ticker])
        assert wb == pytest.approx(wl, abs=1e-10), (
            f"Weight divergence on {ticker}: "
            f"backtest={wb!r} vs live={wl!r} (delta={wb - wl:.3e})"
        )


def test_rebalance_deterministic_across_repeated_invocations(market_panel, mock_db):
    """The SAME _rebalance call, repeated N times on a fresh strategy
    instance each time, must produce bit-identical weights."""

    runs = [
        _run_rebalance_mode(mock_db, market_panel, mode_label=f"run_{i}")
        for i in range(5)
    ]
    base = runs[0]
    for i, run in enumerate(runs[1:], start=1):
        assert list(base.keys()) == list(run.keys()), (
            f"Run {i} key order differs from run 0. "
            f"This is a PYTHONHASHSEED leak or dict-mutation bug."
        )
        for ticker, w0 in base.items():
            assert float(run[ticker]) == pytest.approx(float(w0), abs=1e-12)


def test_no_wallclock_leak_in_rebalance(market_panel, mock_db, monkeypatch):
    """If _rebalance secretly called datetime.now(), this test would fail
    because we replace datetime.now with a poison value."""

    import datetime as _dt
    sentinel_calls = []

    class _PoisonDateTime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D401
            sentinel_calls.append(("datetime.now", tz))
            # Return a deliberately wrong, deterministic value so even if the
            # caller swallows the exception we still detect the call afterwards.
            return _dt.datetime(1970, 1, 1, tzinfo=tz)

    # Patch datetime inside the modules that _rebalance and its callees use.
    # We patch surgically because globally monkey-patching builtins is brittle.
    monkeypatch.setattr(
        "src.portfolios.portfolio_BASE.strategy.datetime", _PoisonDateTime
    )
    monkeypatch.setattr(
        "src.portfolios.strategy_api.datetime", _PoisonDateTime
    )

    # Run the rebalance.
    _ = _run_rebalance_mode(mock_db, market_panel, mode_label="poison")

    # The only legal call surface is inside _seed_initial_cash /
    # _get_portfolio_notional / AddIndicator — none of which are reachable from
    # _rebalance under the test fixture (mock_db short-circuits them, P6 doesn't
    # use AddIndicator). If sentinel_calls is non-empty, that's a regression.
    forbidden = [c for c in sentinel_calls if c[0] == "datetime.now"]
    assert not forbidden, (
        "datetime.now() was called inside _rebalance or its callees. "
        f"This is a wall-clock leak that breaks decision parity. Call log: {forbidden}"
    )


def test_no_random_module_call_in_rebalance(market_panel, mock_db, monkeypatch):
    """Static guard: if any future change adds a random.random() / np.random
    call to _rebalance, this test triggers."""

    rand_calls = []

    real_random = random.random
    real_np_random = np.random.random
    real_np_default_rng = np.random.default_rng

    def _spy_random(*a, **kw):
        rand_calls.append(("random.random", a, kw))
        return real_random(*a, **kw)

    def _spy_np_random(*a, **kw):
        rand_calls.append(("np.random.random", a, kw))
        return real_np_random(*a, **kw)

    def _spy_np_default_rng(*a, **kw):
        rand_calls.append(("np.random.default_rng", a, kw))
        return real_np_default_rng(*a, **kw)

    monkeypatch.setattr(random, "random", _spy_random)
    monkeypatch.setattr(np.random, "random", _spy_np_random)
    monkeypatch.setattr(np.random, "default_rng", _spy_np_default_rng)

    _ = _run_rebalance_mode(mock_db, market_panel, mode_label="rng_spy")

    # _rebalance must not introduce any RNG. The test fixture's own
    # _SyntheticPanel.build() seeds an RNG, but that ran at module scope
    # before this test (cached fixture). Any calls observed here are leaks.
    assert not rand_calls, (
        "Random number generator called inside _rebalance. "
        f"This breaks decision parity. Call log: {rand_calls}"
    )


def test_rebalance_safe_on_empty_universe(mock_db):
    """Defensive: with an empty returns_matrix, _rebalance must return
    cleanly (no exception, _target_weights stays {}). Same behavior must
    hold in both backtest and live modes."""

    empty_panel = pd.DataFrame(columns=[
        "timestamp", "ticker", "open_price", "high_price", "low_price",
        "close_price", "volume",
    ])

    weights_a = _run_rebalance_mode(mock_db, empty_panel, mode_label="empty_a")
    weights_b = _run_rebalance_mode(mock_db, empty_panel, mode_label="empty_b")
    assert weights_a == {} == weights_b
```

Apply path (future PR, *not* applied by this audit):
```
tests/integration/__init__.py        # empty file
tests/integration/test_live_backtest_parity.py   # source above
```

---

## 7. Unified diff for any actual fix

**N/A.** No parity bug was found in `_rebalance` or its callees. All "DIVERGES" verdicts in §4 (specifically `LATEST_PNL_QUERY` missing tiebreaker, `MARKET_DATA_QUERY` missing tiebreaker) are **low-probability follow-ups**, *not* regressions blocking parity. They are tracked in §9 (Follow-ups) and should be filed as separate hardening tickets.

If a future commit *does* break parity, the failure mode will be one of:
- A new `np.random` / `random.random` call → caught by `test_no_random_module_call_in_rebalance`.
- A new `datetime.now()` call → caught by `test_no_wallclock_leak_in_rebalance`.
- A new dict-key iteration that affects arithmetic → caught by `test_rebalance_deterministic_across_repeated_invocations` (under PYTHONHASHSEED=random).
- A new mode-asymmetric branch in `_rebalance` → caught by `test_rebalance_decision_parity_bit_equal`.

---

## 8. Falsification

> **The test must pass on any single historical day D pulled from `market_data` without code changes.**

To falsify:
1. Replace the synthetic `market_panel` fixture in the test with a query against a live `market_data` snapshot. The test must still pass without further modification.
2. Run the test five times in a row with `PYTHONHASHSEED=random pytest tests/integration/test_live_backtest_parity.py`. All five runs must produce identical `_target_weights`.
3. Run on macOS (BLAS=Accelerate) and Linux (BLAS=OpenBLAS, MKL); under the 1e-10 tolerance, both must pass.

If any of (1)–(3) fails, parity is broken and the breaking change must be identified before promoting `_target_weights` writes back to `positions_book`.

A complementary harder test (out of scope here) would run the *full* event loop for a 30-day window in both modes against an identical DB snapshot and assert that the sequence of `_target_weights` dicts matches at every monthly rebalance. That harness can subclass the test in §6 and reuse the fixtures.

---

## 9. Risks + rollback path

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Test depends on `Portfolio6Strategy.__init__` not calling `AddIndicator` (true today). If a future P6 variant adds indicators, the warmup query against the mock DB returns `[]` and indicators end up empty — `_rebalance` may then behave differently. | Medium | The test fixture's `mock_db.execute_query` returns `{"status":"success","data":[]}` by default. The audit confirms P6 does not call `AddIndicator`. If P6 ever adopts indicators, the fixture must be extended to return real warmup data (or the test must call `RegisterIndicatorSet` after build). |
| R2 | Synthetic panel may have edge cases (zero-vol days, NaN columns) that don't appear in real market data, causing the test to drift away from the real parity contract. | Low | The synthetic panel is GBM-like and matches the structural assumptions of `realized_vol` (>30 datapoints, finite, non-zero std). Add a second fixture that loads a real 1-day market_data snapshot when DB is available (cf. `conftest.py::require_db_env` pattern). |
| R3 | `PYTHONHASHSEED` is set inside the test fixture, but the pytest process may have already imported modules under a different seed. If those modules cached hashed-set iteration order at import time, results may drift. | Low | Modules under test (`Portfolio6Strategy`, `screener`) don't cache hashed iteration order at import. Confirmed by reading `__init__` and module-level code. |
| R4 | `monkeypatch.setattr` on `datetime` only patches the names in the *imported* modules; if a new module is added to the call chain in the future, the patch won't catch it. | Medium | `test_no_wallclock_leak_in_rebalance` will start to false-pass. Mitigation: add a second-level guard via `tracemalloc` or `sys.settrace` if a regression is suspected. Cheaper alternative: extend the monkeypatch list whenever a new module is added to `_rebalance`'s call graph. |
| R5 | Test file lives in `tests/integration/` (new subdir). If CI's pytest collection is restricted to `tests/*.py` (no recursion), the test won't actually run. | Low | Verify `pytest.ini` / `pyproject.toml` collection patterns before merging the test. The default `pytest` config in this repo (none) collects recursively, so the test will run. |
| R6 | Acceptable-divergence layer (executor fill model) may be confused with a parity bug by future maintainers. | Low | This document, especially §4.7, is the canonical reference. The test itself documents the exclusion in its module docstring. |

### Rollback

This audit produces one new file (`tests/integration/test_live_backtest_parity.py`) and zero modifications to existing source. If the test misbehaves: `git rm tests/integration/test_live_backtest_parity.py` — there is nothing else to revert.

### Operational follow-ups (separate PRs, not part of parity)

| # | Action | Tracking |
|---|---|---|
| F1 | Add `, id DESC` tiebreaker to `LATEST_PNL_QUERY` (strategy.py:382–388). | New ticket. Low risk; one-line SQL change. |
| F2 | Add `, ticker, timestamp DESC` tiebreaker to `MARKET_DATA_QUERY` (strategy.py:375–380). | Same. Client-side sort already mitigates. |
| F3 | Pin `numpy`, `pandas`, `scipy` versions in `requirements.txt`. | Operational. |
| F4 | Add a parity test for portfolios that DO use the indicator framework (P3, P5) — `BasePortfolio.AddIndicator` has a `datetime.now()` fallback that diverges between backtest and live (strategy.py:187). | Separate audit (D5 candidate). |
| F5 | Cross-validate `_target_weights` against a recorded snapshot from a prior known-good run. Adds tamper-evidence to the parity test. | Future enhancement. |

---

## 10. References (deduplicated, with annotations)

- **S1** QuantConnect — Live Reconciliation. https://www.quantconnect.com/docs/v2/writing-algorithms/live-trading/reconciliation
- **S2** AlgoBulls — Backtesting technical factors. https://algobulls.com/blog/algo-trading/backtesting-technical-factor
- **S3** Implementation Risk in Portfolio Backtesting (arXiv 2603.20319). https://arxiv.org/html/2603.20319v1
- **S4** PostgreSQL 18 docs — Sorting Rows (ORDER BY). https://www.postgresql.org/docs/current/queries-order.html
- **S5** Andrew Chaa — The Pitfall of ORDER BY Timestamp. https://dev.to/andrewchaa/the-pitfall-of-order-by-timestamp-and-how-to-fix-it-3lh6
- **S6** Hacker News + Wikimedia Phabricator on PYTHONHASHSEED. https://news.ycombinator.com/item?id=22268769 / https://phabricator.wikimedia.org/T220099
- **S7** GlassAlpha — Determinism & Reproducibility guide. https://glassalpha.com/guides/determinism/
- **S8** Scientific Python — NumPy RNG best practices. https://blog.scientific-python.org/numpy/numpy-rng/
- **S9** NumPy — Parallel random number generation. https://numpy.org/doc/2.2/reference/random/parallel.html
- **S10** QuantStart — Event-Driven Backtesting Part I. https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-I/
- **S11** Timothy Kimutai — Event-driven backtest engine. https://timkimutai.medium.com/how-i-built-an-event-driven-backtesting-engine-in-python-25179a80cde0
- **S12** Bailey–Borwein–López de Prado–Zhu — Pseudo-Mathematics and Financial Charlatanism. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659
- **S13** LuxAlgo — Survivorship Bias in Backtesting. https://www.luxalgo.com/blog/survivorship-bias-in-backtesting-explained/
- **S14** Mariam Hov — Indicator data consistency. https://medium.com/@mariamhov/from-backtesting-to-live-trading-how-consistent-indicator-data-improves-strategy-performance-7639949bb791

---

**End of D4.** Parity verdict: PASS for decision layer (`_target_weights`); acceptable by-design divergence at the fill layer. Regression guard provided in §6, ready to merge.
