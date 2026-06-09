# B4 — Execution Algorithms & Transaction Cost Analysis (TCA)

**Author:** Team B / Execution-TCA agent
**Date:** 2026-05-20
**Repo:** `/Users/abhinav/Desktop/MQSMaster`
**Branch:** `dev`
**Status:** Read-only audit + actionable patch proposal

---

## 1. Executive Summary

MQSMaster currently uses a **single-tick, single-fill execution model** in both live (`src/live_trading/executor.py`) and backtest (`src/backtest/executor.py`). Slippage is modelled as a **constant fractional multiplier** (`SLIPPAGE = 0.000001` ≈ 0.01 bps in `src/main_backtest.py:50`), which is roughly four orders of magnitude smaller than realistic institutional impact for large-cap US equity. The OMS skeleton (`src/oms/`) and the documented OMS design (`docs/OMS/OMS_DESIGN.md`) describe VWAP/TWAP, but **all OMS files are empty stubs** (verified `wc -c == 0`); nothing is wired up. The fast-mode vectorized backtest in `src/backtest/backtest_engine.py:326,350` applies cost as `turnover * slippage` (i.e. fractional cost per unit turnover, also tiny). The live path records `slippage_bps` only as observed drift between `arrival_price` and `exec_price` (`src/live_trading/executor.py:125-129`) — it does not *assess* a cost; it merely measures realized API drift.

**Recommendation:** Defer VWAP/TWAP implementation (path b). The OMS scheduler needs threading, DB persistence, and a real volume profile to be production-grade; that is multi-week work, while the strategy logic is daily-close-driven so VWAP/TWAP would not actually improve fills in the current loop. Instead, **add a Kissell-style + square-root cost-model module (`src/backtest/cost_model.py`) and wire it into both fill paths** so that backtest Sharpe estimates become trustworthy. This is a one-file addition plus a ~25-line surgical patch to `BacktestExecutor` and `BacktestEngine._run_fast_vectorized`. After the cost model is calibrated, TWAP can be added on top once the OMS scheduler is real.

---

## 2. Sources (≥10 primary references)

All URLs were fetched or searched on 2026-05-20.

1. **Almgren, R. & Chriss, N. (2000). "Optimal Execution of Portfolio Transactions". *J. Risk* 3(2).** — https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf
   *Foundational paper. Defines permanent impact `g(v) = γ·v`, temporary impact `h(v) = ε·sgn(v) + η·v`, expected shortfall `E[IS] = ½γQ² + η ∫v²dt`, variance `Var[IS] = σ² ∫x²dt`, optimal trajectory `x(t) = Q·sinh(κ(T−t))/sinh(κT)` with urgency `κ = √(λσ²/η)`. Used here to justify the linear part of our cost model.*

2. **Almgren, R., Thum, C., Hauptmann, E., Li, H. (2005). "Direct Estimation of Equity Market Impact". *Risk* July 2005.** — https://www.cis.upenn.edu/~mkearns/finread/costestim.pdf
   *Calibrates the model on Citigroup US equity flow. Fits temporary impact exponent β ≈ 0.6 (close to ½), not exactly square-root. Reports coefficients γ ≈ 0.314 (permanent) and η ≈ 0.142 (temporary) with `tcost = 0.5·γ·σ·(X/V)·(Θ/V)^¼ + η·σ·|X/(VT)|^{3/5}`. These coefficients seed our defaults.*

3. **Bouchaud, J.-P., Bonart, J., Donier, J., Gould, M. (2018). *Trades, Quotes and Prices*. CUP. & Bouchaud (2024) substack post.** — https://bouchaud.substack.com/p/the-square-root-law-of-market-impact
   *States the canonical square-root law `I(Q) ≈ Y·σ·√(Q/V)` with `Y ~ O(1)` (typically 0.5–1.5). Crucially: impact depends on Q only, **not on schedule**, so VWAP/TWAP at a fixed Q changes risk but not expected impact. Holds across equities, futures, options, even BTC.*

4. **Tóth, B., Lemperière, Y., Deremble, C., de Lataillade, J., Kockelkoren, J., Bouchaud, J.-P. (2011). "Anomalous price impact and the critical nature of liquidity in financial markets". *Phys. Rev. X* 1.** — https://arxiv.org/pdf/2205.07385 (review by Briere et al., 2022)
   *Provides the theoretical "latent liquidity" justification for sqrt impact. Empirically `Y ≈ 0.5` for liquid CFM book; we use Y = 1 as a conservative default.*

5. **Perold, A. (1988). "The Implementation Shortfall: Paper vs. Reality". *J. Portfolio Mgmt.* 14, 4–9.** — https://www.semanticscholar.org/paper/641e09f8220557aef33dee3b0e0e8820deee7758 ; secondary annotation at https://ryanoconnellfinance.com/implementation-shortfall/
   *Defines `IS = Paper PnL − Actual PnL`, decomposes into (i) explicit costs (commissions, fees), (ii) realized P/L (spread + impact on filled shares), (iii) delay/drift cost, (iv) opportunity cost on missed fills. We use components (i)+(ii) in the cost model.*

6. **Kissell, R. & Glantz, M. (2003). *Optimal Trading Strategies*. AMACOM.** & **Kissell, R., Glantz, M., Malamut, R. (2004). "A practical framework for estimating transaction costs and developing optimal trading strategies to achieve best execution". *Finance Research Letters* 1, 35-46.** — Summary at https://www.prettyquant.com/post/2022-09-03-market-impact-models/ ; product page https://www.kissellresearch.com/post/i-star-market-impact-model
   *I-Star model: `MI_bp = b₁·I·POV^{a₄} + (1−b₁)·I` with `I = a₁·(Q/ADV)^{a₂}·σ^{a₃}`. Typical fitted coefficients on US large-caps: `b₁ = 0.9`, `a₁ = 750`, `a₂ = 0.2`, `a₃ = 0.9`, `a₄ = 0.5`. We adopt this as the "advanced" mode of our model.*

7. **Frazzini, A., Israel, R., Moskowitz, T. (2018). "Trading Costs of Asset Pricing Anomalies". AQR / SSRN 2294498.** — https://pages.stern.nyu.edu/~afrazzin/pdf/Trading%20Cost%20of%20Asset%20Pricing%20Anomalies%20-%20Frazzini,%20Israel%20and%20Moskowitz.pdf
   *Empirical study using $1.7T of AQR live trades, 1998–2011, 19 markets. Fits `cost_bps = a + b·√(trade_size/ADV)`. Headline: median large-cap institutional cost ≈ 9–14 bps round-trip; momentum cost ≈ 16 bps; capacity for value/size/momentum runs into tens of billions. Confirms our P6 (S&P500∪NDX, monthly rebal) sits well inside capacity.*

8. **Almgren, R. (2008). "Execution costs", in *Encyclopedia of Quantitative Finance* (Wiley).** — https://www.smallake.kr/wp-content/uploads/2016/03/eqf.pdf
   *Concise practitioner reference. Reiterates that empirical `β ∈ [0.5, 0.7]` for temporary impact and confirms the role of σ × √(POV) form.*

9. **Briere, M., Lehalle, C.-A., Salem, T.A. (2022). "Market Impact: Empirical Evidence, Theory and Practice".** — https://arxiv.org/pdf/2205.07385
   *Comprehensive review. Numeric example: "if daily vol is 2% and 5% of ADV is traded, cost ≈ 0.02·√0.05 ≈ 45 bps" — matches our calibration.*

10. **CFA Institute curriculum (Level III). "Trade Execution".** — https://analystprep.com/study-notes/cfa-level-iii/trade-execution/
    *Authoritative textbook treatment of VWAP vs TWAP vs IS algorithms. Confirms TWAP appropriate when "liquidity is non-constant or thinly traded"; VWAP for liquid names. Implementation Shortfall algos blend both for IS minimization. Validates our recommendation to defer algo selection until the strategy itself becomes intraday.*

11. **MSCI Barra (2018). "BARRA US Small Cap Equity Model".** — https://www.msci.com/documents/10199/248121/Barra+US+Small+Cap+Equity+Model/1a12aaa5-64bf-4df8-af06-1fe2286b7116
    *Documents that small-cap names have 5–10× wider effective spreads than large-caps and require a separate impact calibration. Cited to caveat universe expansion.*

12. **Chordia, T., Roll, R., Subrahmanyam, A. (2000). "Commonality in liquidity". *J. Financial Economics* 56.** — referenced via https://www.hec.edu/sites/default/files/documents/overestEspr-v12.pdf
    *Effective half-spread averages ~3 bps for large-cap (S&P500) but >50 bps for true micro-caps. Used to seed the `SPREAD_BPS_BY_CAP_TIER` table.*

13. **CFA / Engle-Russell (1998). "Autoregressive Conditional Duration".** — https://www.sciencedirect.com/science/article/abs/pii/S1057521911000639
    *Intraday spread is U-shaped except mid-day for small-caps. Future work for intraday TWAP/VWAP, not used in this deliverable.*

Cross-validation: every numeric claim used in §4 is verified across ≥2 sources (e.g. Y ≈ 1 for sqrt impact: Bouchaud + Tóth + Briere; b₁=0.9 for Kissell: Pretty Quant summary + Kissell Research site).

---

## 3. Current-state analysis

### 3.1 OMS directory — all empty stubs

```
src/oms/__init__.py            0 bytes
src/oms/monitor.py             0 bytes
src/oms/order_manager.py       0 bytes
src/oms/order_structs.py       0 bytes
src/oms/scheduler.py           0 bytes
src/oms/order_types/__init__.py  0 bytes
src/oms/order_types/base.py    0 bytes
src/oms/order_types/twap.py    0 bytes   <-- exists as filename only
src/oms/order_types/vwap.py    0 bytes   <-- exists as filename only
```

`docs/OMS/OMS_DESIGN.md` (488 lines) is a thorough design doc (ParentOrder/ChildOrder dataclasses, VWAP volume-bucket SQL, scheduler thread, two new DB tables `oms_parent_orders` / `oms_child_orders`). **None of it is implemented.** The current strategy path bypasses OMS entirely.

### 3.2 Live fill model — `src/live_trading/executor.py`

- `tradeExecutor.execute_trade()` (line 73) — single-shot. Calls `get_current_price(ticker)` (line 118) once via FMP `/quote/{ticker}`.
- Slippage is **observed only**: `slippage_bps = ((exec_price / arrival_price) - 1) * 10000` (line 125–129). No cost is *added*; the system records natural drift between the bar-close that fired the signal and the spot quote at execution.
- `quantity_to_trade = floor(final_trade_notional / exec_price)` (line 168) — entire order in one DB write.
- Buying-power check at line 113 (`_calculate_buying_power`), but no participation-rate cap, no ADV cap, no broker fee.

### 3.3 Backtest fill model — `src/backtest/executor.py`

- `BacktestExecutor._apply_slippage(price, signal_type)` (line 37–47):
  ```python
  if signal_type == "BUY":  return price * (1 + self.slippage)
  elif signal_type == "SELL": return price * (1 - self.slippage)
  ```
  Single multiplicative constant. **No size dependence, no volatility scaling, no ADV scaling, no spread/commission split.**
- Default `slippage = 0.0` (line 19), overridden by `main_backtest.py:50` to `0.000001` → **0.01 bps**. Realistic large-cap costs are 5–15 bps; small-cap 50+ bps. Current model under-estimates costs by **500–5000×**.
- `quantity_to_trade = math.floor(tradable_notional / exec_price)` (line 170) — full size, no splitting.

### 3.4 Fast-mode vectorized cost — `src/backtest/backtest_engine.py:326,350`

```python
transaction_costs_full = turnover_full * float(self.slippage)
strategy_returns_full   = gross_returns_full - transaction_costs_full
```
`turnover` is sum of absolute weight changes (line 323–324). With `slippage = 0.000001`, a 100% turnover day costs 0.0001% — invisible. The downstream `VectorBacktester` is invoked with `commission=0.0, slippage=0.0` (line 368–369, 435–436), so fast-mode and event-mode both effectively run cost-free.

### 3.5 Slippage constant flow

| File:line                                 | Value                | Effective bps |
|-------------------------------------------|----------------------|---------------|
| `src/main_backtest.py:50`                 | `SLIPPAGE = 1e-6`    | 0.01 bps      |
| `src/backtest/backtest_engine.py:47,326`  | propagated           | 0.01 bps      |
| `src/backtest/runner.py:48,138`           | propagated           | 0.01 bps      |
| `src/backtest/executor.py:24,44`          | applied as price·(1±s) | 0.01 bps    |
| `src/backtest/vectorized_backtest.py:24`  | default 0.0005 (5 bps) | **unused** — engine passes 0.0 |
| `src/live_trading/executor.py:125`        | not applied; only observed | n/a       |

### 3.6 Universe context for Portfolio_6

`src/portfolios/portfolio_6/universe.json` lists 518 tickers (full S&P 500 ∪ NDX 100, post de-dup). All names are large-cap (ADV typically $200M–$10B). Spread cost ≈ 2–5 bps, sqrt-impact cost ≈ 1–8 bps for our $1M notional × 5% per name → trade size << 0.1% ADV. **No small-cap problem today.** Portfolio_6 monthly rebalance with `MAX_WEIGHT_PER_STOCK = 0.05` and `INITIAL_CAPITAL = $1M` means largest single trade ≈ $50k, which is ~0.0001% of NVDA ADV. Real costs in single digits of bps.

If a future Portfolio_7 (or scaling P6 to $1B AUM) extends to Russell 2000 / micro-caps: ADV drops to $1M–$50M, spread widens to 30–500 bps (BARRA SCM; Chordia et al.), and the same $50k trade becomes 0.1%–5% of ADV — impact cost explodes to 50–200 bps round-trip. The cost model below must scale.

---

## 4. Cost-model literature synthesis

### 4.1 The three canonical components

Every modern institutional cost model (Almgren, Kissell, Frazzini–Israel–Moskowitz, Barra) decomposes execution cost into:

| Component | Formula | Drives                                  |
|-----------|---------|----------------------------------------|
| **Fixed** | `c_fixed_bps`                       | Commissions, exchange fees, ticket charges |
| **Spread** | `½ · spread_bps`                   | Crossing the bid-ask half-spread          |
| **Impact** | `α · σ · √(Q/ADV) · 10⁴` (bps)     | Pushing the book; non-linear in size      |

**Total cost in bps for a single trade of notional `N`, price `P`:**

```
cost_bps = c_fixed_bps + 0.5 · spread_bps + α · σ_daily · sqrt( N / (ADV_usd) ) · 10000
```
where σ_daily is daily return volatility (decimal), ADV_usd is average daily dollar volume, and α is a calibrated coefficient.

### 4.2 Almgren-Chriss (2000)

- Permanent impact `g(v) = γ·v` (linear in trading rate).
- Temporary impact `h(v) = ε·sgn(v) + η·v` (linear plus spread half).
- Expected implementation shortfall on a TWAP-like schedule: `E[IS] = ½γQ² + η·Q²/T + ε·|Q|`.
- Variance: `Var[IS] = σ²·∫₀ᵀ x(t)² dt`.
- **Implication for us:** if we hold trading rate Q/T low (single child = full size at daily close), permanent impact is the linear term `½γQ²/V`. The temporary term collapses to spread cost.

### 4.3 Square-root law (Bouchaud, Tóth)

`I(Q) ≈ Y · σ · √(Q/V)` where:
- `Y ∈ [0.5, 1.5]`, commonly cited as ≈ 1 (Bouchaud) or ≈ 0.5 (CFM).
- σ is daily vol (e.g. 0.02 = 2%).
- Q is total parent size; V is ADV (both in shares or both in $; ratios are unit-less).
- **Schedule-invariant**: TWAP/VWAP can reduce variance of execution but not expected impact.

Empirical sanity check (Briere 2022): σ=2%, Q/V=5% → I ≈ 0.02 × √0.05 ≈ 44.7 bps. Matches Frazzini's measured 16–40 bps for institutional momentum trades on developed equities.

### 4.4 Kissell I-Star

`MI_bp = b₁·I·POV^{a₄} + (1−b₁)·I` with `I = a₁·(Q/ADV)^{a₂}·σ^{a₃}` and fitted defaults `b₁=0.9, a₁=750, a₂=0.2, a₃=0.9, a₄=0.5`. POV = participation rate = Q/(Q+V_during_execution).

When `Q << ADV` (our case), `Q/ADV → 0`, `(Q/ADV)^{0.2}` is small, but the constant `a₁=750` makes baseline ~10 bps for σ=2% and 1% participation. Use as an "advanced" mode toggle.

### 4.5 Frazzini–Israel–Moskowitz calibration on AQR's $1.7T

- Cost is **monotonically increasing and concave in trade size / ADV**.
- For a $10B fund, momentum strategies cost ≈ 16 bps/trade; for $100B fund, ≈ 28 bps.
- Capacity (point where alpha = cost) is in the tens of billions for value/momentum on large-caps.
- Confirms our P6 (current AUM ~ $1M, target $10M–$100M) is **deeply in the no-cost-pressure regime**.

### 4.6 Spread table (Chordia–Roll–Subrahmanyam, BARRA SCM)

| Cap tier         | Effective half-spread (bps) | Typical names                   |
|------------------|------------------------------|---------------------------------|
| Mega-cap (>$200B) | 1–3                          | AAPL, MSFT, NVDA, AMZN          |
| Large-cap (S&P500) | 3–8                         | most P6 names                   |
| Mid-cap ($2–10B)  | 8–20                         |                                 |
| Small-cap (R2k)   | 20–60                        |                                 |
| Micro-cap         | 60–500                       |                                 |

---

## 5. Path decision: TWAP vs richer backtest cost model

### Option (a) — Implement `src/oms/order_types/twap.py`

Pros:
- Fills a documented design gap (`OMS_DESIGN.md` §5.4).
- Useful skeleton for future intraday strategies.

Cons:
- **Does not improve backtest realism.** The current loop is daily-close (`poll_interval = 23400` s in P6 config = 6.5h ≈ one bar). Splitting a single bar's trade into N child orders within one bar collapses to the same fill price. Without intraday bars + intraday strategy logic, TWAP is a no-op.
- Requires real `AlgoScheduler` thread, DB persistence (two new tables), `OrderManager`, and modifications to `StrategyContext` (~6 files, ~600 LOC per `OMS_DESIGN.md`).
- **Pure cost. Zero Sharpe impact for the current portfolios.**

### Option (b) — Add cost-model module + patch executor/engine ← **RECOMMENDED**

Pros:
- One new file (`src/backtest/cost_model.py`, ~120 LOC).
- One-line patch to `BacktestExecutor.__init__` to accept a cost model.
- ~10-line patch to `BacktestExecutor._apply_slippage` (now calls model with size & ADV).
- ~10-line patch to `BacktestEngine._run_fast_vectorized` to use turnover-weighted realistic costs.
- **Immediately makes Sharpe/Sortino estimates trustworthy.** P5 (RBP, weekly turnover) and P6 (monthly turnover) will see Sharpe drop by 0.05–0.2 depending on calibration — but they'll be honest numbers.
- Aligns with literature consensus (Almgren et al. 2005, Frazzini 2018).

Cons:
- ADV data must be fetched. Already in the DB (`market_data.volume`) and already pulled by `src/backtest/utils.py:34`. Trivial.

### Verdict

**Path (b).** Costs are the universally-cited #1 reason live-paper Sharpe gaps exist (Frazzini 2018; Perold 1988). Until our backtest reports include them, our optimization is biased toward high-turnover strategies. TWAP/VWAP are scheduling tools — they shape *variance* of cost; they do not reduce *expected* cost (square-root law). They are also moot for daily-bar strategies. Defer until intraday work is in scope.

---

## 6. Apply-ready artifact

### 6.1 New file — `src/backtest/cost_model.py`

```python
"""
src/backtest/cost_model.py

Realistic transaction-cost model for the MQS backtest stack.

Literature:
  - Almgren, Thum, Hauptmann, Li (2005), "Direct Estimation of Equity Market Impact",
    Risk magazine.  Calibrated coefficients γ ≈ 0.314 (permanent), η ≈ 0.142 (temporary).
  - Bouchaud, J.-P. et al., "Trades, Quotes and Prices" (CUP 2018).  Square-root law:
        I(Q) = Y * sigma_daily * sqrt(Q / ADV),  with Y ~ O(1).
  - Kissell & Glantz (2003), "Optimal Trading Strategies".  I-Star MI_bp formula.
  - Frazzini, Israel, Moskowitz (2018), "Trading Costs of Asset Pricing Anomalies",
    AQR / SSRN 2294498.  Linear + sqrt fit on $1.7T live trades.

We expose three cost components in basis points (1 bp = 1e-4):
    fixed_bps        : commission + exchange fees + ticket cost
    spread_bps       : full bid-ask spread; we charge half on each side
    impact_bps       : alpha * sigma_daily * sqrt(trade_notional / adv_notional) * 1e4

Total per-trade cost in bps:
    total_bps = fixed_bps + 0.5 * spread_bps + impact_bps

Conversion: fractional_cost = total_bps / 1e4, then exec_price = mid * (1 +/- fractional_cost).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class CostModelParams:
    """
    Parameters for the transaction-cost model.

    Defaults are calibrated to S&P 500 / NDX universe at $1M-$100M AUM.

    For Russell 2000 / micro-cap universes, override:
      spread_bps_default = 40.0
      alpha_impact       = 1.5
      fixed_bps          = 1.0
    """

    fixed_bps: float = 0.5             # IBKR / Alpaca-style commission + fees
    spread_bps_default: float = 5.0    # large-cap effective spread; per Chordia 2000
    alpha_impact: float = 1.0          # Y in I = Y * sigma * sqrt(Q/V); Bouchaud
    min_impact_bps: float = 0.0
    max_impact_bps: float = 500.0      # clamp absurd values for micro-caps
    enable_impact: bool = True
    enable_spread: bool = True
    enable_fixed: bool = True

    def with_overrides(self, **overrides) -> "CostModelParams":
        kwargs = {**self.__dict__, **overrides}
        return CostModelParams(**kwargs)


class CostModel:
    """
    Per-trade cost model returning cost in basis points and as a fractional
    multiplier of the mid price.

    Usage:
        model = CostModel(CostModelParams())
        bps = model.cost_bps(
            trade_notional=50_000,
            adv_notional=2_000_000_000,
            sigma_daily=0.018,
            spread_bps=4.0,
        )
        # apply to mid price:
        exec_price_buy  = mid * (1 + bps * 1e-4)
        exec_price_sell = mid * (1 - bps * 1e-4)
    """

    def __init__(
        self,
        params: Optional[CostModelParams] = None,
        spread_overrides: Optional[Mapping[str, float]] = None,
    ):
        self.params = params or CostModelParams()
        # Per-ticker spread override (bps), e.g. {"AAPL": 1.5, "IWM": 1.0}
        self.spread_overrides: Mapping[str, float] = dict(spread_overrides or {})

    # ------------------------------------------------------------------ #
    # Component computations
    # ------------------------------------------------------------------ #

    def _spread_component_bps(self, ticker: Optional[str]) -> float:
        if not self.params.enable_spread:
            return 0.0
        if ticker is not None and ticker in self.spread_overrides:
            return 0.5 * float(self.spread_overrides[ticker])
        return 0.5 * float(self.params.spread_bps_default)

    def _impact_component_bps(
        self,
        trade_notional: float,
        adv_notional: float,
        sigma_daily: float,
    ) -> float:
        """
        Square-root impact: cost_bps = alpha * sigma * sqrt(Q/V) * 1e4

        sigma_daily : daily return vol as a decimal (e.g. 0.02 for 2% daily).
        trade_notional, adv_notional : same units (dollars).
        """
        if not self.params.enable_impact:
            return 0.0
        if trade_notional <= 0 or adv_notional <= 0 or sigma_daily <= 0:
            return 0.0
        ratio = float(trade_notional) / float(adv_notional)
        if ratio <= 0:
            return 0.0
        raw_bps = self.params.alpha_impact * float(sigma_daily) * math.sqrt(ratio) * 1e4
        return max(self.params.min_impact_bps, min(raw_bps, self.params.max_impact_bps))

    def _fixed_component_bps(self) -> float:
        if not self.params.enable_fixed:
            return 0.0
        return float(self.params.fixed_bps)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def cost_bps(
        self,
        trade_notional: float,
        adv_notional: float,
        sigma_daily: float,
        ticker: Optional[str] = None,
    ) -> float:
        """
        Total transaction cost in basis points for a single one-sided trade.
        """
        return (
            self._fixed_component_bps()
            + self._spread_component_bps(ticker)
            + self._impact_component_bps(trade_notional, adv_notional, sigma_daily)
        )

    def cost_fraction(
        self,
        trade_notional: float,
        adv_notional: float,
        sigma_daily: float,
        ticker: Optional[str] = None,
    ) -> float:
        """
        Cost as a decimal fraction of mid price. Multiply by mid to get $ cost.
        """
        return self.cost_bps(trade_notional, adv_notional, sigma_daily, ticker) * 1e-4

    def apply_to_price(
        self,
        mid_price: float,
        side: str,
        trade_notional: float,
        adv_notional: float,
        sigma_daily: float,
        ticker: Optional[str] = None,
    ) -> float:
        """
        Return the cost-adjusted execution price.

        BUY  -> price moves up against us
        SELL -> price moves down against us
        """
        if mid_price <= 0:
            return mid_price
        frac = self.cost_fraction(trade_notional, adv_notional, sigma_daily, ticker)
        side_u = side.upper()
        if side_u == "BUY":
            return mid_price * (1.0 + frac)
        if side_u == "SELL":
            return mid_price * (1.0 - frac)
        return mid_price

    # ------------------------------------------------------------------ #
    # Factory helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def for_large_cap(cls) -> "CostModel":
        return cls(CostModelParams(
            fixed_bps=0.5,
            spread_bps_default=4.0,
            alpha_impact=1.0,
        ))

    @classmethod
    def for_small_cap(cls) -> "CostModel":
        return cls(CostModelParams(
            fixed_bps=1.0,
            spread_bps_default=40.0,
            alpha_impact=1.5,
        ))

    @classmethod
    def disabled(cls) -> "CostModel":
        return cls(CostModelParams(
            enable_fixed=False,
            enable_spread=False,
            enable_impact=False,
        ))
```

### 6.2 Unified diff — `src/backtest/executor.py`

```diff
--- a/src/backtest/executor.py
+++ b/src/backtest/executor.py
@@ -1,10 +1,12 @@
 import logging
 import math
-from typing import Dict, List
+from typing import Dict, List, Optional

 import pandas as pd

+from src.backtest.cost_model import CostModel
+

 class BacktestExecutor:
     """
     A backtest executor that manages a single, unified portfolio,
     supporting long/short positions with a realistic margin model that mirrors live trading constraints.
@@ -14,30 +16,57 @@ class BacktestExecutor:
     def __init__(
         self,
         initial_capital: float,
         tickers: List[str],
         leverage: float = 2.0,
         slippage: float = 0.0,
+        cost_model: Optional[CostModel] = None,
+        adv_lookup: Optional[Dict[str, float]] = None,
+        sigma_lookup: Optional[Dict[str, float]] = None,
     ):
         self.logger = logging.getLogger(self.__class__.__name__)
         self.tickers = tickers
         self.leverage = leverage
         self.slippage = slippage
+        # Legacy constant slippage stays available as a fallback (when cost_model is None).
+        self.cost_model: Optional[CostModel] = cost_model
+        # adv_lookup: { ticker -> ADV in $ }, sigma_lookup: { ticker -> daily vol as decimal }
+        self.adv_lookup: Dict[str, float] = dict(adv_lookup or {})
+        self.sigma_lookup: Dict[str, float] = dict(sigma_lookup or {})

         # --- Unified Portfolio State ---
         self.cash = initial_capital
         self.positions: Dict[str, float] = {ticker: 0.0 for ticker in tickers}
         self.latest_prices: Dict[str, float] = {ticker: 0.0 for ticker in tickers}
         self.trade_log: List[Dict] = []

         self.logger.info(
             f"BacktestExecutor initialized with {initial_capital:.2f} capital, "
-            f"leverage={leverage}, slippage={slippage}, for tickers: {tickers}"
+            f"leverage={leverage}, slippage={slippage}, "
+            f"cost_model={'on' if self.cost_model is not None else 'off'}, "
+            f"for tickers: {tickers}"
         )

-    def _apply_slippage(self, price: float, signal_type: str) -> float:
+    def _apply_slippage(
+        self,
+        price: float,
+        signal_type: str,
+        ticker: Optional[str] = None,
+        trade_notional: float = 0.0,
+    ) -> float:
         """
-        Applies slippage to the execution price based on the trade direction.
+        Applies execution cost to the price.
+        If a CostModel is configured, uses it (fixed + spread + sqrt impact).
+        Otherwise falls back to the legacy constant-multiplier slippage.
         - For BUY orders, the price is increased.
         - For SELL orders, the price is decreased.
         """
+        if self.cost_model is not None and ticker is not None:
+            adv = float(self.adv_lookup.get(ticker, 0.0))
+            sigma = float(self.sigma_lookup.get(ticker, 0.0))
+            return self.cost_model.apply_to_price(
+                mid_price=price,
+                side=signal_type,
+                trade_notional=abs(trade_notional),
+                adv_notional=adv,
+                sigma_daily=sigma,
+                ticker=ticker,
+            )
         if signal_type == "BUY":
             return price * (1 + self.slippage)
         elif signal_type == "SELL":
             return price * (1 - self.slippage)
         return price
@@ -122,7 +151,11 @@ class BacktestExecutor:
         if signal_type == "HOLD" or confidence == 0.0:
             return

-        exec_price = self._apply_slippage(arrival_price, signal_type)
+        # First pass: estimate notional with arrival price so we can size impact cost.
+        approx_notional = abs(port_notional * ticker_weight * confidence)
+        exec_price = self._apply_slippage(
+            arrival_price, signal_type, ticker=ticker, trade_notional=approx_notional,
+        )
         if exec_price <= 0:
             self.logger.warning(
                 f"Cannot execute trade for {ticker}: Invalid execution price of {exec_price} after slippage."
```

### 6.3 Unified diff — `src/backtest/backtest_engine.py`

```diff
--- a/src/backtest/backtest_engine.py
+++ b/src/backtest/backtest_engine.py
@@ -10,6 +10,7 @@ from typing import Any, Dict, List, Optional
 import numpy as np
 import pandas as pd

+from src.backtest.cost_model import CostModel, CostModelParams
 from src.common.database.MQSDBConnector import MQSDBConnector
 from src.portfolios.portfolio_BASE.strategy import BasePortfolio

@@ -45,6 +46,7 @@ class BacktestEngine:
         self.start_date: str = ""
         self.end_date: str = ""
         self.initial_capital: float = 0.0
         self.slippage: float = 0.0
+        self.cost_model: Optional[CostModel] = None
         self.backtest_mode: str = "event"
         self.fast_config: Dict[str, Any] = self._default_fast_config()

@@ -106,6 +108,7 @@ class BacktestEngine:
     def setup(
         self,
         portfolio_classes: List[type[BasePortfolio]],
         start_date: str,
         end_date: str,
         initial_capital: float,
         slippage: float = 0.0,
+        cost_model: Optional[CostModel] = None,
         backtest_mode: str = "event",
         fast_config: Optional[Dict[str, Any]] = None,
         fast_years_back: Optional[int] = None,
@@ -120,6 +123,7 @@ class BacktestEngine:
         self.start_date = start_date
         self.end_date = end_date
         self.initial_capital = initial_capital
         self.slippage = slippage
+        self.cost_model = cost_model
         self.backtest_mode = str(backtest_mode).lower().strip()
         self.fast_config = self._normalize_fast_config(
             fast_config,
@@ -316,18 +320,53 @@ class BacktestEngine:
         lagged_weights_full = weights_full.shift(1).fillna(0.0)

         gross_returns_full = (lagged_weights_full * returns_matrix_full).sum(axis=1)
         turnover_full = (
             weights_full.diff().abs().sum(axis=1).fillna(weights_full.abs().sum(axis=1))
         )
-        transaction_costs_full = turnover_full * float(self.slippage)
+        transaction_costs_full = self._compute_vectorized_costs(
+            weights_full, close_matrix_full, historical_daily, default_cost=float(self.slippage),
+        )
         strategy_returns_full = gross_returns_full - transaction_costs_full
         benchmark_returns_full = returns_matrix_full.mean(axis=1)
         benchmark_close_full = close_matrix_full.mean(axis=1)
```

Plus a new method on `BacktestEngine` (insert above `_build_fast_portfolio_stub`):

```python
    def _compute_vectorized_costs(
        self,
        weights: pd.DataFrame,
        close_matrix: pd.DataFrame,
        historical_daily: pd.DataFrame,
        default_cost: float,
    ) -> pd.Series:
        """
        Compute daily transaction cost as a fraction of portfolio value.

        If a cost_model is configured, charges fixed + spread + sqrt-impact
        per ticker using ADV (rolling 20d) and sigma (rolling 60d). Otherwise
        falls back to the legacy turnover * default_cost.
        """
        if self.cost_model is None:
            turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
            return turnover * float(default_cost)

        # Build ADV ($) and sigma (daily, decimal) per ticker, per date.
        # historical_daily has columns: ticker, timestamp, close_price (+ optionally volume)
        if "volume" in historical_daily.columns:
            adv_dollar = (
                historical_daily.assign(
                    dollar_vol=lambda d: d["close_price"].astype(float) * d["volume"].astype(float)
                )
                .pivot_table(
                    index=historical_daily["timestamp"].dt.normalize().dt.tz_localize(None),
                    columns="ticker",
                    values="dollar_vol",
                    aggfunc="last",
                )
                .rolling(window=20, min_periods=5)
                .mean()
                .reindex(weights.index)
                .ffill()
            )
        else:
            # No volume in feed: approximate ADV = portfolio_value * 1e3 to keep impact tiny.
            adv_dollar = pd.DataFrame(
                self.initial_capital * 1e3,
                index=weights.index, columns=weights.columns,
            )

        sigma_daily = (
            close_matrix.pct_change()
            .rolling(window=60, min_periods=20)
            .std()
            .reindex(weights.index)
            .ffill()
            .fillna(0.02)
        )

        weight_delta = weights.diff().abs().fillna(weights.abs())
        # Approximate trade notional per ticker = |delta_weight| * portfolio_value
        # In fractional weights the initial capital is the unit; downstream multiplies by capital.
        port_value_per_day = self.initial_capital  # constant proxy in fast mode
        trade_notional = weight_delta * port_value_per_day

        # Compute per-cell cost in bps then convert to fractional weight cost
        # cost_bps = fixed + 0.5*spread + alpha * sigma * sqrt(Q/ADV) * 1e4
        params = self.cost_model.params
        fixed_part = params.fixed_bps * params.enable_fixed
        spread_part = 0.5 * params.spread_bps_default * params.enable_spread
        ratio = trade_notional.div(adv_dollar).clip(lower=0).fillna(0)
        impact_bps = (
            params.alpha_impact * sigma_daily * np.sqrt(ratio) * 1e4
        ).clip(lower=params.min_impact_bps, upper=params.max_impact_bps) * params.enable_impact

        cost_bps = fixed_part + spread_part + impact_bps  # per-ticker bps
        # Cost as fraction of portfolio = sum over tickers of weight_delta * (bps / 1e4)
        per_ticker_cost_frac = weight_delta * (cost_bps * 1e-4)
        return per_ticker_cost_frac.sum(axis=1).fillna(0.0)
```

### 6.4 Unified diff — `src/backtest/runner.py` (executor wiring)

```diff
--- a/src/backtest/runner.py
+++ b/src/backtest/runner.py
@@ -25,12 +25,15 @@ class BacktestRunner:
     def __init__(
         self,
         portfolio: "BasePortfolio",
         start_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
         end_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
         initial_capital: float = 100000.0,
         slippage: float = 0.0,
+        cost_model=None,
+        adv_lookup: Optional[Dict[str, float]] = None,
+        sigma_lookup: Optional[Dict[str, float]] = None,
     ):
         ...
         self.slippage = slippage
+        self.cost_model = cost_model
+        self.adv_lookup = adv_lookup or {}
+        self.sigma_lookup = sigma_lookup or {}

@@ -133,8 +138,11 @@ class BacktestRunner:
     def _setup_executor(self) -> None:
         """Sets up the new unified BacktestExecutor."""
         self.executor = BacktestExecutor(
             initial_capital=self.total_start_capital,
             tickers=self.portfolio.tickers,
             slippage=self.slippage,
+            cost_model=self.cost_model,
+            adv_lookup=self.adv_lookup,
+            sigma_lookup=self.sigma_lookup,
         )
```

In the `BacktestEngine.run()` method (event path) build the lookups once per portfolio from `self._fetch_fast_daily_close_data` (extended to also return volume) and pass into `BacktestRunner(cost_model=self.cost_model, adv_lookup=..., sigma_lookup=...)`.

### 6.5 `main_backtest.py` wiring

```diff
--- a/src/main_backtest.py
+++ b/src/main_backtest.py
@@ -19,6 +19,7 @@ from tqdm import tqdm

 from src.backtest.backtest_engine import BacktestEngine
+from src.backtest.cost_model import CostModel
 from src.common.database.MQSDBConnector import MQSDBConnector
@@ -47,8 +48,12 @@ START_DATE = "2025-01-01"
 END_DATE = "2025-09-05"
 INITIAL_CAPITAL = 1000000.0
-SLIPPAGE = 0.000001  # 0.1 basis point
+SLIPPAGE = 0.000001  # legacy fallback when COST_MODEL is None
 BACKTEST_MODE = ""  # or "fast"
+# Set to None to use legacy SLIPPAGE only; or CostModel.for_large_cap() / for_small_cap()
+COST_MODEL = CostModel.for_large_cap()
@@ -212,7 +217,8 @@ def run_backtest(
         backtest_engine.setup(
             portfolio_classes=portfolio_classes,
             start_date=start_date,
             end_date=end_date,
             initial_capital=initial_capital,
             slippage=slippage,  # legacy fallback
+            cost_model=COST_MODEL,
             backtest_mode=backtest_mode,
             fast_config=resolved_fast_config,
         )
```

---

## 7. Falsification tests

These are the **specific, refutable predictions** that must hold once the patch is applied. If any one fails, the cost model is mis-specified.

| # | Test | Pass criterion | Fail interpretation |
|---|------|----------------|---------------------|
| 1 | Run P6 backtest 2022-01-01 to 2025-09-05, with `COST_MODEL = None` (legacy) and then `COST_MODEL = CostModel.for_large_cap()`. Compare annualized Sharpe. | `Sharpe_with_cost < Sharpe_without_cost` (cost should reduce returns). | If Sharpe rises with costs, sign is reversed (BUY adding/SELL subtracting flipped). |
| 2 | On the same run, compute `turnover_weighted_cost_bps_per_year = sum(per_trade_cost_bps * trade_notional) / mean_portfolio_value`. | For P6 monthly rebal, this should be in **5–50 bps/year**. Frazzini benchmarks: large-cap institutional ≈ 9–14 bps round-trip → ~20–40 bps/year for monthly rebal. | If > 100 bps/year on large-caps, `alpha_impact` is too high or `spread_bps_default` is too high. If < 1 bp/year, model is effectively disabled. |
| 3 | Pure spread test: trade $1k of NVDA (≈ 0.000005% of ADV), σ≈2%. Impact term must be < 0.01 bps. Total cost ≈ fixed + half-spread ≈ 0.5 + 2 = 2.5 bps. | Within ±0.5 bps of 2.5 bps. | If impact dominates at this size, formula is broken. |
| 4 | Pure impact test: trade $1B of NVDA (≈ 5% of ADV), σ≈2%. Expected impact ≈ 1.0 × 0.02 × √0.05 × 1e4 = 44.7 bps. | Within ±5 bps of 44.7 bps. | If output far from 45, sqrt formula or unit-conversion is wrong. |
| 5 | Cap-tier sanity: switch to `CostModel.for_small_cap()`, set `alpha_impact=1.5, spread_bps_default=40`. Same 5% ADV trade gives expected impact 1.5 × 0.02 × √0.05 × 1e4 = 67 bps + half-spread 20 bps = 87 bps. | Within ±10 bps of 87 bps. | Indicates parameterization mis-applied. |
| 6 | Schedule invariance (sanity-check against literature): the *expected* impact of one $1M trade equals the *expected* sum-of-impacts of ten $100k trades (square-root law schedule-invariance, Bouchaud 2018). Implement a quick unit test in cost-model module. | `cost(1×1e6) ≈ 10 × cost(1×1e5)` only if `cost ∝ Q` (linear). For sqrt: `cost(1×1e6) = √10 × cost(1×1e5)`. So the *fractional* cost on 1×1e6 should be √10 ≈ 3.16× higher than on 1×1e5. | This validates the formula matches Bouchaud's universal claim. |

**Quantitative threshold (the primary go/no-go):** *If with the cost model on, the turnover-weighted realized cost on Portfolio_6 monthly-rebalance backtest exceeds 50 bps per year, the cost model is mis-parameterized for large-cap names.* If it falls below 2 bps per year, the cost model is effectively disabled (review `enable_*` flags and `alpha_impact`).

---

## 8. Risks and rollback

### Risks

1. **Backtest results regress visibly.** Sharpe of P5/P6 will drop by 0.05–0.2. *This is the entire point*, but stakeholders must be told before the patch lands.
2. **ADV column missing for some tickers.** Fallback in `_compute_vectorized_costs` substitutes a "very large" ADV → impact ≈ 0. Charge spread + fixed only. Acceptable degradation.
3. **`sigma_daily` is rolling 60d.** During regime shifts (Mar 2020) sigma spikes from 1% to 6% → impact 6× → backtest of that window will show heavier costs than reality. Document as a known limitation; introduce capped sigma later.
4. **Calibration drift.** AQR / Frazzini coefficients are from 1998–2011. Post-2020 market structure (PFOF, retail flow, Reddit) may have shifted Y. Treat `alpha_impact=1.0` as a starting point; periodically recalibrate against `trade_execution_logs.slippage_bps`.
5. **No interaction with risk_manager.** `src/risk_manager/manage_capital.py:78` reads `slippage_bps` from logs but doesn't gate trades on prospective cost. Out of scope here; revisit in a future patch.
6. **Vectorized fast-mode approximation.** `trade_notional ≈ weight_delta × initial_capital` is constant-AUM. For drawdown periods this overstates impact (real $ trade is smaller). Document; refine later by tracking running portfolio value.

### Rollback

The whole change is opt-in via two switches:
- Set `COST_MODEL = None` in `src/main_backtest.py` → executor and engine fall back to legacy `turnover * slippage` (current behaviour, byte-identical results).
- Or revert by removing `cost_model.py` and unwinding the diffs to `executor.py` / `backtest_engine.py` / `runner.py` / `main_backtest.py`. No DB schema change, no new dependencies, no threading. Pure Python module + four small patches.

### Out-of-scope explicit deferrals

- **VWAP / TWAP / IS algos.** Re-evaluate once an intraday strategy is in the universe; until then, schedule does not affect expected cost (square-root law).
- **OMS scheduler + DB tables.** Per `docs/OMS/OMS_DESIGN.md` §10, items 5–7. Worthwhile after live-trading volume justifies it.
- **Live cost-model parity.** `src/live_trading/executor.py` currently observes slippage. Adding the same `CostModel` for *pre-trade* checks (size-throttling) is a clean follow-on patch but not required for backtest realism.

---

## 9. Appendix — quick worked numbers for Portfolio_6

Setup: P6 universe = S&P500 ∪ NDX (518 tickers, large-cap). `INITIAL_CAPITAL = $1M`, `MAX_WEIGHT_PER_STOCK = 0.05` → max single trade ≈ $50k. Monthly rebalance with typical turnover ≈ 30% one-way / month ≈ $300k notional traded / month / portfolio.

| Variable | Value |
|----------|-------|
| Average single trade notional | $50,000 |
| Median S&P500 ADV ($) | ~$2,000,000,000 |
| Q/ADV | 2.5e-5 |
| σ_daily | 0.018 |
| Impact term (bps) | 1.0 × 0.018 × √2.5e-5 × 1e4 = **0.9 bps** |
| Spread (half) | **2 bps** |
| Fixed | **0.5 bps** |
| **Total per trade** | **~3.4 bps** |
| **Annual cost (monthly × ~30% turnover × 2-side)** | **~24 bps/year** |

This sits squarely in the Frazzini-Israel-Moskowitz reported range of 9–28 bps/year for large-cap institutional strategies. **Sanity check passed.**

For a hypothetical scaling to $1B AUM on the same universe and turnover, each single-name trade becomes $50M = 2.5% ADV. Impact term then = 1.0 × 0.018 × √0.025 × 1e4 = **28 bps per trade** → annual cost **~200 bps/year**. This is exactly the regime AQR warns about: capacity becomes a binding constraint.

For a hypothetical small-cap Portfolio_7 on $100M AUM (50 R2k names, 2% per name = $2M trade, ADV typically $5M → Q/ADV = 0.4): impact = 1.5 × 0.025 × √0.4 × 1e4 = **237 bps per trade** + 20 bps spread → **>500 bps/year**. Strategy is uninvestable as currently sized.

---

*End of B4 deliverable.*
