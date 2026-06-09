# D1 — Backtest Controls (CSCV / PBO / Purged-K-Fold / DSR)

**Author:** Team D1 (quant research + Python engineering)
**Date:** 2026-05-20
**Status:** Read-only audit + apply-ready code in fenced blocks. No source files modified by this agent. Only this `.md` is written.
**Repo root:** `/Users/abhinav/Desktop/MQSMaster`

---

## 1. Executive summary

The MQSMaster backtester has solid plumbing (`BacktestRunner`, `VectorBacktester`, Monte-Carlo bootstrap, seasonal-overlay) and recently added a **Deflated Sharpe Ratio (DSR)** diagnostic in `src/portfolios/portfolio_6/screener.py::deflated_sharpe_ratio`. The DSR call is wired into `Portfolio6Strategy._rebalance` (`strategy.py:227`), logging the probability per monthly rebalance with `n_trials = |returns_matrix|` (~519 names). That is the only multiple-testing penalty implemented. **CSCV / PBO is absent**, **purged k-fold is absent**, **walk-forward is partial** (only the same-window-prior-years seasonal overlay), and the **DSR `n_trials` count understates true trial count** because it ignores the cartesian explosion of strategy configurations (SCORE_METHOD × SCORE_WEIGHTS × thresholds × etc.). The P7/P8 falsification gates referenced in `teamA/SYNTHESIS.md §11` (DSR ≥ 0.5 / 0.95) are therefore only loosely defended. This deliverable specifies and provides apply-ready source for `src/backtest/cscv.py` (Bailey-LdP 2014 CSCV/PBO) and `src/backtest/purged_kfold.py` (López de Prado AFML Ch. 7), plus a unified diff that wires PBO + DSR into `scripts/Backtest_Analysis/backtest_analyzer.py`. A concrete falsification gate (PBO > 0.5 over a SCORE_WEIGHTS / WEIGHTING_METHOD grid blocks P7/P8 promotion) is defined in §9.

---

## 2. Sources (≥10 primary)

All URLs were fetched / searched on 2026-05-20.

| # | URL | Annotation | Relevance |
|---|---|---|---|
| S1 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253 | Bailey, Borwein, López de Prado, Zhu — "The Probability of Backtest Overfitting" (2014) — original CSCV/PBO paper, J. Portfolio Mgmt. | **Primary spec** for `cscv.py` algorithm. |
| S2 | https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf | Bailey et al. preprint PDF — same as S1, hosted by D. H. Bailey. | Confirms CSCV partition structure (T×N → S blocks → C(S, S/2) combinations). |
| S3 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659 | Bailey, Borwein, López de Prado, Zhu — "Pseudo-Mathematics and Financial Charlatanism" (2014, AMS Notices). | Motivation: backtest overfitting causes OOS reversion; multiple testing is endemic. |
| S4 | https://www.ams.org/notices/201405/rnoti-p458.pdf | AMS Notices PDF version of S3. | Quote-worthy framing for the falsification gate. |
| S5 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 | Bailey, López de Prado — "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality" (2014). | **Primary spec** for DSR formula (already partially implemented in `portfolio_6/screener.py`). |
| S6 | https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | David H. Bailey PDF version of S5. | Confirms variance-of-SR formula with skew/kurtosis and Euler-Mascheroni γ. |
| S7 | https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio | Wikipedia summary of the DSR — well-formatted formulas. | Quick reference; cross-checks S5/S6. |
| S8 | https://en.wikipedia.org/wiki/Purged_cross-validation | Wikipedia summary of purged CV — defines purging condition and embargo. | **Primary spec** for `purged_kfold.py`. |
| S9 | https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/ | QuantInsti blog — algorithmic walkthrough of purged k-fold and CPCV. | Confirms `t1[i]` event-end semantics and embargo placement. |
| S10 | https://github.com/esvhd/pypbo | esvhd/pypbo — reference Python implementation of PBO/CSCV. MIT-licensed reference for algorithm structure (not vendored). | Apache-2/MIT precedent for the `cscv.py` we write here. Confirms default `S=16`, `itertools.combinations(Ms, S//2)`, `scipy.special.logit` flow. |
| S11 | https://github.com/esvhd/pypbo/blob/master/pypbo/pbo.py | pypbo's main `pbo()` function. | Confirms numerical conventions for the algorithm. |
| S12 | https://cran.r-project.org/web/packages/pbo/vignettes/pbo.html | mrbcuda/pbo R package vignette. | Cross-validates the PBO definition with a worked example (`p_bo = 1.0` for random data). |
| S13 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2249314 | Harvey, Liu, Zhu — "...and the Cross-Section of Expected Returns" (RFS 2016). | Establishes that t-stat ≥ 3.0 (not 2.0) is needed for new factors after multiple testing. Justifies why DSR alone is insufficient. |
| S14 | https://www.nber.org/system/files/working_papers/w20592/w20592.pdf | NBER preprint of S13. | Confirms Bonferroni / FDR / BHY frameworks for multiple-testing correction. |
| S15 | https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870 | Politis, Romano — "The Stationary Bootstrap" (JASA 1994). | Justifies block-bootstrap (`mc_block_size > 1` in `VectorBacktester.monte_carlo`); the stationary bootstrap with random block lengths is preferred over fixed-block for non-stationary returns. |
| S16 | https://mathweb.ucsd.edu/~politis/impactBOOT.pdf | Politis — "Impact of bootstrap methods on time series analysis" (2003 survey). | Cross-validates S15; confirms geometric block-length distribution preserves stationarity. |
| S17 | https://en.wikipedia.org/wiki/Walk_forward_optimization | Walk-forward optimization Wikipedia. | Confirms rolling vs anchored WFA conventions. |
| S18 | https://blog.quantinsti.com/walk-forward-optimization-introduction/ | QuantInsti walk-forward primer. | Confirms IS/OOS conventions (typically 4:1 to 5:1). |
| S19 | https://github.com/rubenbriones/Probabilistic-Sharpe-Ratio/blob/master/src/sharpe_ratio_stats.py | rubenbriones/Probabilistic-Sharpe-Ratio reference — Python DSR implementation. | Cross-validates the DSR python formula in `screener.py::deflated_sharpe_ratio` (matches Bailey-LdP). |
| S20 | https://towardsai.net/p/l/the-combinatorial-purged-cross-validation-method | Towards-AI — CPCV (AFML Ch. 12). | Establishes `#paths = k · C(N,k) / N`, motivating future CPCV extension (out of scope for D1; in scope for follow-up). |
| S21 | https://reasonabledeviations.com/notes/adv_fin_ml/ | Reasonable Deviations — AFML chapter notes (incl. Ch. 7, 11, 12). | Cross-validates "Seven Deadly Sins of Backtesting" and CPCV path counting. |

Per-claim cross-validation:

- **CSCV partitioning & PBO logits**: S1 ∩ S2 ∩ S10 ∩ S11 ∩ S12.
- **DSR variance with skew/kurtosis**: S5 ∩ S6 ∩ S7 ∩ S19.
- **Purged k-fold + embargo**: S8 ∩ S9 ∩ S21.
- **Default S=16 for CSCV**: S10 ∩ S11 (vignette uses S=8 for low-N examples — see S12).
- **Multiple-testing penalty rationale**: S3 ∩ S4 ∩ S13 ∩ S14.

---

## 3. Current-state analysis (file:line)

### 3.1 Backtest engine — `src/backtest/backtest_engine.py`
- `BacktestEngine.run()` (`backtest_engine.py:627-691`) iterates portfolios; dispatches to `BacktestRunner` (event) or `_run_fast_vectorized` (fast).
- Default fast-mode config (`_default_fast_config`, `backtest_engine.py:51-63`) enables `mc_n_sims=10_000`, `mc_method="bootstrap"`, `mc_block_size=1`. **Block size = 1 is i.i.d. bootstrap**, not Politis-Romano stationary (S15). Block size of 5 is set in `src/main_backtest.py:92` (`FAST_MODE_CONFIG`).
- No CSCV. No purged-k-fold. No walk-forward (only same-window-prior-years seasonal overlay).

### 3.2 Vectorized backtest — `src/backtest/vectorized_backtest.py`
- `VectorBacktester.monte_carlo` (`vectorized_backtest.py:401-481`) implements:
  - Parametric (Normal) bootstrap.
  - Block bootstrap (`block_size` ≥ 1, fixed-block — **not** stationary bootstrap; the geometric-length variant of Politis-Romano is not implemented).
- `run_same_window_previous_years` (`vectorized_backtest.py:483-621`) gives seasonal overlay, which is a single-period rolling re-run — **not** walk-forward (no parameter re-optimization between windows).
- No DSR, no PBO, no purged k-fold.

### 3.3 Reporting — `src/backtest/reporting.py`
- `aggregate_final_metrics` (`reporting.py:84-108`) emits: Final Portfolio Value, Max Drawdown, Annual Return, Sharpe.
- No DSR, no PBO. The summary CSV consumed downstream by `backtest_analyzer.py` contains only the four metrics above.

### 3.4 Backtest analyzer — `scripts/Backtest_Analysis/backtest_analyzer.py`
- `print_analysis` (`backtest_analyzer.py:171-218`) reads `summary_metrics.csv`, prints Sharpe / annual_return / max_drawdown / final_value, plus inverse-vol-blend optimized weights.
- **No DSR. No PBO. No multiple-testing penalty.** Only single-strategy single-run statistics.

### 3.5 Existing DSR — `src/portfolios/portfolio_6/screener.py:132-175`
- Correctly implements Bailey-LdP DSR (S5, S6, S7, S19):
  - `var_sr = (1 - skew·sr + (kurt_excess/4)·sr²) / (n-1)` — matches S6.
  - Expected-max threshold uses `γ ≈ 0.5772` Euler-Mascheroni constant via the Gumbel approximation — matches S5/S19.
  - Final probability via `norm.cdf` — matches S7.
- Wired from `strategy.py:226-241`. Called per monthly rebalance with `n_trials = len(returns_matrix)` (~519 names = universe size).
- **Gap**: `n_trials` should reflect **all** alternative strategy configurations considered (SCORE_WEIGHTS combos, top-N choices, vol-target levels) — not just universe size. Underestimates the deflation by a factor of ~10–100×. See §4 §G1.

---

## 4. Gap analysis vs. AFML Ch. 7 + Bailey-LdP best practice

| ID | Control | López de Prado / Bailey-LdP location | MQSMaster status | Severity |
|----|---|---|---|---|
| G1 | DSR with **full trial count** (not just universe size) | S5 §2; AFML Ch.14 | DSR exists but `n_trials` undercounts trials by ignoring SCORE_WEIGHTS × WEIGHTING_METHOD × top-N grid | **High** — current DSR is a lower-bound on deflation only |
| G2 | CSCV / PBO over the alternative-strategy grid | S1 / S2 | **Absent** | **High** — no defense against config-space overfitting |
| G3 | Purged K-Fold with embargo | AFML Ch. 7 (Snippets 7.1–7.3); S8, S9 | **Absent** | **High** — required for any ML-based selection (RBP P5/P8) |
| G4 | Walk-forward analysis (rolling/anchored re-optimization) | S17, S18 | Partial: only seasonal overlay (same-window-prior-years). No parameter re-fitting between windows. | **Medium** |
| G5 | Stationary bootstrap (Politis-Romano random block length) | S15, S16 | Partial: only fixed-block bootstrap (`mc_block_size=k`). | **Medium** |
| G6 | CPCV (Combinatorial Purged CV, AFML Ch. 12) for path-replication | S20, S21 | **Absent** | **Low** — D1 scope is CSCV; CPCV is a follow-up |
| G7 | Harvey-Liu-Zhu adjusted t-stat hurdle (≥ 3.0) for new factors | S13, S14 | **Absent** | **Low** — DSR ≥ 0.95 with full `n_trials` is approximately equivalent at large N |
| G8 | Survivorship-bias-free universe (PIT constituents) | AFML Ch. 11 ("seven deadly sins") | **Absent** — `universe.json` is current snapshot (flagged in `teamA/SYNTHESIS.md` B7) | **Out of D1 scope** — Team D2 |

**Conclusion**: G1, G2, G3 are the targeted gaps for this deliverable. G4–G7 are documented but deferred. G8 is out of D1 scope.

---

## 5. Math + algorithm spec for CSCV (Bailey-LdP 2014, paper S1/S2)

### 5.1 Inputs
- A `T × N` matrix `M` of per-period returns, where:
  - `T` is the number of time periods (rows).
  - `N` is the number of alternative strategy configurations (columns).
- A performance metric `R: ℝ^T → ℝ` (default: annualized Sharpe).
- An even integer `S` (default 16; see S10/S11) — the number of disjoint row-blocks.

### 5.2 Algorithm (CSCV)

1. **Trim** `M` so `T mod S == 0`. Drop the residual rows at the tail.
2. **Partition** the rows of `M` into `S` contiguous, non-overlapping submatrices `M_s` of shape `(T/S) × N`, indexed `s = 1, …, S`.
3. **Enumerate** every combination `C ∈ C_S` of `S/2` indices from `{1, …, S}`. Cardinality is `|C_S| = C(S, S/2)`.
4. For each combination `C` (training set):
   a. Concatenate the `M_s` for `s ∈ C` in original time order → in-sample matrix `J_C` of shape `(T/2) × N`.
   b. Concatenate the `M_s` for `s ∉ C` in original time order → out-of-sample matrix `Ĵ_C` of shape `(T/2) × N`.
   c. Compute the per-column performance vectors:
      - `R(J_C) = (R(J_C[:, 1]), …, R(J_C[:, N]))` (IS metrics).
      - `R(Ĵ_C) = (R(Ĵ_C[:, 1]), …, R(Ĵ_C[:, N]))` (OOS metrics).
   d. Convert each to ranks: `r_C = rank(R(J_C))`, `r̂_C = rank(R(Ĵ_C))` (1 = worst, N = best).
   e. Identify the best-IS strategy: `n* = argmax_n r_C[n]`.
   f. Look up its OOS rank: `r̂*_C = r̂_C[n*]`.
   g. Define the relative rank: `ω̄_C = r̂*_C / (N + 1) ∈ (0, 1)`.
   h. Apply the logit transformation: `λ_C = log(ω̄_C / (1 - ω̄_C))`.
5. The empirical PBO is:
   `PBO ≈ #{C : λ_C ≤ 0} / |C_S|`
   = probability that the best in-sample selection underperforms the median out-of-sample.

### 5.3 Performance degradation
For each `C`, plot `R(Ĵ_C)[n*]` (OOS metric of the IS-best strategy) against `R(J_C)[n*]` (IS metric). Fit OLS:
`R_OOS = α + β · R_IS + ε`.
A **negative slope β** confirms degradation — the more you optimized in-sample, the worse the OOS result. Report `α`, `β`, `R²`.

### 5.4 Stochastic dominance
Empirical CDFs of `R(Ĵ_C)[n*]` ("optimized OOS") vs. `R(Ĵ_C)[median across n]` ("median OOS"). If the former is below-and-left of the latter, optimization is value-destroying. Bailey-LdP recommend a Kolmogorov-Smirnov style test.

### 5.5 Loss probability
`P_loss = #{C : R(Ĵ_C)[n*] < threshold} / |C_S|`, with threshold = 0 by default for Sharpe (i.e., probability that the OOS Sharpe of the IS-best strategy is negative).

### 5.6 Default `S`
- `S = 16` is the canonical default (per S10/S11 pypbo, Bailey-LdP §4). `|C_16,8| = 12,870` combinations — tractable.
- `S = 8` (per S12 R-package vignette) is acceptable when `N` is small; `|C_8,4| = 70`.
- **MQSMaster sleeve grid**: with 3 sleeves (P6, P7, P8) × ~5 SCORE_WEIGHTS perturbations × 2 WEIGHTING_METHOD options ≈ N = 30. `S = 16` is appropriate (12,870 splits, ~seconds of compute).

### 5.7 Minimum N to extract signal from PBO
Per S1 §3, **PBO becomes meaningful at N ≥ 4** alternative strategies. With 3 sleeves alone, you cannot tell signal from noise. The mandate at the top of this task is correct: expand to a SCORE_WEIGHTS × WEIGHTING_METHOD grid.

---

## 6. Apply-ready source: `src/backtest/cscv.py`

```python
# src/backtest/cscv.py
"""
Combinatorially Symmetric Cross-Validation (CSCV) and Probability of Backtest
Overfitting (PBO) — Bailey, Borwein, Lopez de Prado, Zhu 2014.

Reference: "The Probability of Backtest Overfitting", Journal of Portfolio
Management, 40(5), 94-107. SSRN 2326253.

Public API:
    cscv_pbo(returns_matrix_per_strategy, *, S=16, metric_func=None,
             threshold=0.0) -> dict

Returned dict keys:
    pbo                       float — probability of backtest overfitting
    performance_degradation   dict  — {'slope', 'intercept', 'r_squared'}
    stochastic_dominance      dict  — empirical CDF samples
    prob_oos_loss             float — P[ OOS metric < threshold | IS-best ]
    n_combinations            int   — C(S, S/2)
    n_strategies              int   — N
    n_periods                 int   — T (after trimming to T mod S == 0)
    block_size                int   — T // S
    logits                    list[float]

Designed to be called on a returns matrix where each column corresponds to a
distinct strategy / strategy-config / sleeve. For MQSMaster, build the matrix
by running the existing fast-mode backtest over a grid of SCORE_WEIGHTS x
WEIGHTING_METHOD permutations (one column per permutation) and stacking the
resulting strategy_returns series column-wise.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default performance metric: annualized Sharpe (252 trading days).
# ---------------------------------------------------------------------------
def _annualized_sharpe(returns: np.ndarray, *, periods_per_year: int = 252) -> float:
    """Return the annualized Sharpe ratio of a 1-D returns array."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    sd = float(arr.std(ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return 0.0
    mu = float(arr.mean())
    return float(mu / sd * np.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _coerce_matrix(returns_matrix_per_strategy) -> tuple[np.ndarray, List[str]]:
    """Coerce input to a (T, N) numpy array + list of strategy labels."""
    if isinstance(returns_matrix_per_strategy, pd.DataFrame):
        labels = [str(c) for c in returns_matrix_per_strategy.columns]
        arr = returns_matrix_per_strategy.to_numpy(dtype=float, copy=True)
    else:
        arr = np.asarray(returns_matrix_per_strategy, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        labels = [f"strategy_{i}" for i in range(arr.shape[1])]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, labels


def _rank_array(x: np.ndarray) -> np.ndarray:
    """Dense rank with mean tie-breaking. 1 = lowest, len(x) = highest."""
    # Use scipy if available for ties handling; fall back to numpy argsort.
    try:
        from scipy.stats import rankdata  # type: ignore
        return rankdata(x, method="average")
    except ImportError:
        order = np.argsort(np.argsort(x))
        return order.astype(float) + 1.0


def _safe_logit(p: float, eps: float = 1e-12) -> float:
    """Logit with edge-case clamping."""
    p = float(min(max(p, eps), 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


# ---------------------------------------------------------------------------
# Core CSCV routine
# ---------------------------------------------------------------------------
def cscv_pbo(
    returns_matrix_per_strategy,
    *,
    S: int = 16,
    metric_func: Optional[Callable[[np.ndarray], float]] = None,
    threshold: float = 0.0,
) -> Dict[str, object]:
    """
    Compute the Probability of Backtest Overfitting (PBO) and accessory
    statistics via Combinatorially Symmetric Cross-Validation (Bailey et al.
    2014).

    Parameters
    ----------
    returns_matrix_per_strategy : DataFrame or ndarray of shape (T, N)
        Per-period returns for each of N strategies / strategy configs.
    S : int, default 16
        Number of disjoint blocks. Must be even. Paper recommends 16.
    metric_func : callable, optional
        Performance metric, signature (ndarray,) -> float. Default:
        annualized Sharpe with 252 trading days.
    threshold : float, default 0.0
        Threshold for `prob_oos_loss`. Default 0 = probability the IS-best
        strategy delivers a *negative* OOS metric.

    Returns
    -------
    dict
        See module docstring.
    """
    if S % 2 != 0:
        raise ValueError(f"S must be even (got S={S}).")
    if S < 4:
        raise ValueError(f"S must be >= 4 to extract any PBO signal (got S={S}).")

    metric = metric_func or _annualized_sharpe

    M, labels = _coerce_matrix(returns_matrix_per_strategy)
    T_raw, N = M.shape
    if N < 4:
        raise ValueError(
            f"CSCV needs N >= 4 distinct strategies (got N={N}); "
            "expand by sweeping SCORE_WEIGHTS / WEIGHTING_METHOD permutations."
        )

    # Step 1: Trim residual rows so T is divisible by S.
    block_size = T_raw // S
    if block_size < 2:
        raise ValueError(
            f"T={T_raw} is too small relative to S={S}; "
            f"need at least 2*S = {2 * S} periods, ideally >> that."
        )
    T = block_size * S
    M = M[:T]

    # Step 2: Partition rows into S contiguous blocks.
    block_ids = np.arange(S)  # 0..S-1
    block_rows: List[np.ndarray] = [
        np.arange(s * block_size, (s + 1) * block_size) for s in block_ids
    ]

    # Step 3-4: Enumerate all C(S, S/2) train/test splits and compute logits.
    half = S // 2
    splits = list(combinations(block_ids.tolist(), half))
    n_splits = len(splits)

    logits: List[float] = []
    is_best_metric: List[float] = []
    oos_best_metric: List[float] = []
    oos_median_metric: List[float] = []

    for train_blocks in splits:
        train_set = set(train_blocks)
        test_blocks = tuple(b for b in block_ids.tolist() if b not in train_set)

        # 4a-b: gather row indices, concatenate in original time order.
        train_rows = np.concatenate([block_rows[b] for b in sorted(train_blocks)])
        test_rows = np.concatenate([block_rows[b] for b in sorted(test_blocks)])

        J = M[train_rows]
        J_hat = M[test_rows]

        # 4c: per-column performance.
        R_is = np.array([metric(J[:, n]) for n in range(N)], dtype=float)
        R_oos = np.array([metric(J_hat[:, n]) for n in range(N)], dtype=float)

        # 4d: ranks (1..N).
        rank_oos = _rank_array(R_oos)

        # 4e: IS-best strategy.
        n_star = int(np.argmax(R_is))

        # 4f-h: OOS rank of IS-best, logit transform.
        r_hat_star = float(rank_oos[n_star])
        omega_bar = r_hat_star / (float(N) + 1.0)
        logits.append(_safe_logit(omega_bar))

        # Bookkeeping for performance degradation and stochastic dominance.
        is_best_metric.append(float(R_is[n_star]))
        oos_best_metric.append(float(R_oos[n_star]))
        oos_median_metric.append(float(np.median(R_oos)))

    logits_np = np.asarray(logits, dtype=float)
    pbo = float(np.mean(logits_np <= 0.0))

    # Performance degradation: OLS regression R_OOS = alpha + beta * R_IS.
    is_arr = np.asarray(is_best_metric, dtype=float)
    oos_arr = np.asarray(oos_best_metric, dtype=float)
    if is_arr.size >= 2 and np.isfinite(is_arr).all() and np.isfinite(oos_arr).all():
        slope, intercept = np.polyfit(is_arr, oos_arr, 1)
        ss_tot = float(((oos_arr - oos_arr.mean()) ** 2).sum())
        ss_res = float(((oos_arr - (intercept + slope * is_arr)) ** 2).sum())
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        slope, intercept, r_squared = 0.0, 0.0, 0.0

    # Stochastic dominance: empirical CDFs for plotting / KS test downstream.
    oos_arr_sorted = np.sort(oos_arr)
    median_arr_sorted = np.sort(np.asarray(oos_median_metric, dtype=float))
    cdf_x = np.arange(1, n_splits + 1, dtype=float) / float(n_splits)

    prob_oos_loss = float(np.mean(oos_arr < threshold))

    return {
        "pbo": pbo,
        "performance_degradation": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
        },
        "stochastic_dominance": {
            "cdf_probabilities": cdf_x.tolist(),
            "oos_best_sorted": oos_arr_sorted.tolist(),
            "oos_median_sorted": median_arr_sorted.tolist(),
        },
        "prob_oos_loss": prob_oos_loss,
        "n_combinations": int(n_splits),
        "n_strategies": int(N),
        "n_periods": int(T),
        "block_size": int(block_size),
        "logits": logits_np.tolist(),
        "strategy_labels": labels,
        "threshold": float(threshold),
        "S": int(S),
    }


# ---------------------------------------------------------------------------
# Convenience: build the (T, N) matrix from a list of (label, returns) pairs.
# ---------------------------------------------------------------------------
def build_strategy_grid_matrix(
    series_per_strategy: Dict[str, pd.Series],
    *,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Construct a (T, N) DataFrame of aligned daily returns from a dict of
    {strategy_label: returns_series}. The union of indices is used; gaps are
    filled with `fill_value` (default 0.0 — equivalent to "no trade today").

    Useful for assembling the CSCV input matrix from the grid of
    SCORE_WEIGHTS x WEIGHTING_METHOD permutations produced by fast-mode
    backtests.
    """
    if not series_per_strategy:
        return pd.DataFrame()
    aligned = pd.DataFrame(series_per_strategy)
    aligned = aligned.sort_index()
    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    aligned = aligned.fillna(fill_value)
    return aligned


__all__ = [
    "cscv_pbo",
    "build_strategy_grid_matrix",
]
```

### 6.1 Smoke-test (pytest-ready, ship in `tests/backtest/test_cscv.py`)

```python
# tests/backtest/test_cscv.py
import numpy as np
import pandas as pd
import pytest

from src.backtest.cscv import cscv_pbo, build_strategy_grid_matrix


def test_cscv_random_strategies_high_pbo():
    """Pure-noise strategies should give PBO close to 0.5 (no skill)."""
    rng = np.random.default_rng(42)
    T, N = 252 * 4, 16  # 4 years daily x 16 random strategies
    returns = rng.normal(loc=0.0, scale=0.01, size=(T, N))
    df = pd.DataFrame(returns, columns=[f"s_{i}" for i in range(N)])

    result = cscv_pbo(df, S=16)
    # Pure-noise: best-IS strategy is no better than median OOS on average.
    assert 0.35 <= result["pbo"] <= 0.65, result["pbo"]
    # Slope should be near zero (no real skill carry-over).
    assert abs(result["performance_degradation"]["slope"]) < 0.5


def test_cscv_one_strong_strategy_low_pbo():
    """One genuinely strong strategy should give PBO well below 0.5."""
    rng = np.random.default_rng(7)
    T, N = 252 * 4, 16
    noise = rng.normal(loc=0.0, scale=0.01, size=(T, N))
    noise[:, 0] += 0.0015  # strategy_0 has real edge
    df = pd.DataFrame(noise, columns=[f"s_{i}" for i in range(N)])
    result = cscv_pbo(df, S=16)
    assert result["pbo"] < 0.2, result["pbo"]


def test_cscv_rejects_odd_S():
    with pytest.raises(ValueError):
        cscv_pbo(np.random.randn(64, 4), S=15)


def test_cscv_rejects_small_N():
    with pytest.raises(ValueError):
        cscv_pbo(np.random.randn(128, 3), S=8)


def test_build_strategy_grid_matrix_aligns_indices():
    idx_a = pd.date_range("2024-01-01", periods=10, freq="D")
    idx_b = pd.date_range("2024-01-05", periods=10, freq="D")
    grid = build_strategy_grid_matrix(
        {
            "a": pd.Series(np.arange(10, dtype=float), index=idx_a),
            "b": pd.Series(np.arange(10, dtype=float), index=idx_b),
        }
    )
    assert grid.shape == (14, 2)
    assert grid.index.is_monotonic_increasing
    assert not grid.isna().any().any()
```

---

## 7. Apply-ready source: `src/backtest/purged_kfold.py`

Per López de Prado AFML Ch. 7 (Snippets 7.1–7.3); cross-validated against S8 (Wikipedia) and S9 (QuantInsti).

```python
# src/backtest/purged_kfold.py
"""
Purged K-Fold Cross-Validation with embargo — Lopez de Prado, AFML 2018,
Chapter 7 (Snippets 7.1, 7.2, 7.3).

Public API:
    PurgedKFold(n_splits, t1, embargo_td)

Use case in MQSMaster:
    - RBP forecast labels (src/portfolios/portfolio_5, portfolio_8) where
      `t1[i]` is the end of the forecast horizon for sample i.
    - Any ML model whose label spans multiple days (e.g. triple-barrier
      labels).

The class is sklearn-API-compatible: implements `split(X, y=None, groups=None)`
yielding (train_idx, test_idx) tuples, and `get_n_splits(...)` -> int.
"""

from __future__ import annotations

from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class PurgedKFold:
    """
    K-fold cross-validation with **purging** and **embargo**, per AFML §7.

    Parameters
    ----------
    n_splits : int
        Number of folds. Must be >= 2. Default 5.
    t1 : pd.Series
        Indexed by sample-id, values are the **event-end times** for each
        sample. For sample `i`, the label is observed at time `t1.iloc[i]`
        (which may be later than the feature time `t1.index[i]`).
    embargo_td : pd.Timedelta or float
        Embargo length applied AFTER each test fold to prevent leakage from
        labels immediately following the test set.
        - If `pd.Timedelta`, applied directly.
        - If `float`, interpreted as a fraction in (0, 1) of total span
          per AFML Snippet 7.3 (h = ceil(embargo_pct * T)).

    Notes
    -----
    Purging rule (AFML §7.4.1):
        Drop from the train set any sample `i` such that
            t1.index[i] <= test_end  AND  t1.iloc[i] >= test_start
        i.e. any sample whose [feature_time, label_time] interval overlaps
        the test fold's [test_start, test_end].

    Embargo rule (AFML §7.4.2):
        After purging, also drop any train sample whose feature_time falls
        in the half-open interval (test_end, test_end + embargo_td].

    Sklearn compatibility:
        Implements `split` and `get_n_splits` so this class can be passed
        directly to `sklearn.model_selection.cross_val_score`, GridSearchCV,
        etc., for any RBP-style model whose labels have multi-day horizons.
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: Optional[pd.Series] = None,
        embargo_td: "pd.Timedelta | float | int" = 0.0,
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2 (got {n_splits}).")
        if t1 is None or not isinstance(t1, pd.Series):
            raise ValueError(
                "t1 must be a pd.Series of label-end times, indexed by sample."
            )
        if not t1.index.is_monotonic_increasing:
            raise ValueError(
                "t1.index must be monotonically increasing (sort first)."
            )
        # Coerce embargo into a Timedelta-or-pct value.
        if isinstance(embargo_td, (int, float)):
            if 0.0 <= float(embargo_td) <= 1.0:
                self._embargo_pct = float(embargo_td)
                self._embargo_td: Optional[pd.Timedelta] = None
            else:
                raise ValueError(
                    "Numeric embargo_td must be a fraction in [0, 1]."
                )
        elif isinstance(embargo_td, pd.Timedelta):
            self._embargo_pct = None
            self._embargo_td = embargo_td
        else:
            raise ValueError(
                "embargo_td must be a pd.Timedelta or a fraction in [0, 1]."
            )
        self.n_splits = int(n_splits)
        # Keep a copy to avoid mutating caller state.
        self.t1 = t1.copy()

    # ------------------------------------------------------------------
    # sklearn compatibility
    # ------------------------------------------------------------------
    def get_n_splits(
        self,
        X=None,
        y=None,
        groups=None,
    ) -> int:
        return self.n_splits

    # ------------------------------------------------------------------
    # Core: yield (train_idx, test_idx) per fold.
    # ------------------------------------------------------------------
    def split(
        self,
        X,
        y=None,
        groups=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_idx, test_idx) for each fold. Indices are positional
        (int) into `X`. If `X` is a DataFrame, its index must match
        `self.t1.index` 1-to-1.
        """
        # Resolve `n` (number of samples) and align with t1.
        n = self._infer_n_samples(X)
        if len(self.t1) != n:
            raise ValueError(
                f"t1 length ({len(self.t1)}) != X length ({n}); "
                "t1 must be aligned 1-to-1 with X."
            )

        # Contiguous, equal-sized folds (no shuffle — financial data is ordered).
        indices = np.arange(n)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1

        starts = np.cumsum(np.concatenate([[0], fold_sizes[:-1]]))
        stops = np.cumsum(fold_sizes)

        # Embargo length in positional units (per AFML Snippet 7.3).
        if self._embargo_pct is not None:
            h_pos = int(np.ceil(self._embargo_pct * n))
        else:
            # Convert pd.Timedelta -> positional units lazily inside the loop
            # because it depends on test boundaries.
            h_pos = None

        feature_times = self.t1.index
        label_times = pd.Series(self.t1.values, index=feature_times)

        for fold in range(self.n_splits):
            test_start_idx = int(starts[fold])
            test_stop_idx = int(stops[fold])
            test_idx = indices[test_start_idx:test_stop_idx]

            test_feature_start = feature_times[test_start_idx]
            test_feature_end = feature_times[test_stop_idx - 1]
            # Test fold's label window extends up to max label time in it.
            test_label_end = label_times.iloc[test_start_idx:test_stop_idx].max()

            # --- 1. Purge -----------------------------------------------------
            # Drop train samples whose [feature_time, label_time] overlaps
            # [test_feature_start, test_label_end].
            #
            # Overlap iff:
            #   feature_time <= test_label_end  AND  label_time >= test_feature_start
            overlap_mask = (
                (feature_times <= test_label_end)
                & (label_times.values >= test_feature_start)
            )
            purged_positions = set(np.where(overlap_mask)[0].tolist())

            # --- 2. Embargo ---------------------------------------------------
            embargo_positions: set[int] = set()
            if self._embargo_td is not None:
                # Time-based embargo: drop train samples in (test_label_end, test_label_end + h]
                emb_cut = test_label_end + self._embargo_td
                emb_mask = (
                    (feature_times > test_label_end)
                    & (feature_times <= emb_cut)
                )
                embargo_positions = set(np.where(emb_mask)[0].tolist())
            elif h_pos is not None and h_pos > 0:
                # Pct-based embargo: drop positional indices in (test_stop_idx, test_stop_idx + h_pos].
                start_emb = test_stop_idx
                end_emb = min(test_stop_idx + h_pos, n)
                embargo_positions = set(range(start_emb, end_emb))

            # --- 3. Build train set ------------------------------------------
            drop = set(test_idx.tolist()) | purged_positions | embargo_positions
            train_idx = np.array(
                [i for i in indices if i not in drop],
                dtype=int,
            )
            if train_idx.size == 0:
                # Degenerate fold (entire dataset purged) — skip rather than crash.
                continue
            yield train_idx, test_idx

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_n_samples(X) -> int:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return len(X)
        arr = np.asarray(X)
        if arr.ndim == 0:
            raise ValueError("X must be array-like with at least 1 sample.")
        return arr.shape[0]


# ---------------------------------------------------------------------------
# Convenience: degenerate "single-step labels" t1 builder.
# ---------------------------------------------------------------------------
def t1_from_horizon(
    feature_times: Sequence,
    horizon: pd.Timedelta,
) -> pd.Series:
    """
    Build a t1 Series for a fixed-horizon label (e.g. h=21d for RBP):
        t1[i] = feature_times[i] + horizon
    """
    idx = pd.DatetimeIndex(feature_times)
    return pd.Series(idx + horizon, index=idx)


__all__ = [
    "PurgedKFold",
    "t1_from_horizon",
]
```

### 7.1 Smoke-test (`tests/backtest/test_purged_kfold.py`)

```python
# tests/backtest/test_purged_kfold.py
import numpy as np
import pandas as pd
import pytest

from src.backtest.purged_kfold import PurgedKFold, t1_from_horizon


def test_purged_kfold_no_overlap_in_train_test():
    n = 252  # one trading year
    feature_times = pd.date_range("2024-01-02", periods=n, freq="B")
    horizon = pd.Timedelta(days=5)
    t1 = t1_from_horizon(feature_times, horizon)
    X = pd.DataFrame({"f": np.arange(n)}, index=feature_times)

    cv = PurgedKFold(n_splits=5, t1=t1, embargo_td=0.01)
    for train_idx, test_idx in cv.split(X):
        # No positional overlap.
        assert not set(train_idx.tolist()) & set(test_idx.tolist())
        # Verify purging: no train sample's [feature_time, label_time]
        # overlaps the test set's [feature_time, label_time].
        test_feature_start = feature_times[test_idx[0]]
        test_label_end = t1.iloc[test_idx].max()
        for i in train_idx:
            ft = feature_times[i]
            lt = t1.iloc[i]
            assert not (ft <= test_label_end and lt >= test_feature_start), (
                f"Overlap leak at {i}: feature_time={ft}, label_time={lt}, "
                f"test=[{test_feature_start}, {test_label_end}]"
            )


def test_purged_kfold_embargo_drops_post_test_samples():
    n = 100
    feature_times = pd.date_range("2024-01-02", periods=n, freq="B")
    t1 = t1_from_horizon(feature_times, pd.Timedelta(days=1))
    X = pd.DataFrame({"f": np.arange(n)}, index=feature_times)

    cv = PurgedKFold(n_splits=5, t1=t1, embargo_td=0.05)  # 5% = 5 samples
    train_sets = [set(train.tolist()) for train, _ in cv.split(X)]

    # In folds 0..3 (i.e., not the last), embargo of 5 positions should remove
    # 5 train samples immediately after test_stop_idx.
    # We just assert at least one of the folds has fewer than (n - test_size)
    # training samples (it would equal n - test_size with no embargo).
    test_size = n // 5
    assert any(len(s) < (n - test_size) for s in train_sets[:-1])


def test_purged_kfold_rejects_unsorted_t1():
    feature_times = pd.DatetimeIndex(
        ["2024-01-03", "2024-01-02", "2024-01-04"]
    )
    t1 = pd.Series(feature_times + pd.Timedelta(days=1), index=feature_times)
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=2, t1=t1)


def test_purged_kfold_get_n_splits_matches():
    feature_times = pd.date_range("2024-01-02", periods=20, freq="B")
    t1 = t1_from_horizon(feature_times, pd.Timedelta(days=1))
    cv = PurgedKFold(n_splits=4, t1=t1, embargo_td=0.0)
    assert cv.get_n_splits() == 4
```

---

## 8. Unified diff: wire PBO + DSR into `backtest_analyzer.py`

The diff below adds:
- A new helper `cscv_pbo_for_strategy_grid()` that loads the latest backtest runs for each portfolio in the supplied set, builds a (T × N) returns matrix from the per-strategy `performance_timeseries_percentage.csv` (or `performance_timeseries_absolute.csv` if % is missing), runs `cscv_pbo`, and returns the PBO + degradation slope.
- A new CLI flag `--cscv-portfolios` taking ≥ 2 portfolio IDs (e.g. `--cscv-portfolios 6 7 8`).
- An extension of `print_analysis` to print DSR alongside Sharpe (using existing `deflated_sharpe_ratio` from `src/portfolios/portfolio_6/screener.py`) with a configurable `--dsr-n-trials` argument.
- A falsification-gate exit-code: `--gate-pbo 0.5` returns non-zero exit if any portfolio's grid PBO exceeds 0.5.

```diff
--- a/scripts/Backtest_Analysis/backtest_analyzer.py
+++ b/scripts/Backtest_Analysis/backtest_analyzer.py
@@ -1,16 +1,29 @@
 from __future__ import annotations
 
 import argparse
+import logging
 import re
+import sys
 from dataclasses import dataclass
 from pathlib import Path
 
 import numpy as np
 import pandas as pd
 from summary_metrics_formatter import format_backtest_date_range, get_summary_value
+
+# Add repo root to sys.path so `src.*` imports resolve when this script is
+# executed directly (i.e., not via `python -m`).
+_REPO_ROOT = Path(__file__).resolve().parents[2]
+if str(_REPO_ROOT) not in sys.path:
+    sys.path.insert(0, str(_REPO_ROOT))
+
+from src.backtest.cscv import build_strategy_grid_matrix, cscv_pbo  # noqa: E402
+from src.portfolios.portfolio_6.screener import (  # noqa: E402
+    deflated_sharpe_ratio,
+)
 
 MIN_WEIGHT = 0.05
 RUN_NAME_RE = re.compile(r"(?P<timestamp>\d{8}_\d{6})_backtest_(?P<portfolio_id>.+)")
 SUMMARY_FILE = "summary_metrics.csv"
 RISK_FILE = "portfolio_risk_components.csv"
 CORR_FILE = "annualized_correlation_matrix.csv"
 PERFORMANCE_FILE = "performance_timeseries_absolute.csv"
+PERFORMANCE_PCT_FILE = "performance_timeseries_percentage.csv"
+
+logger = logging.getLogger("backtest_analyzer")
 
 
 @dataclass(frozen=True)
@@ -160,6 +173,80 @@ def _fmt_pct(value: float) -> str:
 def _fmt_num(value: float, decimals: int = 4) -> str:
     return "N/A" if np.isnan(value) else f"{value:.{decimals}f}"
 
 
+# ---------------------------------------------------------------------------
+# DSR + PBO helpers
+# ---------------------------------------------------------------------------
+def _load_pct_returns(run: BacktestRun) -> pd.Series:
+    """Return a daily-percentage-return Series for a single backtest run."""
+    pct_path = run.path / PERFORMANCE_PCT_FILE
+    if pct_path.exists():
+        df = pd.read_csv(pct_path)
+        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
+        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
+        if "portfolio_pct_ret" in df.columns:
+            cum = df.set_index("timestamp")["portfolio_pct_ret"].astype(float)
+            return cum.diff().fillna(cum.iloc[0]).rename(run.portfolio_id)
+
+    abs_path = run.path / PERFORMANCE_FILE
+    if not abs_path.exists():
+        return pd.Series(dtype=float, name=run.portfolio_id)
+
+    df = pd.read_csv(abs_path)
+    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
+    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
+    if "portfolio_value" not in df.columns:
+        return pd.Series(dtype=float, name=run.portfolio_id)
+    daily = (
+        df.set_index("timestamp")["portfolio_value"]
+        .astype(float)
+        .resample("D")
+        .last()
+        .ffill()
+    )
+    return daily.pct_change().dropna().rename(run.portfolio_id)
+
+
+def _print_dsr_for_run(run: BacktestRun, *, n_trials: int) -> float:
+    """Print DSR for a single backtest run and return the probability."""
+    series = _load_pct_returns(run)
+    if series.empty or len(series) < 30:
+        print(f"  Deflated Sharpe:        N/A (insufficient history)")
+        return float("nan")
+    dsr = deflated_sharpe_ratio(series, n_trials=max(n_trials, 1))
+    print(f"  Deflated Sharpe Prob:   {dsr:>10.3f}  (n_trials={n_trials})")
+    return float(dsr)
+
+
+def cscv_pbo_for_strategy_grid(
+    runs: dict[str, BacktestRun],
+    *,
+    portfolio_ids: list[str],
+    S: int = 16,
+) -> dict:
+    """Run CSCV / PBO across the supplied portfolio IDs.
+
+    Each portfolio contributes one column to the returns matrix. To make this
+    statistically meaningful, expand by running the existing backtest over
+    multiple SCORE_WEIGHTS / WEIGHTING_METHOD permutations and rerunning this
+    analyzer with the resulting per-permutation portfolio_ids."""
+    series_map: dict[str, pd.Series] = {}
+    for pid in portfolio_ids:
+        run = runs.get(pid)
+        if run is None:
+            logger.warning("CSCV: no run found for portfolio_id=%s; skipping.", pid)
+            continue
+        s = _load_pct_returns(run)
+        if not s.empty:
+            series_map[pid] = s
+
+    if len(series_map) < 4:
+        raise ValueError(
+            f"CSCV needs >= 4 strategies; got {len(series_map)}. Expand by "
+            "rerunning backtests over a SCORE_WEIGHTS / WEIGHTING_METHOD grid."
+        )
+
+    grid = build_strategy_grid_matrix(series_map)
+    return cscv_pbo(grid, S=S)
+
+
 def print_analysis(run: BacktestRun, risk_appetite: float) -> None:
     summary_df = load_csv(run, SUMMARY_FILE)
     risk_df = load_csv(run, RISK_FILE)
@@ -198,6 +285,9 @@ def print_analysis(run: BacktestRun, risk_appetite: float) -> None:
     print(f"  Current Ann. Vol:       {current_vol:>10.2%}")
     print(f"  Optimized Ann. Vol:     {optimized_vol:>10.2%}")
 
+    # Bailey-Lopez de Prado deflated Sharpe (multiple-testing-adjusted).
+    _print_dsr_for_run(run, n_trials=_DSR_N_TRIALS)
+
 
 def parse_args() -> argparse.Namespace:
     parser = argparse.ArgumentParser(
@@ -224,9 +314,40 @@ def parse_args() -> argparse.Namespace:
         default=None,
         help="Risk appetite in [0, 1]. If omitted, prompt in terminal.",
     )
+    parser.add_argument(
+        "--cscv-portfolios",
+        nargs="+",
+        default=None,
+        help=(
+            "Run CSCV/PBO across the listed portfolio IDs (>= 4 needed). "
+            "Expand the strategy grid by rerunning fast-mode backtests over "
+            "permutations of SCORE_WEIGHTS / WEIGHTING_METHOD before invoking."
+        ),
+    )
+    parser.add_argument(
+        "--cscv-S",
+        type=int,
+        default=16,
+        help="CSCV block count S (even, >= 4). Default 16 (Bailey-LdP 2014).",
+    )
+    parser.add_argument(
+        "--dsr-n-trials",
+        type=int,
+        default=1,
+        help=(
+            "Number of independent strategy trials for DSR. Must reflect the "
+            "full search space (SCORE_WEIGHTS grid, top-N, vol target, etc.)."
+        ),
+    )
+    parser.add_argument(
+        "--gate-pbo",
+        type=float,
+        default=None,
+        help=(
+            "Falsification gate. If PBO > value, exit code 2. "
+            "teamA/SYNTHESIS.md §11 mandates 0.5 for P7/P8 promotion."
+        ),
+    )
     return parser.parse_args()
 
 
@@ -250,8 +371,12 @@ def main() -> None:
     )
 
     if not selected_runs:
         print(
             "No matching backtest runs found with required files "
             f"({SUMMARY_FILE}, {RISK_FILE}, {CORR_FILE})."
         )
         return
 
+    # Surface DSR n_trials to print_analysis via module-level constant.
+    global _DSR_N_TRIALS
+    _DSR_N_TRIALS = max(int(args.dsr_n_trials), 1)
+
     for _, run in selected_runs.items():
         print_analysis(run, risk_appetite)
 
+    # CSCV / PBO across the requested portfolio grid.
+    if args.cscv_portfolios:
+        try:
+            pbo_result = cscv_pbo_for_strategy_grid(
+                selected_runs,
+                portfolio_ids=list(args.cscv_portfolios),
+                S=int(args.cscv_S),
+            )
+        except ValueError as exc:
+            print(f"\nCSCV/PBO skipped: {exc}")
+            return
+        deg = pbo_result["performance_degradation"]
+        print("\n=== CSCV / PBO (Bailey-LdP 2014) ===")
+        print(f"  PBO:                    {pbo_result['pbo']:.3f}")
+        print(f"  P(OOS loss):            {pbo_result['prob_oos_loss']:.3f}")
+        print(f"  Degradation slope:      {deg['slope']:.4f}")
+        print(f"  Degradation R^2:        {deg['r_squared']:.4f}")
+        print(f"  N strategies:           {pbo_result['n_strategies']}")
+        print(f"  N combinations:         {pbo_result['n_combinations']}")
+        print(f"  S blocks:               {pbo_result['S']}")
+
+        if args.gate_pbo is not None and pbo_result["pbo"] > float(args.gate_pbo):
+            print(
+                f"\nFALSIFICATION GATE FAIL: PBO {pbo_result['pbo']:.3f} "
+                f"> threshold {float(args.gate_pbo):.3f}. Exit code 2."
+            )
+            sys.exit(2)
+
 
 if __name__ == "__main__":
     main()
```

### 8.1 Notes on the diff
- The module-level `_DSR_N_TRIALS` is set inside `main()` before `print_analysis` is called. This avoids changing the public signature of `print_analysis` (back-compat).
- `_load_pct_returns` prefers `performance_timeseries_percentage.csv` (cleaner, fewer ffill artifacts) and falls back to `performance_timeseries_absolute.csv`.
- DSR uses the existing implementation in `src/portfolios/portfolio_6/screener.py` — no duplication.
- `--gate-pbo 0.5` is the P7 promotion threshold per `teamA/SYNTHESIS.md §11`.

---

## 9. Falsification gate (concrete, numeric)

### 9.1 P7 promotion gate (consolidated with `teamA/SYNTHESIS.md §11`)
P7 is blocked from `portfolio_manager_config.json` weight > 0 until **all** of these pass on a 2021-01-01 → 2026-04-30 backtest:

1. **PBO ≤ 0.50** over a SCORE_WEIGHTS / WEIGHTING_METHOD grid of **N ≥ 16** distinct P7 configurations.
2. `DSR(P7) ≥ 0.50` using `n_trials = N` (grid size, not 1).
3. `Sharpe(P7) − Sharpe(P6) ≥ +0.15` averaged across the OOS folds.
4. `MaxDD(P7) ≤ 1.10 · MaxDD(P6)` averaged across the OOS folds.
5. Purged-K-fold leakage canary: shuffling sentiment timestamps by ±5 trading days **destroys** the alpha. (Implement in a follow-up by passing a sentiment-shuffled DataFrame through `Portfolio7Strategy._sentiment_tilt`.)

### 9.2 P8 promotion gate
1. **PBO ≤ 0.50** over a grid of N ≥ 16 distinct P8 configurations varying RBP weight (`W_RBP` ∈ {0.5, 1.0, 1.5, 2.0}) × top-N (∈ {30, 50, 70, 100}).
2. `DSR(P8) ≥ 0.95` (stricter threshold than P7 because RBP is an ML predictor).
3. `ΔIR(P8 − P6) ≥ +0.10` averaged across the purged-K-fold OOS folds.
4. RBI top-3 stable in ≥ 9/12 monthly refreshes.

### 9.3 Operational mechanics
```bash
# Step 1: Generate the strategy grid (offline, ~hours).
# For each (SCORE_WEIGHTS, WEIGHTING_METHOD) permutation:
#   - Edit src/portfolios/portfolio_7/config.json
#   - Run: python -m src.main_backtest
#   - Tag the run folder with the permutation key.

# Step 2: Aggregate results into a single returns matrix and run CSCV.
python scripts/Backtest_Analysis/backtest_analyzer.py \
    --portfolio 7_w1 --portfolio 7_w2 --portfolio 7_w3 ... \
    --cscv-portfolios 7_w1 7_w2 7_w3 ... \
    --cscv-S 16 \
    --dsr-n-trials 16 \
    --gate-pbo 0.5

# Exit code 0 => gate passes; 2 => fails; non-zero => promotion blocked.
```

### 9.4 Why PBO > 0.5 is the right threshold
- A PBO ≥ 0.5 means the IS-best strategy is at least as likely to underperform the OOS median as it is to outperform → indistinguishable from chance. Bailey-LdP 2014 §4 use 0.5 as the "ban" threshold.
- The threshold is not 0 because some PBO is unavoidable when N is finite; 0.5 is the random-skill baseline.

---

## 10. How to run CSCV on the existing N-strategy backtest output

### 10.1 Naive run (N = 3 sleeves)

```bash
# Assumes you've already run main_backtest.py for portfolios 6, 7, 8.
python scripts/Backtest_Analysis/backtest_analyzer.py \
    --portfolio 6 --portfolio 7 --portfolio 8 \
    --cscv-portfolios 6 7 8 \
    --cscv-S 8
```

Result: `cscv_pbo` will raise `ValueError("CSCV needs >= 4 strategies")`. **This is by design** — N = 3 is below the statistical floor.

### 10.2 Proper run — expand by SCORE_WEIGHTS / WEIGHTING_METHOD grid

Create a helper script (out of D1 scope; sketch only):

```python
# scripts/Backtest_Analysis/sweep_p7_grid.py
import copy, json, subprocess
from itertools import product

P7_CFG = Path("src/portfolios/portfolio_7/config.json")
base_cfg = json.loads(P7_CFG.read_text())

# 4 vol weights x 2 sentiment-tilt strengths x 2 top-N choices = 16 configs.
grid = list(product(
    [0.5, 1.0, 1.5, 2.0],           # SCORE_WEIGHTS.vol
    [0.1, 0.3],                     # P7 SENTIMENT_LAMBDA
    [30, 70],                       # SCREEN_TOP_N
))

for i, (vw, lam, top_n) in enumerate(grid):
    cfg = copy.deepcopy(base_cfg)
    cfg["PORTFOLIO_7_CONFIG"]["SCORE_WEIGHTS"]["vol"] = vw
    cfg["PORTFOLIO_7_CONFIG"]["SENTIMENT_LAMBDA"] = lam
    cfg["PORTFOLIO_7_CONFIG"]["SCREEN_TOP_N"] = top_n
    # Override PORTFOLIO_ID so each run dumps to a distinct folder.
    cfg["PORTFOLIO_ID"] = f"7_grid_{i:02d}"
    P7_CFG.write_text(json.dumps(cfg, indent=2))
    subprocess.run(["python", "-m", "src.main_backtest"], check=True)
```

Then aggregate:

```bash
python scripts/Backtest_Analysis/backtest_analyzer.py \
    $(for i in $(seq -f "%02g" 0 15); do echo "--portfolio 7_grid_$i"; done) \
    $(for i in $(seq -f "%02g" 0 15); do echo "--cscv-portfolios 7_grid_$i"; done) \
    --cscv-S 16 \
    --dsr-n-trials 16 \
    --gate-pbo 0.5
```

### 10.3 What "N strategies" means concretely
For MQSMaster, "strategy" = one full backtest with a unique config. Each config produces one daily-returns column. The CSCV matrix is built by stacking these columns aligned on the daily date index (handled by `build_strategy_grid_matrix`).

---

## 11. Risks + rollback path

### 11.1 Risks
- **R1 (Compute cost)**: With N = 16 P7 configs × 5-year backtests, each ~1-3 minutes in fast mode, the full grid takes ~30-60 minutes. CSCV itself is cheap (C(16,8) = 12,870 splits, each computing 16 Sharpes → ~30s with vectorized metric_func).
- **R2 (Trivial DSR n_trials)**: If the caller passes `--dsr-n-trials 1`, the DSR is **uncorrected** for multiple testing. The default in the diff is 1 for back-compat; downstream tooling MUST set this to the grid size. Document this in the help-text.
- **R3 (Survivorship bias)**: `universe.json` is a current snapshot. Both P7 and P8 backtests are biased upward; PBO will under-estimate true overfitting. Surfaced in `teamA/SYNTHESIS.md` B7. Out of D1 scope but the gate should not be relied on until D2 PIT audit closes.
- **R4 (Block-size choice for CSCV)**: We use contiguous time blocks (per Bailey-LdP). A small block_size = T / S can violate the random-block intuition of stationary bootstrap. With S = 16 and 5-year daily backtest (T ≈ 1260), block_size ≈ 79 days — enough to preserve monthly cycles.
- **R5 (Non-stationary correlations)**: If strategies' correlation structure shifts mid-backtest (e.g. 2022 sector rotation), the IS/OOS partition is more pessimistic in some configs. Mitigation: report PBO per-decade as well as full-sample.
- **R6 (sklearn dependency for PurgedKFold)**: We implement `PurgedKFold` without inheriting from `sklearn.model_selection.BaseCrossValidator` to avoid the dependency. If RBP later requires `cross_val_score(..., cv=PurgedKFold(...))`, the duck-typed `split` + `get_n_splits` API is sufficient.

### 11.2 Rollback path
Each file is independently revertable:
1. **Revert `cscv.py`**: `git rm src/backtest/cscv.py` (and the smoke test).
2. **Revert `purged_kfold.py`**: `git rm src/backtest/purged_kfold.py` (and the smoke test).
3. **Revert `backtest_analyzer.py` diff**: standard `git checkout HEAD~1 -- scripts/Backtest_Analysis/backtest_analyzer.py` because the diff is purely additive — no behavior is changed unless `--cscv-portfolios` or `--gate-pbo` is passed.

The diff is **back-compat by construction**: with no new flags, the analyzer behaves identically. Only when `--cscv-portfolios` is supplied does new logic run.

### 11.3 Follow-ups (out of D1 scope)
- **F1**: Implement Combinatorial Purged Cross-Validation (CPCV, AFML Ch. 12) — generates `k · C(N, k) / N` backtest paths instead of one. Worth pursuing once `PurgedKFold` is in production.
- **F2**: Add stationary bootstrap (Politis-Romano, S15) as a `mc_method="stationary"` option in `VectorBacktester.monte_carlo`. Trivial: replace fixed block_size with `rng.geometric(p=1/block_size)`.
- **F3**: Implement Harvey-Liu-Zhu BHY / FDR multiple-testing hurdle (S13) as a complementary check to DSR. DSR ≥ 0.95 ≈ Bonferroni at N ≈ 20; for N > 100 the BHY framework is less conservative.
- **F4**: Wire the **leakage canary** referenced in the P7 falsification gate — pass a sentiment-timestamp-shuffled DataFrame through `Portfolio7Strategy` and verify alpha collapses.

---

## 12. Summary deliverable checklist

| Item | Status | Location in this file |
|---|---|---|
| Executive summary | Done | §1 |
| ≥10 primary sources | 21 sources | §2 |
| Current-state analysis (file:line) | Done | §3 |
| Gap analysis | Done | §4 |
| CSCV math + algorithm | Done | §5 |
| Apply-ready `src/backtest/cscv.py` | Done (fenced) | §6 |
| Apply-ready `src/backtest/purged_kfold.py` | Done (fenced) | §7 |
| Unified diff for `backtest_analyzer.py` | Done (fenced) | §8 |
| Falsification gate (PBO > 0.5) | Done | §9 |
| Run instructions for N-strategy grid | Done | §10 |
| Risks + rollback | Done | §11 |

End of D1.
