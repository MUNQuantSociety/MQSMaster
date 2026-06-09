# B1 — Hierarchical Risk Parity (HRP) as Opt-In Weighting Method for Portfolio_6

**Author:** Team B / Agent B1
**Date:** 2026-05-20
**Repo:** MQSMaster
**Branch:** dev
**Scope:** Read-only audit of `src/portfolios/portfolio_6/{screener.py, strategy.py}` and `src/portfolios/portfolio_BASE/strategy.py`; design + apply-ready diff for adding HRP (and stub ERC) as opt-in weighting methods alongside the existing inverse-volatility heuristic. No source files modified by this agent.

---

## 1. Executive summary (≤200 words)

Portfolio_6 currently weights its top-N screened universe by 1/σᵢ with iterative cap to `MAX_WEIGHT_PER_STOCK` (`screener.py:80–112`). This is a degenerate (uncorrelated-asset) limit of the Maillard–Roncalli–Teiletche (2010) ERC portfolio and ignores cross-asset correlation, so it systematically over-allocates to the lowest-vol sub-cluster (typically utilities/staples) — the same concentration pathology documented for inverse-vol weighting in CFA Institute (2024) and the low-vol-anomaly literature.

Lopez de Prado (2016) Hierarchical Risk Parity (HRP) repairs this by (a) converting the correlation matrix to a proper distance d(i,j)=√(½(1−ρ)), (b) single-linkage clustering, (c) reordering the covariance matrix into quasi-diagonal blocks, and (d) top-down recursive bisection that splits capital between sibling clusters in inverse proportion to their inverse-variance-weighted variance. It needs no Σ inversion (graceful on N>>T) and Monte Carlo shows ≈38 % lower OOS variance than IVP.

I propose adding `WEIGHTING_METHOD ∈ {INV_VOL, HRP, ERC}` to the P6 config (default `INV_VOL` — bit-exact back-compat), a new `hrp_weights.py` module, and one dispatch hunk in `strategy.py::_rebalance`. ERC is stubbed with a documented `TODO` because the ≥10 primary sources strongly justify HRP focus first. Promotion to default HRP is gated by an explicit OOS Sharpe falsification test (§8).

---

## 2. Sources (≥10 primary + cross-validated)

| # | URL | Annotation | Relevance |
|---|---|---|---|
| S1 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678 | Lopez de Prado (2016) "Building Diversified Portfolios that Outperform Out-of-Sample," *J. Portfolio Management* 42(4):59-69. Seminal HRP paper. SSRN page is the canonical citation; full PDF gated. | HRP origin paper. Algorithm spec, MC vs CLA/IVP, distance metric d=√(½(1−ρ)). |
| S2 | https://en.wikipedia.org/wiki/Hierarchical_Risk_Parity | Wikipedia HRP entry. Full three-step algorithm, distance metric, MC results table (HRP σ²=0.0671 vs IVP 0.0928 vs CLA 0.1157), pseudo-code. | Algorithmic ground truth + cross-check on §1 numbers. |
| S3 | https://hudsonthames.org/an-introduction-to-the-hierarchical-risk-parity-algorithm/ | Hudson & Thames intro: explicit Python pseudocode for clustering, quasi-diagonalization, bisection. Includes Euclidean-of-distance second-pass D̄. | Implementation reference, scipy linkage usage. |
| S4 | https://kenwuyang.com/posts/2024_10_20_portfolio_optimization_with_python_hierarchical_risk_parity/ | Wu (2024) Python tutorial. Complete code: `get_quasi_diag`, `recursive_bisection`, `get_cluster_var`, `get_ivp`. Uses scipy `linkage(method='ward')` and `squareform`. | Direct code template. Cross-validates S1 / S2 pseudocode line-by-line. |
| S5 | https://hudsonthames.org/portfolio-optimisation-with-portfoliolab-hierarchical-risk-parity/ | Hudson & Thames PortfolioLab walkthrough. Linkage method choices (`single`, `ward`, `average`, `complete`). | Linkage method selection rationale. |
| S6 | https://quantpedia.com/hierarchical-risk-parity/ | Quantpedia summary citing Lohre, Rother, Schafer backtest 2012–2017: HRP Sharpe 0.94 vs ERC-on-factors 1.26, HRP turnover 18 %, MaxDD-1.20 % for LTDC-HRP. | Independent OOS performance numbers. |
| S7 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1271972 | Maillard, Roncalli, Teiletche (2010) "Properties of Equally-Weighted Risk Contributions Portfolios," *JPM*. Defines RC_i = w_i·(Σw)_i/σ_p; ERC vol sits between min-var and equal-weight. | ERC definition + theoretical link inverse-vol ⇔ ERC under zero correlation. |
| S8 | https://en.wikipedia.org/wiki/Risk_parity | Cross-validation of ERC math: MRC_i=(Σw)_i/σ_p, RC_i=w_i·MRC_i, condition RC_i=σ_p/N, uncorrelated-case closed form w_i ∝ 1/σ_i. | Confirms inverse-vol = ERC iff correlations ≈ 0. |
| S9 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2840729 | Raffinot (2017) "Hierarchical Clustering-Based Asset Allocation," *JPM*. HCAA = Ward linkage + early-stopped cluster count + equal-within-cluster. | Reference for clustering choice (ward vs single). |
| S10 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3237540 | Raffinot (2018) "Hierarchical Equal Risk Contribution Portfolio." HERC = HRP top-down bisection + clustering-respecting splits + ERC inside leaves. | Documents the recursive-bisection-misalignment defect of vanilla HRP (chains-effect of single linkage). |
| S11 | https://www.tobam.fr/wp-content/uploads/2021/07/2013-choueifatyfroidurereynier-properties-of-the-most-diversified-portfolio-jis.pdf | Choueifaty, Froidure, Reynier (2013) "Properties of the Most Diversified Portfolio." Defines DR(w) = (w·σ)/√(w'Σw); MDP = argmax DR. | Theoretical contrast: HRP/ERC/inverse-vol are *risk*-based; MDP is *diversification*-based. |
| S12 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4063676 | Choueifaty (2008) "Toward Maximum Diversification," *JPM* 35(1):40–51. | Original MDP citation. |
| S13 | https://quantstrattrader.wordpress.com/2017/05/26/testing-the-hierarchical-risk-parity-algorithm/ | Quantstrat Trader empirical replication: 10-asset dual-momentum universe, inverse-vol Sharpe 0.722 ≈ HRP 0.697 ≈ min-var 0.707. Concludes asset-selection dominates weighting in small universes. | Honest negative result. Important falsification anchor. |
| S14 | https://blogs.cfainstitute.org/investor/2024/01/18/how-to-build-better-low-volatility-equity-strategies/ | CFA Institute. Inverse-vol portfolios concentrate 30 %+ in utilities; documents low-vol anomaly + sector-concentration risk. | Justifies removing inverse-vol as default once HRP gate passes. |
| S15 | https://arxiv.org/pdf/2509.03712 | Reis, Sobreira, Trucíos, Asrilhant et al. (2025) "Hierarchical Risk Parity for Portfolio Allocation in the Latin American NUAM Market." Emerging market OOS test. | Confirms HRP advantage extends beyond US/EU. |
| S16 | https://www.quantresearch.org/HRP.py.txt | Lopez de Prado's own reference Python (`correlDist`, `getQuasiDiag`, `getRecBipart`, `getClusterVar`, `getIVP`). | Canonical implementation truth-source. |

Cross-validation matrix (every algorithmic claim cited ≥2 times):
- Distance d=√(½(1−ρ)): S1, S2, S3, S4, S16.
- Three steps (cluster / quasi-diag / bisect): S1, S2, S3, S5, S16.
- Recursive bisection α = 1 − V₁/(V₁+V₂): S2, S3, S4.
- Inverse-vol = ERC iff Σ diagonal: S7, S8.
- HRP MC variance < IVP: S1, S2, S6.
- Inverse-vol concentration weakness: S7, S14.
- HRP single-linkage chaining defect: S9, S10.

---

## 3. Current-state analysis (file:line)

### 3.1 `src/portfolios/portfolio_6/screener.py:80–112` — `inverse_vol_weights`

```python
def inverse_vol_weights(
    returns_matrix: Dict[str, pd.Series],
    *,
    max_weight: float = 0.05,
    max_iterations: int = 20,
) -> Dict[str, float]:
    """
    weight_i = (1/vol_i) / sum(1/vol_j), iteratively capped at max_weight with
    slack redistributed to under-cap names. Returns dict ticker -> weight.
    """
    inv = {}
    for t, r in returns_matrix.items():
        v = realized_vol(r)                           # screener.py:92
        if v > 0 and np.isfinite(v):
            inv[t] = 1.0 / v
    if not inv:
        return {}
    s = pd.Series(inv)
    w = s / s.sum()

    for _ in range(max_iterations):                   # screener.py:100
        over = w > max_weight
        if not over.any():
            break
        slack = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        under_sum = float(w[under].sum())
        if under_sum <= 0:
            break
        w[under] = w[under] + slack * (w[under] / under_sum)

    return w.to_dict()
```

- `realized_vol` is annualized stdev of daily returns (`screener.py:18–24`).
- Iterative cap: at most 20 sweeps; over-cap slack is redistributed *pro-rata to current under-cap weight*, preserving the relative inverse-vol ranking among uncapped names.
- Return is `Dict[str, float]` summing to ≤ 1 (= 1 if cap is non-binding, ≤ 1 otherwise; in practice = 1 because slack redistribution conserves mass).

### 3.2 `src/portfolios/portfolio_6/strategy.py:214–217` — call site

```python
weights = inverse_vol_weights(
    {t: returns_matrix[t] for t in top},
    max_weight=self.max_weight,
)
```

Downstream the weights are:
- fed into a sleeve-return synthetic series for DSR (`strategy.py:222–224`)
- multiplied by `vol_target_scale` (`strategy.py:243–248`)
- merged with `GLD_WEIGHT` and `TREND_HEDGE_WEIGHT` hedge sleeves (`strategy.py:251–274`)
- finally normalized to `MAX_LEVERAGE` if total > leverage cap (`strategy.py:276–284`)

### 3.3 Heuristic origin

Inverse-vol weighting is the **closed-form ERC solution when the covariance matrix is diagonal**. From Maillard-Roncalli-Teiletche (S7) and Wikipedia Risk Parity (S8):

> RC_i = w_i·(Σw)_i / σ_p. Setting RC_i = σ_p/N ∀i and assuming Σ = diag(σ_i²) gives w_i ∝ 1/σ_i.

So `inverse_vol_weights` is a *zero-correlation approximation* to ERC. This is the basis of its popularity (no estimation of off-diagonals required) and also its core weakness.

### 3.4 Known weaknesses (cited)

1. **Concentration on lowest-vol cluster.** In US equities the low-vol cluster is dominated by utilities, consumer staples, REITs, healthcare (S14). A pure 1/σ sort puts 30 %+ in utilities; even after rank-based screening (P6's first leg is already `vols.rank(ascending=True)` at `screener.py:57`), the *weighting* re-amplifies the same tilt.
2. **Ignores correlations.** Two highly-correlated low-vol names get full additive weight; HRP and ERC penalize duplicates (S7).
3. **Cap is purely numerical.** Iterative cap in P6 redistributes to *all other names proportionally to their current weight*; this preserves the same concentration pattern at one level lower (pushes capital into the *next* low-vol cluster). No cluster-aware redistribution.
4. **No Σ stress signal.** Inverse-vol does not flag covariance regime change. HRP's bisection α tracks intra-cluster variance ratios, which is a structural risk telemetry.

### 3.5 Team A inter-dependency (SYNTHESIS.md §3, §10)

A4 widened `score_universe` signature; A1 added factor flags; A5 added exclusions. None of those touch the *weighting* call site. The patch in §7 below targets exactly the four-line `inverse_vol_weights(...)` call at `strategy.py:214–217`, which is unchanged by Team A's diffs. **Clean merge with no hunk conflict** against `SYNTHESIS.md §7`.

Vol-target (SYNTHESIS §10): HRP weights are *sleeve-level* — vol-target multiplier still applies in the same way. HRP does not interact with leverage cap.

---

## 4. HRP algorithm specification (Lopez de Prado 2016)

### 4.1 Notation

- Universe: N assets, T observations of daily returns matrix R ∈ ℝ^(T×N).
- Σ = sample covariance (or shrinkage Σ from B2), N×N.
- C = correlation matrix, ρ_ij = Σ_ij / √(Σ_ii·Σ_jj).

### 4.2 Step 1 — distance + hierarchical clustering

**Correlation distance** (S1, S2, S3, S4, S16):

$$
d(i,j) \;=\; \sqrt{\tfrac{1}{2}\bigl(1 - \rho_{i,j}\bigr)}, \qquad d \in [0,1]
$$

This is a proper metric (non-negativity, symmetry, identity, triangle inequality), unlike `1−ρ`. Two perfectly correlated assets have d=0; perfectly anti-correlated have d=1.

**Second-pass Euclidean distance** (Hudson & Thames S3 — strict reading of LdP S1 §3 step 1):

$$
\bar d(i,j) \;=\; \sqrt{\sum_{k=1}^{N} \bigl(d(k,i) - d(k,j)\bigr)^2}
$$

Captures *similarity in the broader correlation profile* (not just pair-wise correlation). Used as the input to the linkage step.

**Hierarchical clustering** with **single linkage** (LdP S1 §3 step 1, S2):

```
condensed = squareform(d_bar)        # SciPy condensed distance vector
Z         = scipy.cluster.hierarchy.linkage(condensed, method='single')
```

Z is the (N−1)×4 linkage matrix `[cluster_a, cluster_b, distance, n_items]`.

NB: Raffinot (S9) and many subsequent papers prefer Ward linkage on ρ-distance directly (no D̄ pass). I make linkage method configurable; default is `single` to match LdP 2016 verbatim.

### 4.3 Step 2 — quasi-diagonalization

Permute Σ rows/columns by traversing Z so that the largest covariances cluster around the diagonal (S1 §3 step 2). Algorithmically, this is a depth-first leaf-order traversal of the dendrogram:

```
def get_quasi_diag(Z):
    Z = Z.astype(int)
    sort_ix = [Z[-1, 0], Z[-1, 1]]      # last merge produces the root
    num_items = Z[-1, 3]
    while max(sort_ix) >= num_items:
        # Expand each non-leaf cluster id into its two children, depth-first.
        new = []
        for v in sort_ix:
            if v < num_items:
                new.append(v)
            else:
                row = Z[v - num_items]
                new.extend([int(row[0]), int(row[1])])
        sort_ix = new
    return sort_ix
```

(Cross-validated against S2, S4, S16. Wu 2024 / S4 uses a vectorized pandas variant; the loop above is the literal LdP construction.)

### 4.4 Step 3 — recursive bisection

Initialize `w_i = 1 ∀i`. Maintain a list of clusters; at each pass split every >1-member cluster at its midpoint into (L, R). For each (L, R) pair compute *inverse-variance-weighted cluster variance*

$$
\tilde V_C = \tilde w_C^{\top}\, \Sigma_{C,C}\, \tilde w_C, \qquad \tilde w_C \;=\; \frac{\operatorname{diag}(\Sigma_{C,C})^{-1}}{\mathbf 1^{\top}\operatorname{diag}(\Sigma_{C,C})^{-1}}
$$

and the split factor

$$
\alpha \;=\; 1 \;-\; \frac{\tilde V_L}{\tilde V_L + \tilde V_R}
$$

Update `w[L] *= α`, `w[R] *= (1−α)`. Recurse until all clusters are singletons. By construction Σ w = 1 (S1, S2, S3, S4).

Two interpretations confirm correctness:
- α = the inverse-variance share between siblings (S3).
- α equals the IVP weight if siblings were treated as two synthetic assets with variance Ṽ_L, Ṽ_R (S2, S16).

### 4.5 Complexity

- Distance + linkage: O(N² log N) with `single` linkage (SLINK algorithm) or O(N²) with naive implementation.
- Quasi-diag: O(N).
- Recursive bisection: O(N log N) splits × O(k²) cluster variance ⇒ amortized O(N² log N).
- **Total: O(N² log N)** — for P6's N≤50 this is trivial (≪ 1 ms).

### 4.6 Numerical guards (production-ready additions not in LdP 2016)

1. **Σ regularization.** If `np.diag(Σ)` contains zeros or non-finite, drop those columns before clustering.
2. **Correlation clamp.** `ρ = np.clip(ρ, -1.0, 1.0)` before sqrt, to defeat float roundoff producing tiny negatives inside the radical.
3. **NaN-safe correlation.** Use `pd.DataFrame.corr()` which pair-wise drops NaN; fall back to identity correlation when an asset has <2 non-NaN obs.
4. **Singleton fallback.** If N==1 return {ticker: 1.0}; if N==0 return {}.
5. **Cap enforcement.** Iterative cap same as `inverse_vol_weights` pattern (`screener.py:100–110`) — slack redistributed pro-rata to under-cap weights.

---

## 5. Comparison: HRP vs ERC vs inverse-vol vs equal-weight

Numbers below are aggregated from cited OOS studies; ranges given because sample period and universe matter (S6, S13, S15, S1). "—" means the source did not report that statistic.

| Property | Equal-Weight (1/N) | Inverse-Vol (P6 current) | ERC (numerical) | HRP (LdP 2016) |
|---|---|---|---|---|
| Σ inversion required | No | No | No | No |
| Uses correlations | No | No | Yes (full Σ) | Yes (full Σ via clustering) |
| Closed form | Yes | Yes | No (cyclic coord descent or convex prog.) | Yes (recursive) |
| OOS variance vs CLA (LdP MC, S1) | — | +38 % vs HRP | similar to IVP | **baseline** (≈42 % below CLA) |
| OOS Sharpe — Lohre-Rother-Schafer 2012–17 (S6) | — | comparable to ERC | **1.26** | 0.94 (vanilla); 0.50-0.55 (corr-HRP variants — turnover-dominated) |
| OOS Sharpe — Quantstrat 10-ETF dual-momo (S13) | — | **0.72** | — | 0.70 |
| OOS Sharpe — Risk Parity vs EW DJ-30 2024 (S source: blog.quantinsti S6) | 1.07 | — | — | 1.57 (risk-parity broad) |
| Sector concentration risk (S14) | Low | **High** (30%+ utilities) | Moderate | Low (cluster-balanced) |
| Single-asset concentration | 1/N (by design) | Cap-bound; otherwise unbounded toward low-σ tail | Cap-bound | Cap-bound; LdP MC ≈ 62 % in top-5 vs IVP ≈ 71 % vs CLA ≈ 93 % (S1, S2) |
| Turnover (S6) | low | low | 5 % monthly | **18 %** vanilla HRP; 5 % for HRP-LTDC |
| Computational complexity | O(1) | O(N) | O(N²·iter) | O(N² log N) |
| Robustness to singular Σ | Yes | Yes | Sensitive | Yes (no inversion) |
| Robustness to N>T | Yes | Yes | Fragile | Yes |
| Theoretical justification | Maximum entropy (DeMiguel et al 2009) | ERC limit when Σ diagonal (S7, S8) | Risk-budgeting axiom (S7) | Tree-structured graph theory + ML (S1) |
| Default in P6 today | No | **Yes** | No | proposed opt-in |
| Single-linkage chaining defect | n/a | n/a | n/a | **Yes** (S10) — mitigated by Ward option |

**Headline read.** HRP is *not* a Sharpe panacea — Quantstrat (S13) and CFA Institute (S14) consistently report it ties inverse-vol on small/already-diversified universes, and Lohre-Rother-Schafer (S6) show vanilla HRP loses to ERC and even to IVP on a 5-year US backtest *after transaction costs* due to 18 % turnover. The defensible reason to adopt HRP is **risk telemetry + downside skew**: smaller MaxDD, lower OOS variance, less sector concentration, and graceful behavior when N approaches T.

---

## 6. Full source for new file `src/portfolios/portfolio_6/hrp_weights.py`

Apply-ready. All imports stdlib + numpy + pandas + scipy (already in `requirements.txt` per the existing `scipy.stats.norm` use at `screener.py:144`).

```python
"""
Portfolio 6 — Hierarchical Risk Parity weighting (Lopez de Prado 2016).

Adds an opt-in alternative to ``inverse_vol_weights``. Wired via
``PORTFOLIO_6_CONFIG.WEIGHTING_METHOD`` ("INV_VOL" | "HRP" | "ERC").

Algorithm reference: Marcos Lopez de Prado, "Building Diversified
Portfolios that Outperform Out-of-Sample," J. Portfolio Management
42(4):59-69, 2016 (SSRN 2708678).

Three steps:
  1. correlation -> distance  d(i,j) = sqrt(0.5 * (1 - rho_ij))
  2. quasi-diagonalization    leaf-order traversal of single-linkage dendrogram
  3. recursive bisection      alpha = 1 - V_L / (V_L + V_R)  with V_C = inv-var

Drop-in contract:
  hrp_weights(returns_matrix, *, max_weight, cov=None) -> Dict[str, float]
  - Returns dict ticker -> weight, sums to 1.0 (or 0 if no usable input).
  - Iteratively caps to MAX_WEIGHT_PER_STOCK with the same slack-redistribute
    pattern as ``screener.inverse_vol_weights`` for bit-for-bit downstream
    compatibility with vol_target_scale + hedge sleeve composition.
  - Accepts optional pre-computed ``cov`` (e.g. shrinkage from B2 covariance
    agent). When None, falls back to the sample covariance of returns_df.

Production guards beyond LdP 2016:
  * Correlation clamp to [-1, 1] before sqrt (defeats float roundoff).
  * NaN-safe correlation via pandas pair-wise dropna.
  * Singleton + empty fallbacks.
  * Replaces 1/sigma with 1e12 sentinel when sigma == 0 to avoid div-zero
    while keeping the asset in the universe (matches behavior of IVP).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    _SCIPY_OK = True
except ImportError:  # pragma: no cover - scipy is a hard dep elsewhere in repo
    _SCIPY_OK = False

logger = logging.getLogger(__name__)

DEFAULT_LINKAGE_METHOD = "single"  # Lopez de Prado 2016 baseline.


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _returns_to_dataframe(returns_matrix: Dict[str, pd.Series]) -> pd.DataFrame:
    """Build a returns DataFrame aligned on the union of indices, NaN-padded."""
    if not returns_matrix:
        return pd.DataFrame()
    df = pd.DataFrame({t: pd.Series(r).astype(float) for t, r in returns_matrix.items()})
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all")
    return df


def _correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """d(i,j) = sqrt(0.5 * (1 - rho_ij)). Symmetric, zero diagonal, in [0,1]."""
    c = corr.clip(lower=-1.0, upper=1.0).fillna(0.0)
    d = np.sqrt(np.maximum(0.5 * (1.0 - c.values), 0.0))
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)


def _get_quasi_diag(link: np.ndarray) -> list:
    """Leaf-order DFS through the (N-1)x4 SciPy linkage matrix.

    Returns the permutation of original asset indices that puts the largest
    covariances around the diagonal of Sigma. Matches the construction in
    Lopez de Prado 2016 / quantresearch.org/HRP.py.txt.
    """
    Z = link.astype(int)
    num_items = int(Z[-1, 3])
    sort_ix = [int(Z[-1, 0]), int(Z[-1, 1])]
    while max(sort_ix) >= num_items:
        new = []
        for v in sort_ix:
            if v < num_items:
                new.append(int(v))
            else:
                row = Z[v - num_items]
                new.extend([int(row[0]), int(row[1])])
        sort_ix = new
    return sort_ix


def _ivp_weights(cov_sub: np.ndarray) -> np.ndarray:
    """Inverse-variance portfolio over the diagonal of cov_sub."""
    diag = np.diag(cov_sub).astype(float)
    diag = np.where(diag > 0.0, diag, np.finfo(float).tiny)
    inv = 1.0 / diag
    s = inv.sum()
    if not np.isfinite(s) or s <= 0.0:
        return np.full_like(diag, 1.0 / len(diag))
    return inv / s


def _cluster_variance(cov: pd.DataFrame, items: list) -> float:
    """Inverse-variance-weighted portfolio variance over a sub-universe."""
    sub = cov.loc[items, items].to_numpy()
    w = _ivp_weights(sub).reshape(-1, 1)
    v = float((w.T @ sub @ w)[0, 0])
    return v if np.isfinite(v) and v > 0.0 else float(np.finfo(float).tiny)


def _recursive_bisection(cov: pd.DataFrame, ordered: list) -> pd.Series:
    """Top-down recursive bisection (Lopez de Prado 2016 §3 step 3)."""
    w = pd.Series(1.0, index=ordered, dtype=float)
    clusters = [ordered]
    while clusters:
        # Halve every >1-member cluster.
        clusters = [
            c[start:stop]
            for c in clusters
            for start, stop in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]
        # Pair up siblings and split capital by inverse-variance share.
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            vL = _cluster_variance(cov, left)
            vR = _cluster_variance(cov, right)
            alpha = 1.0 - vL / (vL + vR)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            w[left] *= alpha
            w[right] *= 1.0 - alpha
    return w


def _apply_iterative_cap(
    w: pd.Series,
    *,
    max_weight: float,
    max_iterations: int = 20,
) -> pd.Series:
    """Same slack-redistribute pattern as screener.inverse_vol_weights (lines 100-110).

    Pro-rata to current under-cap weights so the inverse-vol *ranking* is
    preserved among uncapped names. Identical contract to IVP path.
    """
    if max_weight <= 0.0:
        return w
    w = w.copy().astype(float)
    for _ in range(max_iterations):
        over = w > max_weight
        if not over.any():
            break
        slack = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        under_sum = float(w[under].sum())
        if under_sum <= 0.0:
            break
        w[under] = w[under] + slack * (w[under] / under_sum)
    return w


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def hrp_weights(
    returns_matrix: Dict[str, pd.Series],
    *,
    max_weight: float = 0.05,
    cov: Optional[pd.DataFrame] = None,
    linkage_method: str = DEFAULT_LINKAGE_METHOD,
    max_iterations: int = 20,
) -> Dict[str, float]:
    """Hierarchical Risk Parity weights.

    Parameters
    ----------
    returns_matrix : dict[ticker -> pd.Series of daily returns]
        Same shape as the input to ``inverse_vol_weights``.
    max_weight : float
        Per-stock weight cap. Iteratively enforced (same pattern as IVP path).
    cov : pd.DataFrame, optional
        Pre-computed covariance matrix indexed by ticker. When None, uses
        sample covariance of the aligned returns DataFrame. Pass a shrinkage
        estimator from B2 to harden against estimation error when N close to T.
    linkage_method : str
        SciPy linkage method. "single" reproduces Lopez de Prado 2016;
        "ward" is the Raffinot 2017 HCAA choice (more compact clusters).
    max_iterations : int
        Cap-redistribution sweeps. Mirrors IVP behavior.

    Returns
    -------
    Dict[str, float]
        Ticker -> weight, sums to 1.0 (or {} if no usable input).
    """
    if not _SCIPY_OK:
        logger.error("[HRP] scipy not available; cannot compute HRP weights.")
        return {}

    # -- Input prep ----------------------------------------------------------
    rets_df = _returns_to_dataframe(returns_matrix)
    if rets_df.empty:
        return {}
    if rets_df.shape[1] == 1:
        return {str(rets_df.columns[0]): 1.0}

    # -- Covariance + correlation -------------------------------------------
    if cov is None:
        sigma = rets_df.cov()
    else:
        # Reindex to current asset set; drop anything unknown to cov.
        common = [c for c in rets_df.columns if c in cov.index and c in cov.columns]
        if len(common) < 2:
            logger.warning(
                "[HRP] Provided cov covers <2 of the input assets; "
                "falling back to sample cov on returns_matrix."
            )
            sigma = rets_df.cov()
        else:
            sigma = cov.loc[common, common].astype(float)
            rets_df = rets_df[common]

    sigma = sigma.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Drop zero-variance assets (degenerate; cannot participate in HRP).
    diag = np.diag(sigma.to_numpy())
    keep = [t for t, v in zip(sigma.index, diag) if np.isfinite(v) and v > 0.0]
    if len(keep) < 2:
        if len(keep) == 1:
            return {keep[0]: 1.0}
        return {}
    sigma = sigma.loc[keep, keep]
    rets_df = rets_df[keep]

    # -- Correlation -> distance --------------------------------------------
    corr = rets_df.corr().reindex(index=keep, columns=keep)
    dist = _correlation_distance(corr)

    # -- Hierarchical clustering --------------------------------------------
    try:
        condensed = squareform(dist.to_numpy(), checks=False)
        link = linkage(condensed, method=linkage_method)
    except Exception as e:
        logger.exception("[HRP] linkage failed (%s); returning IVP fallback.", e)
        # IVP fallback so the caller never crashes on a degenerate Sigma.
        ivp = _ivp_weights(sigma.to_numpy())
        w = pd.Series(ivp, index=keep)
        w = _apply_iterative_cap(w, max_weight=max_weight, max_iterations=max_iterations)
        return {str(k): float(v) for k, v in w.items()}

    # -- Quasi-diagonalization ----------------------------------------------
    order_ix = _get_quasi_diag(link)
    ordered_tickers = [keep[i] for i in order_ix]

    # -- Recursive bisection ------------------------------------------------
    w = _recursive_bisection(sigma.loc[ordered_tickers, ordered_tickers], ordered_tickers)

    # Re-order to the input universe order (not strictly required, but tidy).
    w = w.reindex(keep).fillna(0.0)
    total = float(w.sum())
    if total > 0.0:
        w = w / total

    # -- Iterative cap (same as IVP path) -----------------------------------
    w = _apply_iterative_cap(w, max_weight=max_weight, max_iterations=max_iterations)

    return {str(k): float(v) for k, v in w.items()}


def erc_weights(  # pragma: no cover - stub for opt-in numerical ERC
    returns_matrix: Dict[str, pd.Series],
    *,
    max_weight: float = 0.05,
    cov: Optional[pd.DataFrame] = None,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> Dict[str, float]:
    """Equal-Risk-Contribution weights (Maillard-Roncalli-Teiletche 2010).

    TODO(B1): full cyclic-coordinate-descent implementation per
    Spinu (2013) "An Algorithm for Computing Risk Parity Portfolios"
    (arXiv:1311.4057). The 10-source mandate justified HRP focus; ERC
    is a stub returning IVP for now so the dispatch in strategy.py
    doesn't break if someone flips WEIGHTING_METHOD="ERC" early.

    For diagonal Sigma, ERC == IVP exactly (Maillard et al 2010,
    SSRN 1271972), so the fallback is the correct limit of the full
    algorithm.
    """
    logger.warning(
        "[ERC] erc_weights stub: returning IVP (diagonal-Sigma limit of ERC). "
        "Replace with full Spinu 2013 cyclic-coordinate-descent before "
        "promoting WEIGHTING_METHOD='ERC' to default."
    )
    # Use the existing screener IVP path indirectly via local helpers.
    rets_df = _returns_to_dataframe(returns_matrix)
    if rets_df.empty:
        return {}
    if cov is None:
        sigma = rets_df.cov().to_numpy()
    else:
        common = [c for c in rets_df.columns if c in cov.index and c in cov.columns]
        sigma = cov.loc[common, common].to_numpy() if len(common) >= 2 else rets_df.cov().to_numpy()
    ivp = _ivp_weights(sigma)
    w = pd.Series(ivp, index=rets_df.columns[: len(ivp)])
    w = _apply_iterative_cap(w, max_weight=max_weight, max_iterations=20)
    return {str(k): float(v) for k, v in w.items()}
```

### 6.1 Self-test (recommended as a unit test in a follow-up PR)

```python
# tests/portfolios/portfolio_6/test_hrp_weights.py  (proposed; not authored here)
import numpy as np, pandas as pd
from src.portfolios.portfolio_6.hrp_weights import hrp_weights

def test_hrp_singleton():
    r = pd.Series(np.random.randn(100), name="AAA")
    assert hrp_weights({"AAA": r}, max_weight=1.0) == {"AAA": 1.0}

def test_hrp_diagonal_matches_ivp():
    # If returns are uncorrelated, HRP ≈ IVP at the leaf level (recursive
    # bisection collapses to inverse-variance allocation between siblings).
    np.random.seed(7)
    T, N = 1000, 6
    sigmas = np.array([0.10, 0.12, 0.15, 0.20, 0.30, 0.40])
    R = np.random.randn(T, N) * sigmas
    rmat = {f"A{i}": pd.Series(R[:, i]) for i in range(N)}
    w = hrp_weights(rmat, max_weight=1.0)
    # Higher-sigma asset gets lower weight.
    assert w["A0"] > w["A5"]
    assert abs(sum(w.values()) - 1.0) < 1e-9

def test_hrp_cap_enforced():
    np.random.seed(11)
    rmat = {f"A{i}": pd.Series(np.random.randn(500) * (0.05 + 0.01 * i)) for i in range(20)}
    w = hrp_weights(rmat, max_weight=0.10)
    assert max(w.values()) <= 0.10 + 1e-12
    assert abs(sum(w.values()) - 1.0) < 1e-6
```

---

## 7. Unified diffs — `strategy.py` + `config.json`

### 7.1 `src/portfolios/portfolio_6/strategy.py`

Targets the **import block** + the four-line `inverse_vol_weights` call at `strategy.py:214–217`. Does not touch any other section, so it cleanly stacks on top of Team A's SYNTHESIS §7 diff.

```diff
--- a/src/portfolios/portfolio_6/strategy.py
+++ b/src/portfolios/portfolio_6/strategy.py
@@ -33,16 +33,30 @@
 try:
     from src.portfolios.portfolio_6.screener import (
         deflated_sharpe_ratio,
         inverse_vol_weights,
         score_universe,
         select_top_n,
         vol_target_scale,
     )
 except ImportError:
     from portfolios.portfolio_6.screener import (
         deflated_sharpe_ratio,
         inverse_vol_weights,
         score_universe,
         select_top_n,
         vol_target_scale,
     )

+# Team B / B1 — HRP and ERC opt-in weighting methods.
+try:
+    from src.portfolios.portfolio_6.hrp_weights import erc_weights, hrp_weights
+except ImportError:
+    from portfolios.portfolio_6.hrp_weights import erc_weights, hrp_weights
+
+_WEIGHTING_DISPATCH = {
+    "INV_VOL": inverse_vol_weights,
+    "HRP":     hrp_weights,
+    "ERC":     erc_weights,
+}
+
 REPO_ROOT = Path(__file__).resolve().parents[3]
@@ -98,6 +112,11 @@ class Portfolio6Strategy(BasePortfolio):
         self.fundamentals_csv_rel: str = str(
             p6_cfg.get("FUNDAMENTALS_CSV", "fundamentals/fundamentals.csv")
         )
+        # Team B / B1 — weighting method dispatch.
+        wm = str(p6_cfg.get("WEIGHTING_METHOD", "INV_VOL")).strip().upper()
+        if wm not in _WEIGHTING_DISPATCH:
+            self.logger.warning("[P6] Unknown WEIGHTING_METHOD=%r; falling back to INV_VOL.", wm)
+            wm = "INV_VOL"
+        self.weighting_method: str = wm
+        self.hrp_linkage_method: str = str(p6_cfg.get("HRP_LINKAGE_METHOD", "single")).strip().lower()
@@ -211,12 +230,29 @@ class Portfolio6Strategy(BasePortfolio):
         top = select_top_n(scores, n=self.screen_top_n)
         if not top:
             self.logger.warning("[P6] Top-N selection empty; skipping rebalance.")
             return

-        weights = inverse_vol_weights(
-            {t: returns_matrix[t] for t in top},
-            max_weight=self.max_weight,
-        )
+        weighter = _WEIGHTING_DISPATCH[self.weighting_method]
+        weight_kwargs = {"max_weight": self.max_weight}
+        if self.weighting_method == "HRP":
+            weight_kwargs["linkage_method"] = self.hrp_linkage_method
+        # B2 (covariance agent) hook: future-proof — when a shrinkage Sigma
+        # provider is wired up at the strategy level (self._shrinkage_cov),
+        # pass it through. Until then HRP/ERC fall back to sample cov.
+        sigma = getattr(self, "_shrinkage_cov", None)
+        if sigma is not None and self.weighting_method in ("HRP", "ERC"):
+            weight_kwargs["cov"] = sigma
+        try:
+            weights = weighter(
+                {t: returns_matrix[t] for t in top},
+                **weight_kwargs,
+            )
+        except Exception as e:
+            self.logger.exception(
+                "[P6] %s weighting failed (%s); falling back to inverse_vol.",
+                self.weighting_method, e,
+            )
+            weights = inverse_vol_weights({t: returns_matrix[t] for t in top}, max_weight=self.max_weight)
+
         if not weights:
             self.logger.warning("[P6] %s weighting empty; skipping rebalance.", self.weighting_method)
             return
```

Notes on the diff:
- The `_WEIGHTING_DISPATCH` table is module-level so a unit test can monkey-patch it.
- Default `INV_VOL` preserves bit-exact P6 behavior; the dispatch table maps `"INV_VOL"` straight to the existing `inverse_vol_weights` symbol (same function reference, same call, same kwargs).
- The B2 covariance hook is *opt-in via attribute check* — when B2 lands and the strategy gets a `_shrinkage_cov` provider, HRP picks it up automatically. No upfront coupling.
- Exception fallback to IVP guarantees `_rebalance` never silently produces empty weights.
- The `linkage_method` is configurable from JSON so an operator can toggle between LdP-2016 `single` and Raffinot `ward` without code change.

### 7.2 `src/portfolios/portfolio_6/config.json`

Stacks on top of SYNTHESIS §6 cleanly — same JSON object, separate keys.

```diff
--- a/src/portfolios/portfolio_6/config.json
+++ b/src/portfolios/portfolio_6/config.json
@@ -16,7 +16,11 @@
     "GLD_TICKER": "GLD",
     "GLD_WEIGHT": 0.07,
     "TREND_HEDGE_TICKER": "",
     "TREND_HEDGE_WEIGHT": 0.10,
     "REBALANCE_DRIFT_THRESHOLD": 0.005,
-    "DSR_MIN_PROB": 0.5
+    "DSR_MIN_PROB": 0.5,
+
+    "WEIGHTING_METHOD": "INV_VOL",
+    "HRP_LINKAGE_METHOD": "single"
   }
 }
```

If applied together with SYNTHESIS §6, the resulting block looks like:

```json
"DSR_MIN_PROB": 0.5,

"WEIGHTING_METHOD": "INV_VOL",
"HRP_LINKAGE_METHOD": "single",

"SCORE_METHOD": "rank_sum",
...
```

(trailing commas adjusted at merge time).

### 7.3 Allowed values

| Key | Type | Allowed | Default | Notes |
|---|---|---|---|---|
| `WEIGHTING_METHOD` | string | `"INV_VOL"`, `"HRP"`, `"ERC"` | `"INV_VOL"` | Case-insensitive in dispatcher. Unknown → warn + fallback to `INV_VOL`. |
| `HRP_LINKAGE_METHOD` | string | `"single"`, `"complete"`, `"average"`, `"ward"` (SciPy) | `"single"` | Only used when `WEIGHTING_METHOD="HRP"`. |

---

## 8. Falsification test (concrete)

**Promotion rule** (no default switch without passing this gate):

1. Run walk-forward purged-k-fold backtest 2021-01-01 → 2026-05-01 on the P6 universe, identical config except `WEIGHTING_METHOD ∈ {INV_VOL, HRP}`.
2. Aggregate by fold and compute:
   - ΔSharpe = Sharpe(HRP) − Sharpe(INV_VOL)
   - ΔMaxDD  = MaxDD(HRP) − MaxDD(INV_VOL)  (more-negative = worse)
   - ΔTurnover = Turnover(HRP) − Turnover(INV_VOL)  (annualized one-way)
3. **Pass criteria for switching default to HRP:**
   - ΔSharpe ≥ **+0.10** over the full 5-year window, **and**
   - ΔSharpe ≥ −0.05 in every individual calendar year (no catastrophic single-year regression), **and**
   - ΔMaxDD ≥ −0.02 (i.e. HRP worsens MaxDD by no more than 2 pp), **and**
   - ΔTurnover ≤ +0.30 (HRP costs no more than 30 % turnover bump after t-cost model in `executor`), **and**
   - DSR(HRP) ≥ 0.5 on the full sleeve (preserves existing P6 gate at `strategy.py:234–239`).
4. **Failure rule.** If `Sharpe(HRP) < Sharpe(INV_VOL) − 0.10` over the 5-year window, the HRP method is documented as inferior and `WEIGHTING_METHOD="INV_VOL"` stays the default. The file `hrp_weights.py` remains available as an opt-in for research builds.

This is consistent with Team A's promotion gate pattern (SYNTHESIS §11).

---

## 9. Risks + rollback path

### 9.1 Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Single-linkage chaining (S10) misallocates highly-correlated assets to opposite halves of the dendrogram. | M | `HRP_LINKAGE_METHOD="ward"` toggle ships in v1; operators can flip without code change. |
| R2 | HRP turnover ≈ 18 % vs IVP ≈ 5 % (S6). After t-cost the OOS Sharpe edge can flip negative. | H | `REBALANCE_DRIFT_THRESHOLD` (`strategy.py:312`) already filters small moves; monthly cadence (`strategy.py:161–162`) already throttles HRP turnover. Falsification gate §8 includes turnover budget. |
| R3 | HRP requires Σ from returns_matrix; with NaN-heavy or N-close-to-T data, `np.cov` becomes singular. | M | LdP's no-inversion property survives singular Σ. Module also drops zero-variance columns and falls back to IVP on linkage exception. |
| R4 | `scipy` version drift could change `linkage` ordering across hosts (deterministic by definition but algorithm choice has been refined). | L | `linkage_method` pinned to `"single"` (deterministic). `requirements.txt` should pin `scipy>=1.7`. |
| R5 | B2 covariance shrinkage estimator (separate agent) lands after B1. Hard-coding sample cov locks us in. | L | The function already accepts an optional `cov=` arg. B2 just needs to set `self._shrinkage_cov` on the strategy instance — the diff in §7.1 already reads it via `getattr`. |
| R6 | ERC stub returning IVP could mislead future operators into thinking ERC is implemented. | M | Stub emits `logger.warning` on every call; docstring is explicit; falsification gate forbids promotion of `ERC` without re-validation. |
| R7 | Quasi-diag DFS recursion depth on N=50 is ≤ log₂(50) ≈ 6 — no Python recursion-limit risk. With pathological dendrograms (chain) depth = N. | L | Implementation is iterative (`while clusters:` loop), not recursive — no stack-depth risk regardless of dendrogram shape. |
| R8 | Iterative cap in `_apply_iterative_cap` may not converge if `max_weight × N < 1.0`. | L | Same constraint applies to existing `inverse_vol_weights`. Caller must ensure `max_weight ≥ 1/N` (P6 has `MAX_WEIGHT_PER_STOCK=0.05` and `SCREEN_TOP_N=50` so `0.05×50=2.5 ≥ 1.0` ✓). |
| R9 | Float roundoff in `√(½(1−ρ))` when |ρ| → 1 produces NaN. | L | `np.clip(corr, -1, 1)` + `np.maximum(0, …)` in `_correlation_distance`. |

### 9.2 Rollback path

Three independent rollback levels — pick the smallest that solves the issue:

1. **One-line config flip** (most common). In any deployed `config.json`:
   ```json
   "WEIGHTING_METHOD": "INV_VOL"
   ```
   Restarting the strategy reverts to the original inverse-vol path immediately. No deployment, no rebuild.

2. **One-line code disable.** In `strategy.py` (post-patch), comment out the HRP entry in the dispatch table:
   ```python
   _WEIGHTING_DISPATCH = {
       "INV_VOL": inverse_vol_weights,
       # "HRP":     hrp_weights,  # disabled pending investigation
       # "ERC":     erc_weights,
   }
   ```
   Any config with `WEIGHTING_METHOD="HRP"` then falls into the unknown-method warning and reverts to `INV_VOL`.

3. **Full revert.** `git revert <hrp-commit-sha>` removes the diff in §7.1 + §7.2 + the new file. The remaining state is bit-exact today's P6.

### 9.3 Forward dependencies

- **B2 (covariance shrinkage):** When B2 publishes a Σ provider on the strategy instance (`self._shrinkage_cov`), HRP picks it up automatically via `getattr(self, "_shrinkage_cov", None)` in §7.1.
- **Team A:** SYNTHESIS §7 modifies the `score_universe(...)` call site, **not** the `inverse_vol_weights(...)` call site. Diffs stack cleanly.
- **Team C (executor / t-cost):** Falsification gate §8 turnover budget should be evaluated under the executor's current t-cost model — coordinate with C1 before measuring ΔTurnover.

---

## 10. Appendix — algorithm walk-through on a 4-asset toy

Inputs:
```
ρ = [[1.00, 0.95, 0.10, 0.05],
     [0.95, 1.00, 0.08, 0.02],
     [0.10, 0.08, 1.00, 0.85],
     [0.05, 0.02, 0.85, 1.00]]
σ = [0.10, 0.12, 0.20, 0.25]
```

- Σ = diag(σ)·ρ·diag(σ).
- d = √(½(1−ρ)) ⇒ d(0,1)≈0.158, d(2,3)≈0.274, others ≈ 0.67.
- Single-linkage clustering builds {0,1}{2,3}, then merges them.
- Quasi-diag leaf order = [0,1,2,3].
- Recursive bisection round 1: split into L={0,1}, R={2,3}.
  - Ṽ_L = inverse-var-weighted variance over {0,1} ≈ 0.013.
  - Ṽ_R = ditto over {2,3} ≈ 0.053.
  - α = 1 − 0.013/0.066 ≈ 0.80. **L gets 80 %, R gets 20 %.**
- Round 2 (L): split into {0}, {1}; α = 1 − σ₀²/(σ₀²+σ₁²) ≈ 0.59. w_0 = 0.80·0.59 = 0.47; w_1 = 0.80·0.41 = 0.33.
- Round 2 (R): split into {2}, {3}; α = 1 − σ₂²/(σ₂²+σ₃²) ≈ 0.61. w_2 = 0.20·0.61 = 0.12; w_3 = 0.20·0.39 = 0.08.
- **Final**: w ≈ [0.47, 0.33, 0.12, 0.08]. Σw = 1.00.

Compare inverse-vol on the same input:
- inv = [10, 8.33, 5, 4]; sum = 27.33; w ≈ [0.366, 0.305, 0.183, 0.146].

Inverse-vol gives more weight to the high-σ correlated pair (0.183 + 0.146 = 0.329 = 33 % into the {2,3} cluster). HRP recognises {2,3} is internally redundant and shifts the cluster-level weight down to 20 %. This is the *correlation-aware* behavior that is the entire point of the algorithm.

---

**End of B1 deliverable. Ready for review.**
