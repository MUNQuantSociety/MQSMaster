# B3 — Portfolio_6 / P7 / P8 Volatility-Target Audit

**Date:** 2026-05-20
**Author:** Team B (vol-target audit, agent B3)
**Scope:** `src/portfolios/portfolio_6/`, inheritors P7/P8 (per Team A SYNTHESIS), `src/risk_manager/daily_allocator.py`, `src/risk_manager/manage_capital.py`, `src/portfolios/portfolio_manager_config.json`.
**Status:** Read-only. No source files modified by this agent. Recommended action below is **DOCUMENT** — no code patch required to fix double-targeting, but a small clarifying patch (comments + one config-level guard) is proposed.

---

## 1. Executive summary (≤200 words)

There is **no double vol-targeting** in the current MQSMaster system as wired today. Portfolio_6 applies a single sleeve-level scale `min(σ_target / σ_realized, MAX_LEVERAGE)` inside `screener.vol_target_scale` to the **stock sleeve only** (GLD and the optional trend-hedge ticker are added afterward at fixed config weights, never re-scaled by vol). The capital allocator (`src/risk_manager/daily_allocator.py`) is a **pure dollar-weight** rebalancer toward fixed `portfolio_weights` in `portfolio_manager_config.json` — it never measures or targets portfolio volatility. So the two layers do **not** compound multiplicatively.

However, two real risks exist:

1. **Latent risk (silent change of contract):** If anyone later teaches the allocator to vol-target while sleeves keep their own vol-targets, leverage stacks and realized vol collapses well below target (Harvey et al. 2018; ECB 2020 sell-off note).
2. **Already-present compositional fragility:** P6's leverage cap acts on `Σ w_i` *after* hedges are added at constant weight — a stock-sleeve scale-up of 1.5× plus 0.07 GLD plus 0.10 trend hedge can clip and silently change the realized stock/hedge mix.

Recommendation: **DOCUMENT** (comments + a structural invariant). Provide an apply-ready documentation patch. Reject moving vol-target up to the allocator at this stage.

---

## 2. Sources (≥10 primary; ≥2 independent per major claim)

All URLs fetched 2026-05-20. PDF binaries that did not decode through WebFetch are cited via their abstract / publisher pages plus a secondary source giving the same numeric/conceptual finding (cross-validation requirement).

| # | URL | Annotation | Used for |
|---|---|---|---|
| S1 | https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513 | Moreira & Muir (2017) *Volatility-Managed Portfolios*, JoF 72(4):1611-1644. Abstract + bibliographic record. | Canonical citation for `f_t = (c / σ̂²_t) f_{t+1}` per-factor scaling. |
| S2 | https://www.nber.org/papers/w22208 | Moreira & Muir (2016) NBER WP 22208 — same paper, working version. | Cross-validation of factor-by-factor (not nested) application. |
| S3 | https://amoreira2.github.io/alan-moreira.github.io/VolPortfolios_published.pdf | Moreira-Muir published PDF (author site). Binary in WebFetch but URL is canonical. | Source-of-record link. |
| S4 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3175538 | Harvey, Hoyle, Korgaonkar, Rattray, Sargaison, Van Hemert (2018) *The Impact of Volatility Targeting*, JPM 45(1):14-33. | Empirical evidence that vol-targeting works on equity/credit (risk-on) and that Sharpe is *negligibly* improved on bonds/FX/commodities. |
| S5 | https://jpm.pm-research.com/content/45/1/14.abstract | JPM publisher page for S4. | Cross-validation. |
| S6 | https://alphaarchitect.com/volatility-targeting-improves-risk-adjusted-returns/ | Alpha Architect summary of Harvey et al. 2018, with the asset-class breakdown and 10% target reproduced. | Cross-validation of S4 numerics where the SSRN/JPM full text was 403/Payment-Required. |
| S7 | https://quantpedia.com/the-impact-of-volatility-targeting-on-equities-bonds-commodities-and-currencies/ | QuantPedia summary of Harvey et al. — confirms "vol-target works for equities and credit, negligible for bonds/FX/commodities; asset-level and portfolio-level *both* improve Sharpe, no double-counting claim." | Cross-validation (third independent summary). |
| S8 | https://www.man.com/insights/the-impact-of-volatility-targeting | Man Group / AHL author summary of S4. Specifies *both* asset-level and portfolio-level vol scaling are studied, individually. Explicitly does **not** discuss layered/double targeting. | Confirms literature gap on nested vol-targeting. |
| S9 | https://jpm.pm-research.com/content/39/2/28 | Hocquard, Ng, Papageorgiou (2013) *A Constant-Volatility Framework for Managing Tail Risk*, JPM 39(2):28-40. | Single-portfolio constant-vol overlay (futures-based) is preferred over per-asset; cited for the "do it at one layer" principle. |
| S10 | https://www.hillsdaleinv.com/news-events/aima-canada-hillsdale-research-award | AIMA-Canada / Hillsdale 2010 award page for Hocquard-Ng-Papageorgiou. | Cross-validation of S9 provenance and design (use futures to overlay vol target on top of strategic allocation — one overlay, not two). |
| S11 | https://www.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/Leverage-Aversion-and-Risk-Parity.pdf | Asness, Frazzini, Pedersen (2012) *Leverage Aversion and Risk Parity*, FAJ 68(1):47-59. AQR-hosted PDF. | The textbook Risk-Parity pattern: weight by `1/σ_i` at asset level, then multiply by **one** scalar `k` to hit total-portfolio vol target. One layer, not two. |
| S12 | https://w4.stern.nyu.edu/facdir/lpederse/papers/LeverageAversionRP.pdf | Stern (Pedersen) mirror of S11. | Cross-validation of S11. |
| S13 | https://www.researchaffiliates.com/publications/articles/1014-harnessing-volatility-targeting | Research Affiliates *Harnessing Volatility Targeting in Multi-Asset Portfolios*. | Practitioner reference that vol-targeting is normally implemented at one layer (overlay), with leverage caps; useful for the "modest leverage cap (≤ 2×)" norm. |
| S14 | https://www.ecb.europa.eu/press/financial-stability-publications/fsr/focus/2020/html/ecb.fsrbox202005_02~f6616db9be.en.html | ECB FSR May 2020, box on vol-targeting strategies and March-2020 sell-off. AUM ≈ USD 2T, ≈ USD 300B in risk-parity; deleveraging mechanics. | Systemic risk of synchronized deleveraging; relevant to the "what if we add an allocator-level vol target on top" decision. |
| S15 | https://arxiv.org/abs/2212.07288 | Castro & Salas-Molina (2022) *Smoothing Volatility Targeting* (arXiv 2212.07288). | Turnover and leverage spikes from raw σ̂ scaling; argues for **one** smoothed layer rather than nested. |
| S16 | https://www.research.hangukquant.com/p/volatility-targeting-the-asset-level | Hangukquant *Volatility Targeting — The Asset Level*. | Identifies the three hierarchical levels (asset / strategy / portfolio) and notes "no tenable distinction" if levels are simultaneously applied without explicit decomposition — supports the "pick one layer and document" recommendation. |
| S17 | https://stoffelwealth.com/volatility-targeting-a-guide-to-stabilizing-portfolio-risk/ | Practitioner guide; explicit `scale = target_vol / recent_vol` formula identical to `screener.vol_target_scale`. Cross-validates that P6's math is the standard textbook form. | Confirms P6 sleeve formula matches the literature. |
| S18 | https://blog.thinknewfound.com/2023/04/portfolio-tilts-versus-overlays-its-long-short-portfolios-all-the-way-down/ | Newfound *Portfolio Tilts versus Overlays*. | Useful framing for "is the GLD/trend sleeve a tilt or an overlay?" — supports the §4 hedge-sleeve ordering analysis. |

**Cross-validation matrix (≥2 sources per load-bearing claim):**

| Claim | Supporting sources |
|---|---|
| Standard vol-target formula `min(σ_target/σ_realized, cap)` | S1+S15+S17 |
| Vol-targeting only meaningfully improves Sharpe on equity/credit (risk-on) | S4+S6+S7+S8 |
| Risk-Parity scales **once** at the master level after inverse-vol at asset level | S11+S12+S13 |
| Synchronized deleveraging risk when many funds simultaneously vol-target | S14+S15 |
| Practitioner leverage cap norm is 1.5×–2× | S13+S17 (P6 uses 1.5×) |

---

## 3. Current-state analysis — line-by-line trace

### 3.1 Sleeve-level scale inside Portfolio_6

File: `src/portfolios/portfolio_6/screener.py`, lines 115-129

```python
def vol_target_scale(
    portfolio_returns: pd.Series,
    *,
    target_annual_vol: float = 0.13,
    max_scale: float = 1.5,
) -> float:
    if portfolio_returns is None or portfolio_returns.empty:
        return 1.0
    sd = float(portfolio_returns.std())                    # daily σ of weighted sleeve
    if not np.isfinite(sd) or sd <= 0:
        return 1.0
    realized_ann = sd * (TRADING_DAYS ** 0.5)              # annualized
    if realized_ann <= 0:
        return 1.0
    return float(min(target_annual_vol / realized_ann, max_scale))
```

Standard formula `k = min(σ_target / σ_realized, cap)`. Identical to S1, S15, S17.

### 3.2 Where it is applied — `Portfolio6Strategy._rebalance`

File: `src/portfolios/portfolio_6/strategy.py`

| Line(s) | Step | What is scaled |
|---|---|---|
| 196-202 | Collect daily-return matrix for screened universe. | — |
| 204-212 | `score_universe` → `select_top_n` (N=50). | — |
| 214-220 | `inverse_vol_weights(...)` ⇒ `weights` (raw, sums to 1.0, capped per-name at 0.05). | — |
| 222-224 | Build `sleeve_returns` = (returns_df · weight_series).sum(axis=1) — the **unlevered stock sleeve daily return series**. | — |
| 243-247 | `vol_scale = vol_target_scale(sleeve_returns, target_annual_vol=0.13, max_scale=1.5)`. | computes `k` |
| 248 | `weights = {t: w * vol_scale for t, w in weights.items()}` ⇒ stock-sleeve weights now sum to `k` (∈ [0, 1.5]). | **stock sleeve only** |
| 250-259 | `target_weights[GLD] = self.gld_weight` (0.07, **fixed config constant**, not vol-scaled). | hedge sleeve added vol-naïve |
| 261-274 | `target_weights[TREND] = self.trend_weight` (0.10 in config, **fixed**, not vol-scaled; currently disabled because `TREND_HEDGE_TICKER=""`). | hedge sleeve added vol-naïve |
| 276-284 | `total = Σ target_weights`; if `total > MAX_LEVERAGE (1.5)`, divide everything by `total / MAX_LEVERAGE` — i.e. **a second renormalization** that touches both stock and hedge weights. | proportional rescale (not vol-driven) |
| 286 | Persisted as `self._target_weights`. | — |

**Important nuance at line 276-284.** This second rescale is **not** a vol-target, it is a *gross-notional cap*. It only fires when stock-sleeve `k` is at or near `MAX_LEVERAGE=1.5` and adding GLD (0.07) + TREND (0.10) pushes the sum above 1.5. In that branch, the stock sleeve is forced *back down* below `k`, so the realized stock-sleeve vol falls *below* `VOL_TARGET_ANNUAL`. This is a real but distinct issue (see §4 "Caveat — hedge sleeve eats stock-sleeve vol budget").

### 3.3 Allocator-level scaling — `DailyAllocator.run_allocation`

File: `src/risk_manager/daily_allocator.py`, lines 157-210

```python
target_value = total_equity * weight                                # weight from portfolio_manager_config.json
adjustment = target_value - current_total_value
... _execute_internal_transfer(...)  # cash move from master ↔ sub-portfolio
```

The allocator:
- Reads `portfolio_weights` from `portfolio_manager_config.json` (currently `{"1": 0.10, "2": 0.90}`).
- Computes total equity = master cash + Σ (sub-cash + Σ mark-to-market positions).
- For each sub-portfolio, moves cash to/from the master so that the sub-portfolio's *total value* matches its target *capital* share.
- **No volatility measurement.** No `σ_realized`. No `σ_target`. No leverage scaling. Pure dollar reallocation.

`src/risk_manager/manage_capital.py` is the operator's ADD/WITHDRAW capital tool — even more inert from a vol perspective; it just writes a row into `cash_equity_book`.

### 3.4 Manager config

```
{
  "master_portfolio_id": "0",
  "currency": "USD",
  "portfolio_weights": { "1": 0.10, "2": 0.90 }
}
```

No vol-target field at the allocator layer. P6 is not even *registered* in this file (the Team A SYNTHESIS §9 diff would add P7=0.0, P8=0.0 if applied).

### 3.5 Inheritors (per Team A SYNTHESIS §2)

| Portfolio | Vol-target inheritance |
|---|---|
| **P7** (P6 + sentiment tilt) | Subclasses `Portfolio6Strategy`. Tilt is applied to per-name weights `w_i ← w_i · exp(λ z_i)`. SYNTHESIS §2 places tilt *before* `vol_target_scale` (tilt → renormalize → vol-scale). One layer; no double counting. |
| **P8** (P6 + RBP rank overlay) | Subclasses `Portfolio6Strategy`. RBP enters as an extra leg inside `score_universe` (composite score), so it affects *selection* but never the vol scale itself. One layer; no double counting. |

---

## 4. Double-targeting diagnosis

**Verdict: NO double vol-targeting in current code. Yes, latent risk if the allocator is later upgraded.**

### 4.1 Why "no" today

A "double vol-target" requires two multiplicative scalars `k_sleeve` and `k_master` both of the form `σ_target / σ_realized` such that the final position equals `k_master · k_sleeve · w_raw`. In the current code:

- P6 sleeve applies `k_sleeve = min(σ_target/σ_real, 1.5)` (lines 243-248 of `strategy.py`). ✅ exists.
- Allocator applies a **capital weight**, not a vol scale. The 0.10 and 0.90 in `portfolio_manager_config.json` are dollar-share constants. They do not depend on `σ`. ❌ no second vol-scale.

Therefore the realized sub-portfolio vol is `k_sleeve · σ_real` ≈ `σ_target = 13%`. The master-portfolio realized vol is approximately `Σ w_i² σ_i² + 2 Σ_{i<j} w_i w_j σ_i σ_j ρ_ij` where `σ_i` ≈ 13% for each sub that vol-targets. This is the *expected* behavior; no compounding.

This matches the canonical Risk Parity construction (Asness-Frazzini-Pedersen 2012, S11/S12): "weight by 1/σ at the asset level, then multiply by **one** constant `k` to match a target volatility." Our analog: weight by 1/σ at the *name* level inside P6, then multiply by **one** constant `k_sleeve` inside P6, then allocate **dollar shares** at the master. One vol layer, one dollar layer. Clean.

### 4.2 Conditional double-targeting — three trigger paths to monitor

These do *not* exist today but would create double-counting if implemented:

- **T1 — Allocator vol-overlay.** Someone teaches `DailyAllocator` to read recent master P&L, compute realized vol, and rescale `portfolio_weights` to hit a master target σ. Then `final_position = k_master · k_sleeve · w_raw`. Realized vol collapses below target because *both* scalars shrink when σ rises.
- **T2 — Sub-portfolio with vol-target ≠ master target.** If P7 (or P8) sets `VOL_TARGET_ANNUAL` materially different from P6 (e.g. 0.20 for P7), the master inherits a mix of σ-targets. Not double-counting per se, but harder to reason about and undocumented today. SYNTHESIS §10 already flags this in row #B3.
- **T3 — Hedge sleeve also vol-targeted.** If `TREND_HEDGE_TICKER` is filled with an *already* vol-targeted ETF (e.g. KMLM, DBMF, CTA, which internally manage to ~10% vol), and we *additionally* scale its weight by P6's vol scaler in the future, the trend sleeve gets vol-scaled twice. Today the trend sleeve is added at a **fixed** weight (line 263-264), so this is safe — but only because the fixed-weight branch happens to ignore `k_sleeve`.

### 4.3 Caveat — hedge sleeve eats the stock-sleeve vol budget

This is **not** double vol-targeting but it is a related correctness issue worth flagging.

Lines 250-284 of `strategy.py` do:
```
total = (k_sleeve · 1.0) + 0.07 (GLD) + 0.10 (TREND if enabled)
if total > 1.5:
    rescale everything by (1.5 / total)
```

Worked example with `TREND_HEDGE_TICKER="VMOT"` (or any non-empty value):
- `k_sleeve` = 1.5 (cap binding because σ_real < σ_target / 1.5 ≈ 8.67%)
- `total` = 1.5 + 0.07 + 0.10 = 1.67
- `factor` = 1.5 / 1.67 ≈ 0.898
- Stock sleeve actually deployed = 1.5 · 0.898 ≈ **1.347** (not 1.5)
- Realized stock-sleeve vol = 1.347 · σ_real, which is **10.4%** when σ_real is at the saturation point of 7.78%, *below* the 13% target by ≈ 260 bps.

So the cap silently *under-targets* vol when both hedges and full leverage are active. Today `TREND_HEDGE_TICKER=""` so only GLD's 0.07 enters and the cap fires only when `k_sleeve > 1.43` — still possible. This is the only real defect found.

---

## 5. Literature recommendation

**Synthesis of S1, S4, S9, S11 and S13:**

The literature consistently uses **one** vol-targeting layer, not two:

- **Moreira & Muir 2017 (S1, S2):** Each factor (mkt, val, mom, prof, …) is scaled **individually** by `c/σ̂²` — but each factor portfolio is the *terminal* portfolio in their study. There is no allocator on top.
- **Harvey et al. 2018 (S4, S6, S7):** Studies vol-scaling at the **asset** level and at the **portfolio** level *as alternatives*, never as compositions. Both are shown to improve Sharpe on equity/credit; for bonds/FX/commodities the Sharpe improvement is "negligible". This is the strongest support for the "one layer only" practice. They use a single 10% target and do not cascade.
- **Hocquard-Ng-Papageorgiou 2013 (S9, S10):** Constant-vol overlay applied as a **single** futures overlay on top of the strategic asset allocation. Explicitly *one* layer, *one* target.
- **Asness-Frazzini-Pedersen 2012 (S11, S12):** Risk Parity weights = `1/σ_i` at asset level, then multiplied by **one** scalar `k` to hit total-portfolio vol. Inverse-vol weighting and vol-targeting are **two complementary** steps, not two vol targets.
- **Research Affiliates / practitioner consensus (S13, S17):** "Modest leverage" caps (1×–2×); P6's 1.5× is in band.

**Rule the literature is silent on:** nested vol-targeting (sleeve *and* master both running `σ_target/σ_realized`). The closest discussion (S16 Hangukquant) explicitly says: "there is no tenable distinction between the different levels" if you simultaneously apply them — i.e. it is structurally ambiguous and should be avoided unless an explicit decomposition is documented.

**Applied to MQSMaster:**

Each of MQSMaster's portfolios behaves like a Risk-Parity sub-fund: weight by `1/σ` at the name level, scale by **one** `k` at the sleeve level to hit a 13% sleeve-vol target. The master allocator is then a dollar-allocator (analogous to "fund of funds" cash allocation), which is exactly the right separation per S11.

⇒ **Single-sleeve vol-targeting alone is the correct regime for the current architecture. Do not add an allocator-level vol-target.**

---

## 6. Recommended action — **DOCUMENT**

No code patch is required to *fix* a double-targeting bug, because no such bug exists today.

The recommended actions are:

1. **D1 — Documentation patch:** add inline comments in `screener.vol_target_scale`, `Portfolio6Strategy._rebalance`, and `DailyAllocator.run_allocation` declaring the contract: "vol-target lives at sleeve level only; allocator scales by capital weight only". (Apply-ready diff in §7.)
2. **D2 — Config invariant:** add a non-enforced top-level note in `portfolio_manager_config.json` (via a sibling `_README` key — JSON-safe, ignored by code, visible to future maintainers). (Apply-ready diff in §7.)
3. **D3 — Falsification test in §8** to be run in `main_backtest.py` if anyone proposes adding allocator-level vol scaling.
4. **D4 — Latent bug (hedge sleeve eats vol budget under cap-binding).** Out of scope for this audit, but flagged for a follow-up. **Do not bundle** into this DOCUMENT patch — that would silently change P6 vol behavior.

Reject: (b) moving vol-target up to the allocator (no infrastructure for daily master-P&L vol estimation, and breaks the Risk-Parity-style separation that the literature endorses).

Reject: (c) explicit decomposition / "residual vol budget" — premature; would require a meaningful master-level vol model that does not exist.

---

## 7. Unified diffs — apply-ready

All three are pure-comment / pure-doc diffs. None changes program behavior.

### 7.1 `src/portfolios/portfolio_6/screener.py`

```diff
--- a/src/portfolios/portfolio_6/screener.py
+++ b/src/portfolios/portfolio_6/screener.py
@@ -113,12 +113,22 @@ def inverse_vol_weights(
     return w.to_dict()
 
 
 def vol_target_scale(
     portfolio_returns: pd.Series,
     *,
     target_annual_vol: float = 0.13,
     max_scale: float = 1.5,
 ) -> float:
+    """
+    Single-layer sleeve volatility scaler. Returns
+        k = min(target_annual_vol / sigma_realized_ann, max_scale).
+
+    Contract (B3 audit, 2026-05-20):
+      - This is the ONLY vol-target layer in MQSMaster. The capital
+        allocator (src/risk_manager/daily_allocator.py) does NOT vol-target;
+        it allocates by fixed dollar weight from portfolio_manager_config.json.
+      - Adding a second vol-target layer (e.g. inside DailyAllocator) would
+        compound multiplicatively and silently collapse realized vol below
+        target. See .claude/agents-output/teamB/B3_vol_target_audit.md §4.2.
+      - References: Moreira-Muir 2017 (JoF 72:1611), Harvey et al. 2018
+        (JPM 45:14), Asness-Frazzini-Pedersen 2012 (FAJ 68:47).
+    """
     if portfolio_returns is None or portfolio_returns.empty:
         return 1.0
     sd = float(portfolio_returns.std())
```

### 7.2 `src/portfolios/portfolio_6/strategy.py`

```diff
--- a/src/portfolios/portfolio_6/strategy.py
+++ b/src/portfolios/portfolio_6/strategy.py
@@ -240,8 +240,16 @@ class Portfolio6Strategy(BasePortfolio):
         except Exception as e:
             self.logger.exception("[P6] DSR computation failed: %s", e)
 
+        # ---- Sleeve-level vol-target (single authoritative layer) ----
+        # vol_scale = min(VOL_TARGET_ANNUAL / sigma_realized, MAX_LEVERAGE).
+        # Applied only to the stock sleeve. GLD and the optional trend hedge
+        # are added below at FIXED config weights (vol-naive). Any future
+        # change that scales the hedge sleeves by vol_scale, OR adds a
+        # second vol_target layer in the allocator, will compound. See
+        # .claude/agents-output/teamB/B3_vol_target_audit.md §4.
         vol_scale = vol_target_scale(
             sleeve_returns,
             target_annual_vol=self.vol_target_annual,
             max_scale=self.max_leverage,
         )
         weights = {t: w * vol_scale for t, w in weights.items()}
@@ -275,6 +283,12 @@ class Portfolio6Strategy(BasePortfolio):
         total = sum(target_weights.values())
+        # Gross-notional cap (NOT a second vol-target). When stock sleeve is
+        # already at MAX_LEVERAGE and hedge sleeves push total > MAX_LEVERAGE,
+        # this proportional rescale forces the stock sleeve back down, which
+        # under-targets vol by the cap-binding factor. Tolerated; flagged in
+        # B3 §4.3 for a future fix (subtract hedge weights from leverage
+        # budget before computing vol_scale).
         if total > self.max_leverage > 0:
             scale = self.max_leverage / total
             target_weights = {t: w * scale for t, w in target_weights.items()}
```

### 7.3 `src/risk_manager/daily_allocator.py`

```diff
--- a/src/risk_manager/daily_allocator.py
+++ b/src/risk_manager/daily_allocator.py
@@ -155,6 +155,17 @@ class DailyAllocator:
             if conn: self.db_connector.release_connection(conn)
 
     def run_allocation(self):
+        """
+        Daily capital allocator.
+
+        Contract (B3 audit, 2026-05-20): this method scales sub-portfolios
+        ONLY by their fixed dollar weight in portfolio_manager_config.json.
+        It MUST NOT compute realized portfolio volatility and MUST NOT scale
+        weights by sigma_target / sigma_realized. Sleeve-level vol-targeting
+        already lives in src/portfolios/portfolio_6/screener.py
+        (vol_target_scale). Adding a second vol layer here would compound
+        multiplicatively. See .claude/agents-output/teamB/B3_vol_target_audit.md.
+        """
         # ... (no changes in this function)
         logger.info("Starting daily capital allocation...")
```

### 7.4 `src/portfolios/portfolio_manager_config.json`

JSON does not support comments, so encode the invariant as a sibling key. Code reads `master_portfolio_id`, `currency`, `portfolio_weights` only; an extra key is ignored.

```diff
--- a/src/portfolios/portfolio_manager_config.json
+++ b/src/portfolios/portfolio_manager_config.json
@@ -1,8 +1,16 @@
 {
+  "_invariants": [
+    "B3 audit (2026-05-20): portfolio_weights are CAPITAL shares, not vol shares.",
+    "DailyAllocator must not measure realized master-portfolio volatility.",
+    "Sleeve-level vol-targeting (sigma_target / sigma_realized) lives inside",
+    "src/portfolios/portfolio_6/screener.py::vol_target_scale only.",
+    "Adding a second vol-target layer here (or in DailyAllocator) compounds",
+    "and silently collapses realized vol. See .claude/agents-output/teamB/B3_vol_target_audit.md."
+  ],
   "master_portfolio_id": "0",
   "currency": "USD",
   "portfolio_weights": {
     "1": 0.10,
     "2": 0.90
   }
 }
```

These four diffs total ~35 lines of new content, all comments / docstrings / a JSON sibling key. Zero behavioral change.

---

## 8. Falsification test — concrete numeric design

Goal: prove (or refute) that **a hypothetical double-layer config** would collapse realized vol, while the current single-layer config matches target ± 100 bps.

### 8.1 Test harness

In `src/main_backtest.py` (or a parallel scratch script that imports `Portfolio6Strategy` directly), add a feature flag `ALLOCATOR_VOL_TARGET`:

```python
# Hypothetical second layer (DO NOT MERGE INTO PRODUCTION).
def _master_vol_scale(history_returns_df, target=0.13, cap=1.5):
    sd = history_returns_df.sum(axis=1).std()
    realized_ann = sd * (252 ** 0.5)
    return min(target / realized_ann, cap) if realized_ann > 0 else 1.0
```

Wrap `DailyAllocator.run_allocation` so each sub's `target_value` is multiplied by `_master_vol_scale(...)` when the flag is on. Run **three** 10-year backtests on identical price data (2016-01-01 → 2025-12-31), Portfolio_6 only, master allocator weight = 1.0:

| Run | Sleeve vol-target | Allocator vol-target | Predicted realized annual vol |
|---|---|---|---|
| A | OFF (`vol_scale` forced to 1.0) | OFF | ≈ σ_unlevered (typically 18–22% for a low-vol S&P 500 sleeve) |
| B | ON (status quo) | OFF | ≈ 13% ± 100 bps |
| C | ON | ON (`target=0.13`, `cap=1.5`) | ≈ ≤ 7% (i.e. ≤ 50% of the 13% target) |

### 8.2 Pass / fail criteria

- **Pass (audit verdict confirmed):**
  - Run B realized vol ∈ [12.0%, 14.0%] (single-layer is correct).
  - Run C realized vol < 0.5 × 13% = 6.5% (double-layer collapses).
  - Sharpe(B) ≥ Sharpe(C) by ≥ 0.15 (the second layer over-deleverages and forgoes returns without proportional risk reduction).
- **Fail (audit verdict refuted):** if Run C realized vol is within 100 bps of 13%, the double layer is innocuous and §6 should reconsider option (c).

### 8.3 Why "≤ 50% of target" is the right threshold

When both layers scale by `σ_target/σ_realized` and the two σ's are positively correlated (a realistic assumption — sleeve σ and master σ co-move because the master is mostly the sleeve), the *product* of two scalars `k_1 k_2` shrinks as `(σ_t/σ_r)²` in the unbounded case. With both caps at 1.5×, the realistic range is roughly `[0.4×, 1.0×] · [0.4×, 1.0×]` = `[0.16×, 1.0×]`, with median around `0.5×`. So realized vol clusters around `0.5 · σ_unlevered`, which is well below the 13% target when σ_unlevered ≈ 18–22%.

---

## 9. Risks + rollback path

### 9.1 Risks of the recommended **DOCUMENT** action

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Comments drift out of date if anyone later changes the sleeve scaler. | Low | Comments name `.claude/agents-output/teamB/B3_vol_target_audit.md` as the canonical reference; CI grep on `vol_target_scale` could enforce a no-second-layer rule (out of scope here). |
| R2 | The `_invariants` JSON key is non-enforced — a future maintainer could ignore it. | Medium | Pairing with the docstrings in `daily_allocator.py` and `screener.py` makes the contract visible at three independent sites. |
| R3 | The §4.3 hedge-sleeve-eats-vol-budget defect is documented but **not fixed** here. | Medium | Out of scope for this audit (would change P6 behavior). Filed as a separate follow-up. |

### 9.2 Risks of *not* documenting

| # | Risk | Severity | Likelihood |
|---|---|---|---|
| N1 | Future "improvement" to `DailyAllocator` adds a vol-target overlay (T1 in §4.2). Realized vol collapses, Sharpe drops, no error raised. | High | Medium — vol-targeting is a common feature request from quant-finance reviewers. |
| N2 | A new portfolio inherits P6 and overrides `VOL_TARGET_ANNUAL`, creating an inconsistent master vol-profile (T2). | Medium | Medium-High — SYNTHESIS §2 already proposes two inheritors. |
| N3 | `TREND_HEDGE_TICKER` is set to a vol-managed CTA (KMLM/DBMF) without re-evaluating §4.2-T3. | Medium | High — Team C1 is explicitly asked (per SYNTHESIS §12) to pick a trend ticker. |

### 9.3 Rollback path

Each of the four diffs in §7 is pure-comment / pure-data. To rollback:
```
git revert <commit-sha>
```
will undo all comment and JSON sibling additions with zero behavioral change either way.

If a later change accidentally introduces an allocator-level vol-target, the rollback path is:
1. Run the §8 falsification harness (≤ 1 day to produce three vol curves).
2. If Run C confirms the collapse, set the `ALLOCATOR_VOL_TARGET` flag (or its eventual config equivalent) to `false` and redeploy. Single env-var rollback.

---

End of B3 audit.
