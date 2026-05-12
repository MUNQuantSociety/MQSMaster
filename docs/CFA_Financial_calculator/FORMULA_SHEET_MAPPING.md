# BUSI 3550 Formula Sheet -> Code Mapping (Strict 1:1)

This mapping uses the extracted sheet in `formula_sheet_extracted.txt` and maps each formula line to an explicit implementation function.

## Page 1: Time Value of Money, Annuities, Perpetuities

| Formula Sheet Entry | Code Mapping | Status |
|---|---|---|
| PV = FV / (1 + r)^t | `TimeValueOfMoney.present_value` (`src/tvm.py:17`) | Exact |
| FV = PV * (1 + r)^t | `TimeValueOfMoney.future_value` (`src/tvm.py:14`) | Exact |
| T = log(FV/PV) / log(1 + r) | `TimeValueOfMoney.solve_time_periods` (`src/tvm.py:49`) | Exact |
| r = (FV/PV)^(1/T) - 1 | `TimeValueOfMoney.solve_discount_rate` (`src/tvm.py:60`) | Exact |
| FV_T = C0 * (1 + r/m)^(mT) | `TimeValueOfMoney.future_value_compounding` (`src/tvm.py:69`) | Exact |
| Continuous compounding: FV_T = C0 * e^(rT) | `TimeValueOfMoney.future_value_continuous` (`src/tvm.py:84`) | Exact |
| EAR = (1 + r/m)^m - 1 | `TimeValueOfMoney.effective_annual_rate` (`src/tvm.py:93`) | Exact |
| PV of annuity | `TimeValueOfMoney.present_value_annuity_formula` (`src/tvm.py:102`) | Exact |
| PV of growing annuity | `TimeValueOfMoney.present_value_growing_annuity` (`src/tvm.py:113`) | Exact |
| FV of annuity | `TimeValueOfMoney.future_value_annuity` (`src/tvm.py:126`) | Exact |
| PV annuity due = PV annuity * (1 + r) | `TimeValueOfMoney.present_value_annuity_due` (`src/tvm.py:137`) | Exact |
| Perpetuity: PV = C / r | `TimeValueOfMoney.present_value_perpetuity` (`src/tvm.py:143`) | Exact |
| Growing perpetuity: PV = C_(t+1) / (r - g) | `TimeValueOfMoney.present_value_growing_perpetuity` (`src/tvm.py:149`) | Exact |

## Page 2: Bond Value and Stock Value

| Formula Sheet Entry | Code Mapping | Status |
|---|---|---|
| Bond value formula | `BondCalculator.bond_price` (`src/bonds.py:64`) | Exact |
| Holding period return | `BondCalculator.holding_period_return` (`src/bonds.py:118`) | Exact |
| Forward rate: f_n = ((1+r_n)^n / (1+r_(n-1))^(n-1)) - 1 | `BondCalculator.forward_rate` (`src/bonds.py:127`) and `YieldCurveCalculator.one_period_forward_rate` (`src/yield_curves.py:86`) | Exact |
| P0 = D_(t+1)/(r-g) | `EquityValuation.stock_value_constant_growth` (`src/equities.py:61`) | Exact |
| P0 = D1/r | `EquityValuation.stock_value_zero_growth` (`src/equities.py:69`) | Exact |
| Two-stage dividend model | `EquityValuation.stock_value_two_stage_growth` (`src/equities.py:77`) | Exact |
| g = RR * ROE | `EquityValuation.sustainable_growth_rate` (`src/equities.py:103`) | Exact |
| r = Div_(t+1)/P_t + g | `EquityValuation.required_return_from_dividend_growth` (`src/equities.py:109`) | Exact |
| P/share = EPS/r + NPVGO/share | `EquityValuation.price_per_share_eps_npvgo` (`src/equities.py:115`) | Exact |
| V0 from discounted FCF + terminal value | `EquityValuation.firm_value_dcf` (`src/equities.py:136`) | Exact |
| V_N = FCF_(N+1)/(r_WACC-g) = FCF_N(1+g)/(r_WACC-g) | `EquityValuation.terminal_value_perpetual_growth` (`src/equities.py:149`) | Exact |
| Equity value per share = (V0 + Cash0 - Debt0)/Shares0 | `EquityValuation.equity_value_per_share` (`src/equities.py:155`) | Exact |
| Enterprise value = Equity + Debt - Cash | `EquityValuation.enterprise_value` (`src/equities.py:161`) | Exact |

## Page 3: Project Evaluation

| Formula Sheet Entry | Code Mapping | Status |
|---|---|---|
| NPV = PV of all cash flows | `CashFlowCalculator.npv` (`src/cashflows.py:7`) | Exact |
| NPV component form (operating + ECF + tax shield - asset cost) | `CashFlowCalculator.project_npv_from_components` (`src/cashflows.py:229`) | Exact |
| PV(TS) annuity-tax-shield form | `CashFlowCalculator.present_value_tax_shield_annuity` (`src/cashflows.py:186`) | Exact |
| PV(CCATS) | `CashFlowCalculator.present_value_ccats` (`src/cashflows.py:204`) | Exact |
| NPV = sum(CF_t/(1+r)^t) - C0 | `CashFlowCalculator.npv_with_initial_outlay` (`src/cashflows.py:242`) | Exact |
| IRR root equation | `CashFlowCalculator.irr` (`src/cashflows.py:26`) | Exact |
| PI = PV(inflows)/C0 | `CashFlowCalculator.profitability_index` (`src/cashflows.py:151`) | Exact |
| EAB = NPV / A_r^T | `CashFlowCalculator.equivalent_annual_benefit` (`src/cashflows.py:173`) | Exact |

## Pages 3-4: Returns and Dispersion

| Formula Sheet Entry | Code Mapping | Status |
|---|---|---|
| Arithmetic average return | `Portfolio.arithmetic_average_return` (`src/portfolio.py:40`) | Exact |
| Geometric return | `Portfolio.geometric_average_return` (`src/portfolio.py:46`) | Exact |
| Expected return (probability-weighted) | `Portfolio.expected_return_probability` (`src/portfolio.py:59`) | Exact |
| Expected standard deviation | `Portfolio.expected_standard_deviation` (`src/portfolio.py:72`) | Exact |
| Historical standard deviation | `Portfolio.historical_standard_deviation` (`src/portfolio.py:80`) | Exact |

## Pages 4-5: Portfolio, Covariance, Beta, CAPM, WACC, Sharpe, CML

| Formula Sheet Entry | Code Mapping | Status |
|---|---|---|
| E(R_p) = w_A E(R_A) + w_B E(R_B) | `Portfolio.expected_portfolio_return_two_assets` (`src/portfolio.py:89`) | Exact |
| w_i = H_i / sum(H_k) | `Portfolio.portfolio_weight` (`src/portfolio.py:95`) | Exact |
| sigma_p with covariance term | `Portfolio.portfolio_standard_deviation_two_assets(..., covariance_ab=...)` (`src/portfolio.py:104`) | Exact |
| sigma_p with correlation term | `Portfolio.portfolio_standard_deviation_two_assets(..., correlation_ab=...)` (`src/portfolio.py:104`) | Exact |
| corr_ab = cov_ab / (sigma_a sigma_b) | `Portfolio.correlation_from_covariance` (`src/portfolio.py:142`) | Exact |
| cov_ab = sum p_i (r_ai-E(r_a))(r_bi-E(r_b)) | `Portfolio.covariance_two_assets` (`src/portfolio.py:128`) | Exact |
| beta_j = cov_jM / sigma_M^2 | `Portfolio.beta` (`src/portfolio.py:148`) | Exact |
| beta_p = sum w_i beta_i | `Portfolio.portfolio_beta` (`src/portfolio.py:154`) | Exact |
| CAPM: E(R_j) = R_f + beta_j(E(R_m)-R_f) | `Portfolio.capm_expected_return` (`src/portfolio.py:165`) and `EquityValuation.required_rate_of_return` (`src/equities.py:49`) | Exact |
| WACC (debt + preferred + common) | `Portfolio.wacc` (`src/portfolio.py:169`) | Exact |
| WACC (debt + equity) | `Portfolio.wacc` (`src/portfolio.py:169`) | Exact |
| Sharpe ratio | `Portfolio.sharpe_ratio` (`src/portfolio.py:197`) | Exact |
| CML equation | `Portfolio.cml_expected_return` (`src/portfolio.py:203`) | Exact |
| Asset beta: beta_A = (E/(D+E)) beta_E | `Portfolio.asset_beta_from_equity_beta` (`src/portfolio.py:209`) | Exact |
| Equity beta: beta_E = (1 + D/E) beta_A | `Portfolio.equity_beta_from_asset_beta` (`src/portfolio.py:216`) | Exact |

## Naming Convention Alignment Applied

1. Added formula-sheet literal function names where missing (for example `solve_time_periods`, `present_value_growing_annuity`, `profitability_index`, `portfolio_beta`, `cml_expected_return`).
2. Preserved existing public methods and made overlap formulas map directly to formula-sheet methods.
3. Added explicit forward-rate helpers in both fixed-income contexts (`bonds` and `yield_curves`) and FX forward pricing helper in `currencies`.
