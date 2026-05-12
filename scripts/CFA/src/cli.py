from math import exp, sqrt

from .bonds import BondCalculator as bc
from .cashflows import CashFlowCalculator as cfc
from .currencies import CurrencyConverter
from .derivatives import DerivativesCalculator
from .equities import EquityValuation
from .portfolio import Portfolio
from .tvm import TimeValueOfMoney as tvm


class CLI:
    def __init__(self):
        self.options = {
            "1": "Time Value of Money",
            "2": "Cash Flow Analysis",
            "3": "Calculate Bond Valuation",
            "4": "Calculate Equity Valuation",
            "5": "Calculate Portfolio Metrics",
            "6": "Calculate Derivative Pricing",
            "7": "Currency Conversion",
            "8": "Yield Curve Analysis",
            "": "Exit\n",
        }

    def _display_menu(self):
        print("Welcome to the CFA Level 1 Financial Calculator!\n")
        for key, value in self.options.items():
            print(f"{key}: {value}")

    def run(self):
        while True:
            self._display_menu()
            choice = input("Select an option: ")
            if choice == "":
                print("Exiting the calculator. Goodbye!")
                break
            self._handle_choice(choice)

    def _handle_choice(self, choice):
        tvm_sub_menu = {
            "0": "Time Value of Money",
            "1": "FV = PV * (1 + r)^t",
            "2": "PV = FV / (1 + r)^t",
            "3": "Solve Time Periods: T = log(FV/PV)/log(1+r)",
            "4": "Solve Discount Rate: r = (FV/PV)^(1/T) - 1",
            "5": "Compounding FV: C0 * (1 + r/m)^(mT)",
            "6": "Continuous Compounding FV: C0 * e^(rT)",
            "7": "Effective Annual Rate (EAR)",
            "8": "PV of Annuity",
            "9": "PV of Growing Annuity",
            "10": "FV of Annuity",
            "11": "PV of Annuity Due",
            "12": "PV of Perpetuity",
            "13": "PV of Growing Perpetuity",
            "14": "Loan Annuity Payment",
            "": "Exit",
        }
        cashflow_sub_menu = {
            "0": "Cash Flow Analysis",
            "1": "NPV = PV of all cash flows",
            "2": "NPV = Sum(CF_t/(1+r)^t) - C0",
            "3": "Internal Rate of Return (IRR)",
            "4": "Payback Period",
            "5": "Discounted Payback Period",
            "6": "Profitability Index (PI)",
            "7": "Equivalent Annual Benefit (EAB)",
            "8": "PV(Tax Shield) Annuity",
            "9": "PV(CCATS)",
            "10": "Project NPV from Components",
            "11": "Annuity Factor (A_r^N)",
            "": "Exit",
        }
        bond_valuation_sub_menu = {
            "0": "Bond Valuation",
            "1": "Bond Price",
            "2": "Yield to Maturity (YTM)",
            "3": "Current Yield",
            "4": "Holding Period Return",
            "5": "Forward Rate: f_n",
            "": "Exit",
        }
        equity_valuation_sub_menu = {
            "0": "Equity Valuation",
            "1": "P0 = D1/(r-g) (Constant Growth)",
            "2": "P0 = D1/r (Zero Growth)",
            "3": "Two-Stage Dividend Model",
            "4": "g = RR * ROE",
            "5": "r = Div_{t+1}/P_t + g",
            "6": "P/share = EPS/r + NPVGO/share",
            "7": "Firm Value V0 from DCF",
            "8": "Terminal Value V_N",
            "9": "Equity Value Per Share",
            "10": "Enterprise Value",
            "11": "FCFE Value Per Share",
            "12": "Price/Earnings Ratio",
            "13": "Earnings Growth Rate",
            "14": "CAPM Required Return",
            "": "Exit",
        }
        portfolio_metrics_sub_menu = {
            "0": "Portfolio Metrics",
            "1": "Portfolio Expected Return (Value-Weighted)",
            "2": "Portfolio Risk (Variance Approximation)",
            "3": "Arithmetic Average Return",
            "4": "Geometric Average Return",
            "5": "Expected Return (Probability-Weighted)",
            "6": "Expected Standard Deviation",
            "7": "Historical Standard Deviation",
            "8": "Expected Portfolio Return (Two Assets)",
            "9": "Portfolio Weight: w_i = H_i / Sum(H_k)",
            "10": "Portfolio Std Dev (Covariance Form)",
            "11": "Portfolio Std Dev (Correlation Form)",
            "12": "Covariance (Two Assets)",
            "13": "Correlation from Covariance",
            "14": "Beta: beta_j = Cov_jM / sigma_M^2",
            "15": "Portfolio Beta",
            "16": "CAPM Expected Return",
            "17": "WACC",
            "18": "Sharpe Ratio",
            "19": "CML Expected Return",
            "20": "Asset Beta from Equity Beta",
            "21": "Equity Beta from Asset Beta",
            "": "Exit",
        }
        derivative_pricing_sub_menu = {
            "0": "Derivative Pricing",
            "1": "Calculate Option Price using Black-Scholes",
            "2": "Calculate Option Price using Binomial Tree",
            "3": "Calculate Futures Price",
            "": "Exit",
        }
        currency_conversion_sub_menu = {
            "0": "Currency Conversion",
            "1": "Convert Currency using Exchange Rate",
            "2": "Calculate Forward Exchange Rate",
            "": "Exit",
        }
        yield_curve_analysis_sub_menu = {
            "0": "Yield Curve Analysis",
            "1": "Calculate Spot Rates",
            "2": "Calculate Forward Rates from Spot Curve",
            "3": "One-Period Forward Rate",
            "4": "Calculate Yield to Maturity",
            "5": "Calculate Duration",
            "6": "Calculate Convexity",
            "": "Exit",
        }
        try:
            sub_menu_mapping = {
                "1": tvm_sub_menu,
                "2": cashflow_sub_menu,
                "3": bond_valuation_sub_menu,
                "4": equity_valuation_sub_menu,
                "5": portfolio_metrics_sub_menu,
                "6": derivative_pricing_sub_menu,
                "7": currency_conversion_sub_menu,
                "8": yield_curve_analysis_sub_menu,
            }
            if choice in sub_menu_mapping:
                self.sub_menu(sub_menu_mapping[choice])
            else:
                print("Invalid option. Please try again.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def _run_sub_menu(self, choice, menu):
        if menu == "Time Value of Money":
            self.tvm_sub_menu(choice)
        elif menu == "Cash Flow Analysis":
            self.cashflow_sub_menu(choice)
        elif menu == "Bond Valuation":
            self.bond_valuation_sub_menu(choice)
        elif menu == "Equity Valuation":
            self.equity_valuation_sub_menu(choice)
        elif menu == "Portfolio Metrics":
            self.portfolio_metrics_sub_menu(choice)
        elif menu == "Derivative Pricing":
            self.derivative_pricing_sub_menu(choice)
        elif menu == "Currency Conversion":
            self.currency_conversion_sub_menu(choice)
        elif menu == "Yield Curve Analysis":
            self.yield_curve_analysis_sub_menu(choice)
        else:
            print("Invalid sub-menu option.")

    def sub_menu(self, menu):
        while True:
            print(f"\n--- {menu.get('0', 'Sub-Menu')} ---\n")
            for key, value in menu.items():
                if key == "0":
                    continue
                print(f"{key}: {value}")
            op = input(
                "\nSelect an option (Press Enter to continue(Leave blank to Exit)): "
            )

            if op not in menu:
                print("\nInvalid option. Please try again.")
                continue

            print("\n")
            print("---" * 5)
            print(f"You selected: {menu[op]}\n")
            if menu[op] == "Exit":
                print("\nExiting Sub-Menu.")
                break

            try:
                self._run_sub_menu(op, menu.get("0", "Sub-Menu"))
                print("---" * 5)
            except Exception as e:
                print(f"Error: {e}")

    @staticmethod
    def _parse_float_list(raw_values):
        values = [value.strip() for value in raw_values.split(",") if value.strip()]
        if not values:
            raise ValueError("Please enter at least one numeric value.")
        return [float(value) for value in values]

    def tvm_sub_menu(self, choice):
        while True:
            try:
                if choice == "1":
                    principal = float(input("Enter present value (PV): "))
                    rate = float(input("Enter discount rate r (as a decimal): "))
                    time = float(input("Enter time periods T: "))
                    calc = tvm(principal=principal, rate=rate, time=time)
                    result = calc.future_value()
                    print(f"Future Value (FV): {round(result, 6)}")
                    break
                elif choice == "2":
                    future_value = float(input("Enter future value (FV): "))
                    rate = float(input("Enter discount rate r (as a decimal): "))
                    time = float(input("Enter time periods T: "))
                    calc = tvm(principal=future_value, rate=rate, time=time)
                    result = calc.present_value()
                    print(f"Present Value (PV): {round(result, 6)}")
                    break
                elif choice == "3":
                    present_value = float(input("Enter present value (PV): "))
                    future_value = float(input("Enter future value (FV): "))
                    rate = float(input("Enter discount rate r (as a decimal): "))
                    result = tvm.solve_time_periods(present_value, future_value, rate)
                    print(f"Time Periods (T): {round(result, 6)}")
                    break
                elif choice == "4":
                    present_value = float(input("Enter present value (PV): "))
                    future_value = float(input("Enter future value (FV): "))
                    time_periods = float(input("Enter time periods T: "))
                    result = tvm.solve_discount_rate(
                        present_value, future_value, time_periods
                    )
                    print(f"Discount Rate (r): {round(result, 6)}")
                    break
                elif choice == "5":
                    initial_investment = float(input("Enter initial investment C0: "))
                    rate = float(input("Enter nominal rate r (as a decimal): "))
                    periods_per_year = int(
                        input("Enter compounding periods per year m: ")
                    )
                    time = float(input("Enter time in years T: "))
                    result = tvm.future_value_compounding(
                        initial_investment,
                        rate,
                        periods_per_year,
                        time,
                    )
                    print(f"Compounded Future Value (FV_T): {round(result, 6)}")
                    break
                elif choice == "6":
                    initial_investment = float(input("Enter initial investment C0: "))
                    rate = float(input("Enter continuous rate r (as a decimal): "))
                    time = float(input("Enter time in years T: "))
                    result = tvm.future_value_continuous(initial_investment, rate, time)
                    print(f"Continuous-Compounding FV_T: {round(result, 6)}")
                    break
                elif choice == "7":
                    nominal_rate = float(
                        input("Enter nominal annual rate r (as a decimal): ")
                    )
                    periods_per_year = int(
                        input("Enter compounding periods per year m: ")
                    )
                    result = tvm.effective_annual_rate(nominal_rate, periods_per_year)
                    print(f"Effective Annual Rate (EAR): {round(result, 6)}")
                    break
                elif choice == "8":
                    cash_flow = float(input("Enter annuity cash flow C: "))
                    rate = float(input("Enter discount rate r (as a decimal): "))
                    time_periods = int(input("Enter number of periods T: "))
                    result = tvm.present_value_annuity_formula(
                        cash_flow, rate, time_periods
                    )
                    print(f"PV of Annuity: {round(result, 6)}")
                    break
                elif choice == "9":
                    cash_flow = float(input("Enter annuity cash flow C: "))
                    rate = float(input("Enter discount rate r (as a decimal): "))
                    growth_rate = float(input("Enter growth rate g (as a decimal): "))
                    time_periods = int(input("Enter number of periods T: "))
                    result = tvm.present_value_growing_annuity(
                        cash_flow,
                        rate,
                        growth_rate,
                        time_periods,
                    )
                    print(f"PV of Growing Annuity: {round(result, 6)}")
                    break
                elif choice == "10":
                    cash_flow = float(input("Enter annuity cash flow C: "))
                    rate = float(input("Enter discount rate r (as a decimal): "))
                    time_periods = int(input("Enter number of periods T: "))
                    result = tvm.future_value_annuity(cash_flow, rate, time_periods)
                    print(f"FV of Annuity: {round(result, 6)}")
                    break
                elif choice == "11":
                    cash_flow = float(input("Enter annuity cash flow C: "))
                    rate = float(input("Enter discount rate r (as a decimal): "))
                    time_periods = int(input("Enter number of periods T: "))
                    result = tvm.present_value_annuity_due(
                        cash_flow, rate, time_periods
                    )
                    print(f"PV of Annuity Due: {round(result, 6)}")
                    break
                elif choice == "12":
                    cash_flow = float(input("Enter perpetuity cash flow C: "))
                    rate = float(input("Enter discount rate r (as a decimal): "))
                    result = tvm.present_value_perpetuity(cash_flow, rate)
                    print(f"PV of Perpetuity: {round(result, 6)}")
                    break
                elif choice == "13":
                    next_cash_flow = float(
                        input("Enter next-period cash flow C_(t+1): ")
                    )
                    rate = float(input("Enter discount rate r (as a decimal): "))
                    growth_rate = float(input("Enter growth rate g (as a decimal): "))
                    result = tvm.present_value_growing_perpetuity(
                        next_cash_flow,
                        rate,
                        growth_rate,
                    )
                    print(f"PV of Growing Perpetuity: {round(result, 6)}")
                    break
                elif choice == "14":
                    principal = float(input("Enter loan principal: "))
                    rate = float(input("Enter annual interest rate (as a decimal): "))
                    time = float(input("Enter loan term in years: "))
                    compounding_frequency = int(
                        input("Enter payments per year (e.g., 12 for monthly): ")
                    )
                    calc = tvm(principal=principal, rate=rate, time=time)
                    result = calc.annuity_payment(
                        compounding_frequency=compounding_frequency
                    )
                    print(f"Periodic Annuity Payment: {round(result, 6)}")
                    break
                print("---" * 5)
            except Exception as e:
                print(f"Error: {e}")

    def cashflow_sub_menu(self, choice):
        while True:
            try:
                calc = cfc()
                if choice == "1":
                    cash_flows = self._parse_float_list(
                        input("Enter cash flows CF0, CF1, ... separated by commas: ")
                    )
                    discount_rate = float(
                        input("Enter discount rate r (as a decimal): ")
                    )
                    result = calc.npv(discount_rate, cash_flows)
                    print(f"Net Present Value (NPV): {round(result, 6)}")
                    break
                elif choice == "2":
                    future_cash_flows = self._parse_float_list(
                        input("Enter future cash flows CF1..CFN separated by commas: ")
                    )
                    initial_outlay = float(
                        input("Enter initial outlay C0 (positive): ")
                    )
                    discount_rate = float(
                        input("Enter discount rate r (as a decimal): ")
                    )
                    result = calc.npv_with_initial_outlay(
                        discount_rate,
                        future_cash_flows,
                        initial_outlay,
                    )
                    print(f"Net Present Value (NPV): {round(result, 6)}")
                    break
                elif choice == "3":
                    cash_flows = self._parse_float_list(
                        input("Enter cash flows CF0, CF1, ... separated by commas: ")
                    )
                    result = calc.irr(cash_flows)
                    print(f"Internal Rate of Return (IRR): {round(result, 6)}")
                    break
                elif choice == "4":
                    cash_flows = self._parse_float_list(
                        input("Enter cash flows CF0, CF1, ... separated by commas: ")
                    )
                    result = calc.payback_period(cash_flows)
                    print(
                        f"Payback Period: {round(result, 6) if result is not None else result} years"
                    )
                    break
                elif choice == "5":
                    cash_flows = self._parse_float_list(
                        input("Enter cash flows CF0, CF1, ... separated by commas: ")
                    )
                    discount_rate = float(
                        input("Enter discount rate r (as a decimal): ")
                    )
                    result = calc.discounted_payback_period(discount_rate, cash_flows)
                    print(
                        "Discounted Payback Period: "
                        f"{round(result, 6) if result is not None else result} years"
                    )
                    break
                elif choice == "6":
                    cash_flows = self._parse_float_list(
                        input(
                            "Enter cash flows (include C0 first) separated by commas: "
                        )
                    )
                    discount_rate = float(
                        input("Enter discount rate r (as a decimal): ")
                    )
                    initial_outlay_raw = input(
                        "Optional explicit C0 override (leave blank to infer from first cash flow): "
                    ).strip()
                    initial_outlay = (
                        float(initial_outlay_raw) if initial_outlay_raw else None
                    )
                    result = calc.profitability_index(
                        discount_rate,
                        cash_flows,
                        initial_outlay,
                    )
                    print(f"Profitability Index (PI): {round(result, 6)}")
                    break
                elif choice == "7":
                    npv_value = float(input("Enter project NPV: "))
                    discount_rate = float(
                        input("Enter discount rate r (as a decimal): ")
                    )
                    periods = int(input("Enter number of periods T: "))
                    result = calc.equivalent_annual_benefit(
                        npv_value, discount_rate, periods
                    )
                    print(f"Equivalent Annual Benefit (EAB): {round(result, 6)}")
                    break
                elif choice == "8":
                    corporate_tax_rate = float(
                        input("Enter corporate tax rate Tc (as a decimal): ")
                    )
                    initial_investment = float(input("Enter initial investment C0: "))
                    periods = int(input("Enter depreciation periods N: "))
                    discount_rate = float(
                        input("Enter discount rate r (as a decimal): ")
                    )
                    result = calc.present_value_tax_shield_annuity(
                        corporate_tax_rate,
                        initial_investment,
                        periods,
                        discount_rate,
                    )
                    print(f"PV(Tax Shield): {round(result, 6)}")
                    break
                elif choice == "9":
                    corporate_tax_rate = float(
                        input("Enter corporate tax rate Tc (as a decimal): ")
                    )
                    cca_rate = float(input("Enter CCA rate d (as a decimal): "))
                    initial_cost = float(input("Enter initial cost C0: "))
                    discount_rate = float(
                        input("Enter discount rate k (as a decimal): ")
                    )
                    periods = int(input("Enter project life N: "))
                    salvage_value = float(input("Enter salvage value S: "))
                    result = calc.present_value_ccats(
                        corporate_tax_rate,
                        cca_rate,
                        initial_cost,
                        discount_rate,
                        periods,
                        salvage_value,
                    )
                    print(f"PV(CCATS): {round(result, 6)}")
                    break
                elif choice == "10":
                    discount_rate = float(
                        input("Enter discount rate r (as a decimal): ")
                    )
                    after_tax_operating = self._parse_float_list(
                        input(
                            "Enter after-tax operating cash flows separated by commas: "
                        )
                    )
                    ecf = self._parse_float_list(
                        input("Enter ECF cash flows separated by commas: ")
                    )
                    tax_shield = self._parse_float_list(
                        input("Enter tax-shield cash flows separated by commas: ")
                    )
                    cost_of_asset = float(input("Enter cost of asset: "))
                    result = calc.project_npv_from_components(
                        discount_rate,
                        after_tax_operating,
                        ecf,
                        tax_shield,
                        cost_of_asset,
                    )
                    print(f"Project NPV (Component Form): {round(result, 6)}")
                    break
                elif choice == "11":
                    discount_rate = float(
                        input("Enter discount rate r (as a decimal): ")
                    )
                    periods = int(input("Enter number of periods N: "))
                    result = calc.annuity_factor(discount_rate, periods)
                    print(f"Annuity Factor A_r^N: {round(result, 6)}")
                    break
                print("---" * 5)
            except Exception as e:
                print(f"Error: {e}")
        return

    @staticmethod
    def _binomial_option_price(
        spot,
        strike,
        time_to_maturity,
        risk_free_rate,
        volatility,
        steps,
        option_type,
    ):
        if steps <= 0:
            raise ValueError("steps must be greater than 0.")
        if time_to_maturity <= 0:
            raise ValueError("time_to_maturity must be greater than 0.")
        if volatility <= 0:
            raise ValueError("volatility must be greater than 0.")
        if option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'.")

        dt = time_to_maturity / steps
        up = exp(volatility * sqrt(dt))
        down = 1 / up
        growth = exp(risk_free_rate * dt)
        probability = (growth - down) / (up - down)

        if not 0 <= probability <= 1:
            raise ValueError(
                "Invalid binomial probability. Adjust inputs or increase steps."
            )

        payoffs = []
        for j in range(steps + 1):
            terminal_price = spot * (up**j) * (down ** (steps - j))
            if option_type == "call":
                payoffs.append(max(0.0, terminal_price - strike))
            else:
                payoffs.append(max(0.0, strike - terminal_price))

        discount = exp(-risk_free_rate * dt)
        for step in range(steps, 0, -1):
            for j in range(step):
                payoffs[j] = discount * (
                    probability * payoffs[j + 1] + (1 - probability) * payoffs[j]
                )

        return payoffs[0]

    def bond_valuation_sub_menu(self, choice):
        while True:
            try:
                if choice == "1":
                    face_value = float(input("Enter face value: "))
                    coupon_rate = float(input("Enter coupon rate (as a decimal): "))
                    years_to_maturity = float(input("Enter years to maturity: "))
                    market_rate = float(input("Enter market rate (as a decimal): "))
                    coupon_frequency = int(
                        input("Enter coupon frequency per year (default 1): ") or "1"
                    )
                    calc = bc(
                        face_value,
                        coupon_rate,
                        years_to_maturity,
                        market_rate,
                        coupon_frequency,
                    )
                    result = calc.bond_price()
                    print(f"Bond Price: {round(result, 2)}")
                    break
                elif choice == "2":
                    face_value = float(input("Enter face value: "))
                    coupon_rate = float(input("Enter coupon rate (as a decimal): "))
                    years_to_maturity = float(input("Enter years to maturity: "))
                    price = float(input("Enter current bond price: "))
                    coupon_frequency = int(
                        input("Enter coupon frequency per year (default 1): ") or "1"
                    )
                    calc = bc(
                        face_value,
                        coupon_rate,
                        years_to_maturity,
                        0.0,
                        coupon_frequency,
                    )
                    result = calc.yield_to_maturity(price)
                    print(f"Yield to Maturity: {round(result * 100, 4)}%")
                    break
                elif choice == "3":
                    face_value = float(input("Enter face value: "))
                    coupon_rate = float(input("Enter coupon rate (as a decimal): "))
                    price = float(input("Enter current bond price: "))
                    calc = bc(face_value, coupon_rate, 1, 0.0)
                    result = calc.current_yield(price)
                    print(f"Current Yield: {round(result * 100, 4)}%")
                    break
                elif choice == "4":
                    old_price = float(input("Enter beginning (old) price P0: "))
                    new_price = float(input("Enter ending (new) price Pn: "))
                    incomes_raw = input(
                        "Enter income cash flows separated by commas (leave blank if none): "
                    ).strip()
                    incomes = self._parse_float_list(incomes_raw) if incomes_raw else []
                    result = bc.holding_period_return(new_price, old_price, incomes)
                    print(f"Holding Period Return: {round(result, 6)}")
                    break
                elif choice == "5":
                    spot_rate_n = float(input("Enter spot rate r_n (as a decimal): "))
                    n = int(input("Enter maturity n: "))
                    spot_rate_n_minus_1 = float(
                        input("Enter spot rate r_(n-1) (as a decimal): ")
                    )
                    result = bc.forward_rate(spot_rate_n, n, spot_rate_n_minus_1)
                    print(f"Forward Rate f_n: {round(result, 6)}")
                    break
                print("Invalid option. Please try again.")
            except Exception as e:
                print(f"Error: {e}")
        return

    def equity_valuation_sub_menu(self, choice):
        while True:
            try:
                if choice == "1":
                    dividend_next = float(input("Enter D1 (next dividend): "))
                    discount_rate = float(
                        input("Enter required return r (as a decimal): ")
                    )
                    growth_rate = float(input("Enter growth rate g (as a decimal): "))
                    result = EquityValuation.stock_value_constant_growth(
                        dividend_next,
                        discount_rate,
                        growth_rate,
                    )
                    print(f"Stock Value P0 (Constant Growth): {round(result, 6)}")
                    break
                elif choice == "2":
                    dividend = float(input("Enter dividend D1: "))
                    discount_rate = float(
                        input("Enter required return r (as a decimal): ")
                    )
                    result = EquityValuation.stock_value_zero_growth(
                        dividend,
                        discount_rate,
                    )
                    print(f"Stock Value P0 (Zero Growth): {round(result, 6)}")
                    break
                elif choice == "3":
                    dividend_next = float(input("Enter D1 (next dividend): "))
                    discount_rate = float(
                        input("Enter required return r (as a decimal): ")
                    )
                    growth_rate_stage1 = float(
                        input("Enter stage-1 growth rate g1 (as a decimal): ")
                    )
                    years_stage1 = int(input("Enter stage-1 years N: "))
                    growth_rate_stage2 = float(
                        input("Enter stage-2 growth rate g2 (as a decimal): ")
                    )
                    result = EquityValuation.stock_value_two_stage_growth(
                        dividend_next,
                        discount_rate,
                        growth_rate_stage1,
                        years_stage1,
                        growth_rate_stage2,
                    )
                    print(f"Stock Value P0 (Two-Stage): {round(result, 6)}")
                    break
                elif choice == "4":
                    retention_ratio = float(
                        input("Enter retention ratio RR (as a decimal): ")
                    )
                    roe = float(input("Enter return on equity ROE (as a decimal): "))
                    result = EquityValuation.sustainable_growth_rate(
                        retention_ratio, roe
                    )
                    print(f"Sustainable Growth Rate g: {round(result, 6)}")
                    break
                elif choice == "5":
                    dividend_next = float(input("Enter Div_(t+1): "))
                    price_today = float(input("Enter current price P_t: "))
                    growth_rate = float(input("Enter growth rate g (as a decimal): "))
                    result = EquityValuation.required_return_from_dividend_growth(
                        dividend_next,
                        price_today,
                        growth_rate,
                    )
                    print(f"Required Return r: {round(result, 6)}")
                    break
                elif choice == "6":
                    earnings_per_share = float(input("Enter EPS: "))
                    discount_rate = float(
                        input("Enter required return r (as a decimal): ")
                    )
                    npvgo_per_share_raw = input(
                        "Enter NPVGO per share (leave blank to compute from total NPV): "
                    ).strip()
                    if npvgo_per_share_raw:
                        result = EquityValuation.price_per_share_eps_npvgo(
                            earnings_per_share,
                            discount_rate,
                            npvgo_per_share=float(npvgo_per_share_raw),
                        )
                    else:
                        npv_total = float(input("Enter total NPVGO (NPV): "))
                        shares_outstanding = float(input("Enter # of shares: "))
                        result = EquityValuation.price_per_share_eps_npvgo(
                            earnings_per_share,
                            discount_rate,
                            npv_total=npv_total,
                            shares_outstanding=shares_outstanding,
                        )
                    print(f"Price per Share: {round(result, 6)}")
                    break
                elif choice == "7":
                    free_cash_flows = self._parse_float_list(
                        input("Enter FCF1..FCFN separated by commas: ")
                    )
                    wacc = float(input("Enter r_WACC (as a decimal): "))
                    terminal_value = float(
                        input("Enter terminal value V_N (0 if none): ") or "0"
                    )
                    result = EquityValuation.firm_value_dcf(
                        free_cash_flows,
                        wacc,
                        terminal_value,
                    )
                    print(f"Firm Value V0: {round(result, 6)}")
                    break
                elif choice == "8":
                    fcf_n = float(input("Enter FCF_N: "))
                    wacc = float(input("Enter r_WACC (as a decimal): "))
                    growth_rate = float(input("Enter growth rate g (as a decimal): "))
                    result = EquityValuation.terminal_value_perpetual_growth(
                        fcf_n,
                        wacc,
                        growth_rate,
                    )
                    print(f"Terminal Value V_N: {round(result, 6)}")
                    break
                elif choice == "9":
                    firm_value = float(input("Enter firm value V0: "))
                    cash_0 = float(input("Enter cash_0: "))
                    debt_0 = float(input("Enter debt_0: "))
                    shares_outstanding_0 = float(input("Enter shares outstanding_0: "))
                    result = EquityValuation.equity_value_per_share(
                        firm_value,
                        cash_0,
                        debt_0,
                        shares_outstanding_0,
                    )
                    print(f"Equity Value per Share: {round(result, 6)}")
                    break
                elif choice == "10":
                    equity_value = float(input("Enter equity value: "))
                    debt = float(input("Enter debt: "))
                    cash = float(input("Enter cash: "))
                    result = EquityValuation.enterprise_value(equity_value, debt, cash)
                    print(f"Enterprise Value: {round(result, 6)}")
                    break
                elif choice == "11":
                    fcfe_next_year = float(input("Enter expected FCFE next year: "))
                    required_return = float(
                        input("Enter required return on equity r (as a decimal): ")
                    )
                    growth_rate = float(input("Enter growth rate g (as a decimal): "))
                    shares_outstanding = float(input("Enter shares outstanding: "))
                    result = EquityValuation.fcfe_value_per_share(
                        fcfe_next_year,
                        required_return,
                        growth_rate,
                        shares_outstanding,
                    )
                    print(f"FCFE Value per Share: {round(result, 6)}")
                    break
                elif choice == "12":
                    price = float(input("Enter current stock price: "))
                    earnings_per_share = float(
                        input("Enter earnings per share (EPS): ")
                    )
                    result = EquityValuation.price_to_earnings_ratio(
                        price,
                        earnings_per_share,
                    )
                    print(f"Price/Earnings Ratio: {round(result, 6)}")
                    break
                elif choice == "13":
                    previous_earnings = float(input("Enter previous earnings: "))
                    current_earnings = float(input("Enter current earnings: "))
                    result = EquityValuation.earnings_growth_rate(
                        previous_earnings,
                        current_earnings,
                    )
                    print(f"Earnings Growth Rate: {round(result, 6)}")
                    break
                elif choice == "14":
                    risk_free_rate = float(
                        input("Enter risk-free rate R_f (as a decimal): ")
                    )
                    beta = float(input("Enter beta: "))
                    market_return = float(
                        input("Enter expected market return E(R_m) (as a decimal): ")
                    )
                    result = EquityValuation.required_rate_of_return(
                        risk_free_rate,
                        beta,
                        market_return,
                    )
                    print(f"CAPM Required Return: {round(result, 6)}")
                    break
                print("Invalid option. Please try again.")
            except Exception as e:
                print(f"Error: {e}")
        return

    def portfolio_metrics_sub_menu(self, choice):
        while True:
            try:
                if choice == "1":
                    n_assets = int(input("Enter number of assets: "))
                    portfolio = Portfolio()
                    for i in range(1, n_assets + 1):
                        value = float(input(f"Enter value for asset {i}: "))
                        expected_return = float(
                            input(
                                f"Enter expected return for asset {i} (as a decimal): "
                            )
                        )
                        portfolio.add_asset(
                            {
                                "name": f"Asset {i}",
                                "value": value,
                                "expected_return": expected_return,
                                "risk": 0.0,
                            }
                        )
                    result = portfolio.expected_return()
                    print(f"Portfolio Expected Return: {round(result * 100, 4)}%")
                    break
                elif choice == "2":
                    n_assets = int(input("Enter number of assets: "))
                    portfolio = Portfolio()
                    for i in range(1, n_assets + 1):
                        value = float(input(f"Enter value for asset {i}: "))
                        risk = float(
                            input(
                                f"Enter risk (standard deviation) for asset {i} as a decimal: "
                            )
                        )
                        portfolio.add_asset(
                            {
                                "name": f"Asset {i}",
                                "value": value,
                                "expected_return": 0.0,
                                "risk": risk,
                            }
                        )
                    variance = portfolio.risk_assessment()
                    print(f"Portfolio Variance: {round(variance, 6)}")
                    print(f"Portfolio Risk (Std Dev): {round(sqrt(variance), 6)}")
                    break
                print("Invalid option. Please try again.")
            except Exception as e:
                print(f"Error: {e}")
        return

    def derivative_pricing_sub_menu(self, choice):
        while True:
            try:
                if choice == "1":
                    spot = float(input("Enter current underlying price (S): "))
                    strike = float(input("Enter strike price (K): "))
                    time_to_maturity = float(
                        input("Enter time to maturity in years (T): ")
                    )
                    risk_free_rate = float(
                        input("Enter risk-free rate (as a decimal): ")
                    )
                    volatility = float(input("Enter volatility (as a decimal): "))
                    option_type = (
                        input("Enter option type (call/put): ").strip().lower()
                    )

                    calc = DerivativesCalculator()
                    result = calc.black_scholes(
                        spot,
                        strike,
                        time_to_maturity,
                        risk_free_rate,
                        volatility,
                        option_type,
                    )
                    print(f"Black-Scholes Option Price: {round(result, 6)}")
                    break
                elif choice == "2":
                    spot = float(input("Enter current underlying price (S): "))
                    strike = float(input("Enter strike price (K): "))
                    time_to_maturity = float(
                        input("Enter time to maturity in years (T): ")
                    )
                    risk_free_rate = float(
                        input("Enter risk-free rate (as a decimal): ")
                    )
                    volatility = float(input("Enter volatility (as a decimal): "))
                    steps = int(input("Enter number of binomial steps: "))
                    option_type = (
                        input("Enter option type (call/put): ").strip().lower()
                    )

                    result = self._binomial_option_price(
                        spot,
                        strike,
                        time_to_maturity,
                        risk_free_rate,
                        volatility,
                        steps,
                        option_type,
                    )
                    print(f"Binomial Tree Option Price: {round(result, 6)}")
                    break
                print("Invalid option. Please try again.")
            except Exception as e:
                print(f"Error: {e}")
        return

    def currency_conversion_sub_menu(self, choice):
        while True:
            try:
                if choice == "1":
                    amount = float(input("Enter amount to convert: "))
                    from_currency = (
                        input("Enter source currency code (e.g., USD): ")
                        .strip()
                        .upper()
                    )
                    to_currency = (
                        input("Enter target currency code (e.g., EUR): ")
                        .strip()
                        .upper()
                    )
                    from_rate = float(
                        input(f"Enter {from_currency} rate to the same base currency: ")
                    )
                    to_rate = float(
                        input(f"Enter {to_currency} rate to the same base currency: ")
                    )

                    converter = CurrencyConverter(
                        {from_currency: from_rate, to_currency: to_rate}
                    )
                    result = converter.convert(amount, from_currency, to_currency)
                    print(f"Converted Amount ({to_currency}): {round(result, 6)}")
                    break
                elif choice == "2":
                    spot_rate = float(input("Enter current spot rate (S): "))
                    domestic_rate = float(
                        input("Enter domestic interest rate (as a decimal): ")
                    )
                    foreign_rate = float(
                        input("Enter foreign interest rate (as a decimal): ")
                    )
                    time_years = float(input("Enter time to maturity in years: "))

                    result = (
                        spot_rate
                        * ((1 + domestic_rate) ** time_years)
                        / ((1 + foreign_rate) ** time_years)
                    )
                    print(f"Forward Exchange Rate: {round(result, 6)}")
                    break
                print("Invalid option. Please try again.")
            except Exception as e:
                print(f"Error: {e}")
        return

    def yield_curve_analysis_sub_menu(self, choice):
        while True:
            try:
                if choice == "1":
                    face_value = float(input("Enter face value: "))
                    coupon_rate = float(input("Enter coupon rate (as a decimal): "))
                    years_to_maturity = int(input("Enter years to maturity: "))
                    price = float(input("Enter current bond price: "))
                    calc = bc(face_value, coupon_rate, years_to_maturity, 0.0)
                    result = calc.yield_to_maturity(price)
                    print(f"Yield to Maturity: {round(result * 100, 4)}%")
                    break
                elif choice == "2":
                    face_value = float(input("Enter face value: "))
                    coupon_rate = float(input("Enter coupon rate (as a decimal): "))
                    years_to_maturity = int(input("Enter years to maturity: "))
                    ytm = float(input("Enter yield to maturity (as a decimal): "))

                    coupon_payment = face_value * coupon_rate
                    price = 0.0
                    weighted_pv = 0.0
                    for t in range(1, years_to_maturity + 1):
                        cash_flow = coupon_payment
                        if t == years_to_maturity:
                            cash_flow += face_value
                        pv = cash_flow / (1 + ytm) ** t
                        price += pv
                        weighted_pv += t * pv

                    macaulay_duration = weighted_pv / price
                    modified_duration = macaulay_duration / (1 + ytm)
                    print(f"Macaulay Duration: {round(macaulay_duration, 6)}")
                    print(f"Modified Duration: {round(modified_duration, 6)}")
                    break
                elif choice == "3":
                    face_value = float(input("Enter face value: "))
                    coupon_rate = float(input("Enter coupon rate (as a decimal): "))
                    years_to_maturity = int(input("Enter years to maturity: "))
                    ytm = float(input("Enter yield to maturity (as a decimal): "))

                    coupon_payment = face_value * coupon_rate
                    price = 0.0
                    convexity_numerator = 0.0
                    for t in range(1, years_to_maturity + 1):
                        cash_flow = coupon_payment
                        if t == years_to_maturity:
                            cash_flow += face_value
                        pv = cash_flow / (1 + ytm) ** t
                        price += pv
                        convexity_numerator += t * (t + 1) * pv

                    convexity = convexity_numerator / (price * (1 + ytm) ** 2)
                    print(f"Bond Convexity: {round(convexity, 6)}")
                    break
                print("Invalid option. Please try again.")
            except Exception as e:
                print(f"Error: {e}")
        return


if __name__ == "__main__":
    cli = CLI()
    cli.run()
