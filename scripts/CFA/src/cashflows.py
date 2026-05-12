class CashFlowCalculator:
    """
    A class to handle cash flow analysis calculations.
    """

    @staticmethod
    def npv(rate, cash_flows):
        """
        Calculate the net present value (NPV) of a series of cash flows.

        :param rate: Discount rate (as a decimal)
        :param cash_flows: List of cash flows
        :return: Net present value
        """
        if cash_flows is None or len(cash_flows) == 0:
            raise ValueError("cash_flows must be a non-empty list of numbers.")
        if rate <= -1:
            raise ValueError("rate must be greater than -1.")

        npv = 0
        for t, cash_flow in enumerate(cash_flows):
            npv += cash_flow / (1 + rate) ** t
        return npv

    @staticmethod
    def irr(cash_flows):
        """
        Calculate the internal rate of return (IRR) for a series of cash flows.

        :param cash_flows: List of cash flows
        :return: Internal rate of return
        """
        if cash_flows is None or len(cash_flows) == 0:
            raise ValueError("cash_flows must be a non-empty list of numbers.")

        has_negative = any(cf < 0 for cf in cash_flows)
        has_positive = any(cf > 0 for cf in cash_flows)
        if not (has_negative and has_positive):
            raise ValueError(
                "IRR requires at least one negative and one positive cash flow."
            )

        from scipy.optimize import brentq

        def npv_func(rate):
            return CashFlowCalculator.npv(rate, cash_flows)

        # Probe a wide range of rates and solve on the first sign-change interval.
        probe_rates = [
            -0.9999,
            -0.9,
            -0.75,
            -0.5,
            -0.25,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
        ]
        prev_rate = probe_rates[0]
        prev_npv = npv_func(prev_rate)

        if prev_npv == 0:
            return prev_rate

        for current_rate in probe_rates[1:]:
            current_npv = npv_func(current_rate)
            if current_npv == 0:
                return current_rate
            if prev_npv * current_npv < 0:
                return brentq(
                    npv_func, prev_rate, current_rate, xtol=1e-12, maxiter=1000
                )
            prev_rate = current_rate
            prev_npv = current_npv

        raise ValueError(
            "IRR could not be determined for these cash flows. "
            "Try including a clear initial outflow and subsequent inflows."
        )

    @staticmethod
    def payback_period(cash_flows):
        """
        Calculate the payback period for a series of cash flows.

        :param cash_flows: List of cash flows
        :return: Payback period in years
        """
        if cash_flows is None or len(cash_flows) == 0:
            raise ValueError("cash_flows must be a non-empty list of numbers.")

        cumulative_cash_flow = 0.0
        for t, cash_flow in enumerate(cash_flows):
            previous_cumulative = cumulative_cash_flow
            cumulative_cash_flow += cash_flow
            if cumulative_cash_flow >= 0:
                if t == 0:
                    return 0.0
                if cash_flow == 0:
                    return float(t)
                fraction = -previous_cumulative / cash_flow
                return (t - 1) + fraction
        return None  # Payback period not reached

    @staticmethod
    def discounted_payback_period(rate, cash_flows):
        """
        Calculate the discounted payback period for a series of cash flows.

        :param rate: Discount rate (as a decimal)
        :param cash_flows: List of cash flows
        :return: Discounted payback period in years
        """
        if cash_flows is None or len(cash_flows) == 0:
            raise ValueError("cash_flows must be a non-empty list of numbers.")
        if rate <= -1:
            raise ValueError("rate must be greater than -1.")

        cumulative_cash_flow = 0.0
        for t, cash_flow in enumerate(cash_flows):
            discounted_cash_flow = cash_flow / (1 + rate) ** t
            previous_cumulative = cumulative_cash_flow
            cumulative_cash_flow += discounted_cash_flow
            if cumulative_cash_flow >= 0:
                if t == 0:
                    return 0.0
                if discounted_cash_flow == 0:
                    return float(t)
                fraction = -previous_cumulative / discounted_cash_flow
                return (t - 1) + fraction
        return None  # Discounted payback period not reached

    @staticmethod
    def annuity_factor(rate, periods):
        if periods <= 0:
            raise ValueError("periods must be greater than 0.")
        if rate == 0:
            return float(periods)
        if rate <= -1:
            raise ValueError("rate must be greater than -1.")

        return (1 - (1 / (1 + rate) ** periods)) / rate

    @staticmethod
    def profitability_index(rate, cash_flows, initial_outlay=None):
        if cash_flows is None or len(cash_flows) == 0:
            raise ValueError("cash_flows must be a non-empty list of numbers.")
        if rate <= -1:
            raise ValueError("rate must be greater than -1.")

        if initial_outlay is None:
            initial_outlay = abs(cash_flows[0])
            future_cash_flows = cash_flows[1:]
        else:
            future_cash_flows = cash_flows

        if initial_outlay <= 0:
            raise ValueError("initial_outlay must be greater than 0.")

        pv_inflows = 0.0
        for t, cash_flow in enumerate(future_cash_flows, start=1):
            pv_inflows += cash_flow / (1 + rate) ** t

        return pv_inflows / initial_outlay

    @staticmethod
    def equivalent_annual_benefit(npv, rate, periods):
        if periods <= 0:
            raise ValueError("periods must be greater than 0.")
        if rate <= -1:
            raise ValueError("rate must be greater than -1.")

        if rate == 0:
            return npv / periods

        annuity_factor = CashFlowCalculator.annuity_factor(rate, periods)
        return npv / annuity_factor

    @staticmethod
    def present_value_tax_shield_annuity(
        corporate_tax_rate,
        initial_investment,
        periods,
        discount_rate,
    ):
        if not 0 <= corporate_tax_rate <= 1:
            raise ValueError("corporate_tax_rate must be between 0 and 1.")
        if initial_investment < 0:
            raise ValueError("initial_investment cannot be negative.")
        if periods <= 0:
            raise ValueError("periods must be greater than 0.")

        annual_depreciation = initial_investment / periods
        annuity_factor = CashFlowCalculator.annuity_factor(discount_rate, periods)
        return (corporate_tax_rate * annual_depreciation) * annuity_factor

    @staticmethod
    def present_value_ccats(
        corporate_tax_rate,
        cca_rate,
        initial_cost,
        discount_rate,
        periods,
        salvage_value=0.0,
    ):
        if not 0 <= corporate_tax_rate <= 1:
            raise ValueError("corporate_tax_rate must be between 0 and 1.")
        if cca_rate < 0:
            raise ValueError("cca_rate cannot be negative.")
        if initial_cost < 0 or salvage_value < 0:
            raise ValueError("initial_cost and salvage_value cannot be negative.")
        if periods <= 0:
            raise ValueError("periods must be greater than 0.")
        if discount_rate <= -1:
            raise ValueError("discount_rate must be greater than -1.")

        base_term = (corporate_tax_rate * cca_rate) / (discount_rate + cca_rate)
        first_term = (
            base_term * initial_cost * ((1 + 1.5 * discount_rate) / (1 + discount_rate))
        )
        second_term = base_term * salvage_value * (1 / (1 + discount_rate) ** periods)
        return first_term - second_term

    @staticmethod
    def project_npv_from_components(
        rate,
        after_tax_operating_cash_flows,
        ecf_cash_flows,
        tax_shield_cash_flows,
        cost_of_asset,
    ):
        pv_operating = CashFlowCalculator.npv(rate, after_tax_operating_cash_flows)
        pv_ecf = CashFlowCalculator.npv(rate, ecf_cash_flows)
        pv_tax_shield = CashFlowCalculator.npv(rate, tax_shield_cash_flows)
        return pv_operating + pv_ecf + pv_tax_shield - cost_of_asset

    @staticmethod
    def npv_with_initial_outlay(rate, future_cash_flows, initial_outlay):
        if future_cash_flows is None:
            raise ValueError("future_cash_flows cannot be None.")
        if initial_outlay < 0:
            raise ValueError("initial_outlay cannot be negative.")

        pv_inflows = 0.0
        for t, cash_flow in enumerate(future_cash_flows, start=1):
            pv_inflows += cash_flow / (1 + rate) ** t

        return pv_inflows - initial_outlay
