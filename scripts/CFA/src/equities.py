class EquityValuation:
    """
    A class to perform equity valuation calculations.
    """

    @staticmethod
    def dividend_discount_model(dividend, growth_rate, discount_rate):
        """
        Calculate the intrinsic value of a stock using the Dividend Discount Model (DDM).

        :param dividend: Expected annual dividend payment
        :param growth_rate: Expected growth rate of dividends (as a decimal)
        :param discount_rate: Required rate of return (as a decimal)
        :return: Intrinsic value of the stock
        """
        return EquityValuation.stock_value_constant_growth(
            dividend, discount_rate, growth_rate
        )

    @staticmethod
    def price_to_earnings_ratio(price, earnings_per_share):
        """
        Calculate the Price-to-Earnings (P/E) ratio.

        :param price: Current market price of the stock
        :param earnings_per_share: Earnings per share (EPS)
        :return: P/E ratio
        """
        if price <= 0:
            raise ValueError("Price must be greater than zero.")
        if earnings_per_share <= 0:
            raise ValueError("Earnings per share must be greater than zero.")
        return price / earnings_per_share

    @staticmethod
    def earnings_growth_rate(previous_earnings, current_earnings):
        """
        Calculate the earnings growth rate.

        :param previous_earnings: Earnings from the previous period
        :param current_earnings: Earnings from the current period
        :return: Growth rate (as a decimal)
        """
        if previous_earnings <= 0:
            raise ValueError("Previous earnings must be greater than zero.")
        return (current_earnings - previous_earnings) / previous_earnings

    @staticmethod
    def required_rate_of_return(risk_free_rate, beta, market_return):
        """
        Calculate the required rate of return using the Capital Asset Pricing Model (CAPM).

        :param risk_free_rate: Risk-free rate of return (as a decimal)
        :param beta: Beta of the stock
        :param market_return: Expected market return (as a decimal)
        :return: Required rate of return (as a decimal)
        """
        return risk_free_rate + beta * (market_return - risk_free_rate)

    @staticmethod
    def stock_value_constant_growth(dividend_next, discount_rate, growth_rate):
        if dividend_next < 0:
            raise ValueError("dividend_next cannot be negative.")
        if discount_rate <= growth_rate:
            raise ValueError("discount_rate must be greater than growth_rate.")
        return dividend_next / (discount_rate - growth_rate)

    @staticmethod
    def stock_value_zero_growth(dividend, discount_rate):
        if dividend < 0:
            raise ValueError("dividend cannot be negative.")
        if discount_rate <= 0:
            raise ValueError("discount_rate must be greater than 0.")
        return dividend / discount_rate

    @staticmethod
    def stock_value_two_stage_growth(
        dividend_next,
        discount_rate,
        growth_rate_stage1,
        years_stage1,
        growth_rate_stage2,
    ):
        if years_stage1 <= 0:
            raise ValueError("years_stage1 must be greater than 0.")
        if discount_rate == growth_rate_stage1:
            raise ValueError("discount_rate cannot equal growth_rate_stage1.")
        if discount_rate <= growth_rate_stage2:
            raise ValueError("discount_rate must be greater than growth_rate_stage2.")

        pv_stage1 = (dividend_next / (discount_rate - growth_rate_stage1)) * (
            1 - ((1 + growth_rate_stage1) / (1 + discount_rate)) ** years_stage1
        )

        dividend_n = dividend_next * (1 + growth_rate_stage1) ** (years_stage1 - 1)
        dividend_n_plus_1 = dividend_n * (1 + growth_rate_stage2)
        terminal_value_n = dividend_n_plus_1 / (discount_rate - growth_rate_stage2)
        pv_terminal = terminal_value_n / (1 + discount_rate) ** years_stage1

        return pv_stage1 + pv_terminal

    @staticmethod
    def sustainable_growth_rate(retention_ratio, roe):
        if not 0 <= retention_ratio <= 1:
            raise ValueError("retention_ratio must be between 0 and 1.")
        return retention_ratio * roe

    @staticmethod
    def required_return_from_dividend_growth(dividend_next, price_today, growth_rate):
        if price_today <= 0:
            raise ValueError("price_today must be greater than 0.")
        return (dividend_next / price_today) + growth_rate

    @staticmethod
    def price_per_share_eps_npvgo(
        earnings_per_share,
        discount_rate,
        npvgo_per_share=None,
        npv_total=None,
        shares_outstanding=None,
    ):
        if discount_rate <= 0:
            raise ValueError("discount_rate must be greater than 0.")

        if npvgo_per_share is None:
            if npv_total is None or shares_outstanding is None:
                npvgo_per_share = 0.0
            else:
                if shares_outstanding <= 0:
                    raise ValueError("shares_outstanding must be greater than 0.")
                npvgo_per_share = npv_total / shares_outstanding

        return (earnings_per_share / discount_rate) + npvgo_per_share

    @staticmethod
    def firm_value_dcf(free_cash_flows, wacc, terminal_value=0.0):
        if free_cash_flows is None or len(free_cash_flows) == 0:
            raise ValueError("free_cash_flows must be a non-empty list of numbers.")
        if wacc <= -1:
            raise ValueError("wacc must be greater than -1.")

        value = 0.0
        for t, fcf in enumerate(free_cash_flows, start=1):
            value += fcf / (1 + wacc) ** t
        value += terminal_value / (1 + wacc) ** len(free_cash_flows)
        return value

    @staticmethod
    def terminal_value_perpetual_growth(fcf_n, wacc, growth_rate):
        if wacc <= growth_rate:
            raise ValueError("wacc must be greater than growth_rate.")
        return (fcf_n * (1 + growth_rate)) / (wacc - growth_rate)

    @staticmethod
    def equity_value_per_share(firm_value, cash_0, debt_0, shares_outstanding_0):
        if shares_outstanding_0 <= 0:
            raise ValueError("shares_outstanding_0 must be greater than 0.")
        return (firm_value + cash_0 - debt_0) / shares_outstanding_0

    @staticmethod
    def enterprise_value(equity_value, debt, cash):
        return equity_value + debt - cash

    @staticmethod
    def fcfe_value_per_share(
        fcfe_next_year, required_return, growth_rate, shares_outstanding
    ):
        if required_return <= growth_rate:
            raise ValueError("required_return must be greater than growth_rate.")
        if shares_outstanding <= 0:
            raise ValueError("shares_outstanding must be greater than 0.")

        total_equity_value = fcfe_next_year / (required_return - growth_rate)
        return total_equity_value / shares_outstanding
