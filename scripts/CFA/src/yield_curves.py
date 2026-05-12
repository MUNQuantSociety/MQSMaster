class YieldCurveCalculator:
    """
    A class to handle yield curve calculations and analysis.
    """

    def __init__(self):
        pass

    def calculate_spot_rates(self, cash_flows, maturities, face_value=100.0):
        """
        Calculate spot rates from zero-coupon bond prices and maturities.

        :param cash_flows: List of zero-coupon prices
        :param maturities: List of maturities corresponding to prices
        :param face_value: Redemption value of each zero-coupon bond
        :return: List of calculated spot rates
        """
        if len(cash_flows) != len(maturities):
            raise ValueError("cash_flows and maturities must have the same length.")
        if face_value <= 0:
            raise ValueError("face_value must be greater than 0.")

        spot_rates = []
        for price, maturity in zip(cash_flows, maturities):
            if price <= 0:
                raise ValueError("All prices must be greater than 0.")
            if maturity <= 0:
                raise ValueError("All maturities must be greater than 0.")

            spot_rate = (face_value / price) ** (1 / maturity) - 1
            spot_rates.append(spot_rate)
        return spot_rates

    def calculate_forward_rates(self, spot_rates, maturities=None):
        """
        Calculate forward rates from spot rates.

        :param spot_rates: List of spot rates
        :param maturities: Optional list of maturities for each spot rate
        :return: List of calculated forward rates
        """
        if len(spot_rates) < 2:
            return []

        if maturities is None:
            maturities = list(range(1, len(spot_rates) + 1))
        if len(spot_rates) != len(maturities):
            raise ValueError("spot_rates and maturities must have the same length.")

        forward_rates = []
        for i in range(1, len(spot_rates)):
            t1 = maturities[i - 1]
            t2 = maturities[i]
            if t2 <= t1:
                raise ValueError("maturities must be strictly increasing.")

            forward_rate = (
                (1 + spot_rates[i]) ** t2 / (1 + spot_rates[i - 1]) ** t1
            ) ** (1 / (t2 - t1)) - 1
            forward_rates.append(forward_rate)
        return forward_rates

    def plot_yield_curve(self, maturities, rates):
        """
        Plot the yield curve based on maturities and rates.

        :param maturities: List of maturities
        :param rates: List of rates (spot or forward)
        """
        import matplotlib.pyplot as plt

        if len(maturities) != len(rates):
            raise ValueError("maturities and rates must have the same length.")

        plt.figure(figsize=(10, 6))
        plt.plot(maturities, rates, marker="o")
        plt.title("Yield Curve")
        plt.xlabel("Maturity (Years)")
        plt.ylabel("Yield (%)")
        plt.grid()
        plt.show()

    @staticmethod
    def one_period_forward_rate(spot_rate_n, n, spot_rate_n_minus_1):
        if n <= 1:
            raise ValueError("n must be greater than 1.")
        if spot_rate_n <= -1 or spot_rate_n_minus_1 <= -1:
            raise ValueError("spot rates must be greater than -1.")

        return ((1 + spot_rate_n) ** n / (1 + spot_rate_n_minus_1) ** (n - 1)) - 1
