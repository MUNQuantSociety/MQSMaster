from scipy.optimize import brentq


class BondCalculator:
    """
    A class to perform bond valuation calculations.
    """

    def __init__(
        self,
        face_value,
        coupon_rate,
        years_to_maturity,
        market_rate,
        coupon_frequency=1,
    ):
        if face_value <= 0:
            raise ValueError("face_value must be greater than 0.")
        if coupon_rate < 0:
            raise ValueError("coupon_rate cannot be negative.")
        if years_to_maturity <= 0:
            raise ValueError("years_to_maturity must be greater than 0.")
        if market_rate <= -1:
            raise ValueError("market_rate must be greater than -1.")
        if coupon_frequency <= 0:
            raise ValueError("coupon_frequency must be greater than 0.")

        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.years_to_maturity = years_to_maturity
        self.market_rate = market_rate
        self.coupon_frequency = coupon_frequency

    def _number_of_periods(self):
        periods = int(round(self.years_to_maturity * self.coupon_frequency))
        if periods <= 0:
            raise ValueError("years_to_maturity * coupon_frequency must be >= 1.")
        return periods

    def _coupon_payment_per_period(self):
        return self.face_value * self.coupon_rate / self.coupon_frequency

    def present_value_of_coupons(self):
        """
        Calculate the present value of the bond's coupon payments.
        """
        periodic_market_rate = self.market_rate / self.coupon_frequency
        coupon_payment = self._coupon_payment_per_period()
        periods = self._number_of_periods()
        pv_coupons = sum(
            coupon_payment / (1 + periodic_market_rate) ** t
            for t in range(1, periods + 1)
        )
        return pv_coupons

    def present_value_of_face_value(self):
        """
        Calculate the present value of the bond's face value.
        """
        periodic_market_rate = self.market_rate / self.coupon_frequency
        periods = self._number_of_periods()
        return self.face_value / (1 + periodic_market_rate) ** periods

    def bond_price(self):
        """
        Calculate the total price of the bond.
        """
        return self.present_value_of_coupons() + self.present_value_of_face_value()

    def _price_from_ytm(self, ytm):
        periodic_ytm = ytm / self.coupon_frequency
        if periodic_ytm <= -1:
            raise ValueError("ytm implies an invalid periodic discount rate.")

        coupon_payment = self._coupon_payment_per_period()
        periods = self._number_of_periods()
        pv_coupons = sum(
            coupon_payment / (1 + periodic_ytm) ** t for t in range(1, periods + 1)
        )
        pv_face = self.face_value / (1 + periodic_ytm) ** periods
        return pv_coupons + pv_face

    def yield_to_maturity(self, price):
        """
        Calculate annualized yield to maturity (YTM) by solving the bond pricing equation.
        """
        if price <= 0:
            raise ValueError("price must be greater than 0.")

        def objective(ytm):
            return self._price_from_ytm(ytm) - price

        lower_bound = -0.95
        upper_bound = 1.0
        f_low = objective(lower_bound)
        f_high = objective(upper_bound)

        while f_low * f_high > 0 and upper_bound < 100:
            upper_bound *= 2
            f_high = objective(upper_bound)

        if f_low * f_high > 0:
            raise ValueError(
                "Could not bracket a YTM root for the provided bond inputs and price."
            )

        return brentq(objective, lower_bound, upper_bound, xtol=1e-12, maxiter=1000)

    def current_yield(self, price):
        """
        Calculate the current yield of the bond.
        """
        if price <= 0:
            raise ValueError("price must be greater than 0.")
        return (self.face_value * self.coupon_rate) / price

    @staticmethod
    def holding_period_return(new_price, old_price, incomes=None):
        if old_price <= 0:
            raise ValueError("old_price must be greater than 0.")
        if incomes is None:
            incomes = []

        return (new_price - old_price + sum(incomes)) / old_price

    @staticmethod
    def forward_rate(spot_rate_n, n, spot_rate_n_minus_1):
        if n <= 1:
            raise ValueError("n must be greater than 1.")
        if spot_rate_n <= -1 or spot_rate_n_minus_1 <= -1:
            raise ValueError("spot rates must be greater than -1.")

        return ((1 + spot_rate_n) ** n / (1 + spot_rate_n_minus_1) ** (n - 1)) - 1


def main():
    # Example usage of the BondCalculator class
    face_value = 1000.0
    coupon_rate = 0.055
    years_to_maturity = 1
    market_rate = 0.0129  # 1.29% market rate

    bond_calculator = BondCalculator(
        face_value, coupon_rate, years_to_maturity, market_rate
    )
    price = bond_calculator.bond_price()
    ytm = bond_calculator.yield_to_maturity(price)
    current_yield = bond_calculator.current_yield(price)

    print(f"Bond Price: {price:.2f}")
    print(
        f"Coupon Payment: {bond_calculator.face_value * bond_calculator.coupon_rate:.2f}"
    )
    print(f"Yield to Maturity: {ytm:.2%}")
    print(f"Current Yield: {current_yield:.2%}")


if __name__ == "__main__":
    main()
