from math import exp, log


class TimeValueOfMoney:
    def __init__(self, principal, rate, time):
        if time <= 0:
            raise ValueError("time must be greater than 0.")
        if rate <= -1:
            raise ValueError("rate must be greater than -1.")
        self.principal = principal
        self.rate = rate
        self.time = time

    def future_value(self):
        return self.principal * (1 + self.rate) ** self.time

    def present_value(self):
        return self.principal / (1 + self.rate) ** self.time

    def annuity_payment(self, compounding_frequency=1):
        if compounding_frequency <= 0:
            raise ValueError("compounding_frequency must be greater than 0.")

        periodic_rate = self.rate / compounding_frequency
        number_of_payments = self.time * compounding_frequency

        if periodic_rate == 0:
            return self.principal / number_of_payments

        return self.principal * (
            periodic_rate / (1 - (1 + periodic_rate) ** -number_of_payments)
        )

    def present_value_annuity(self, payment=0.0, compounding_frequency=1):
        if compounding_frequency <= 0:
            raise ValueError("compounding_frequency must be greater than 0.")

        periodic_rate = self.rate / compounding_frequency
        number_of_payments = self.time * compounding_frequency

        if periodic_rate == 0:
            return payment * number_of_payments

        return payment * (
            (1 - (1 + periodic_rate) ** -number_of_payments) / periodic_rate
        )

    @staticmethod
    def solve_time_periods(present_value, future_value, rate):
        if present_value <= 0 or future_value <= 0:
            raise ValueError("present_value and future_value must be greater than 0.")
        if rate <= -1:
            raise ValueError("rate must be greater than -1.")
        if rate == 0:
            raise ValueError("rate cannot be 0 when solving with logarithms.")

        return log(future_value / present_value) / log(1 + rate)

    @staticmethod
    def solve_discount_rate(present_value, future_value, time_periods):
        if present_value <= 0 or future_value <= 0:
            raise ValueError("present_value and future_value must be greater than 0.")
        if time_periods <= 0:
            raise ValueError("time_periods must be greater than 0.")

        return (future_value / present_value) ** (1 / time_periods) - 1

    @staticmethod
    def future_value_compounding(initial_investment, rate, periods_per_year, time):
        if initial_investment < 0:
            raise ValueError("initial_investment cannot be negative.")
        if rate <= -1:
            raise ValueError("rate must be greater than -1.")
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be greater than 0.")
        if time < 0:
            raise ValueError("time cannot be negative.")

        return initial_investment * (1 + rate / periods_per_year) ** (
            periods_per_year * time
        )

    @staticmethod
    def future_value_continuous(initial_investment, rate, time):
        if initial_investment < 0:
            raise ValueError("initial_investment cannot be negative.")
        if time < 0:
            raise ValueError("time cannot be negative.")

        return initial_investment * exp(rate * time)

    @staticmethod
    def effective_annual_rate(nominal_rate, periods_per_year):
        if nominal_rate <= -1:
            raise ValueError("nominal_rate must be greater than -1.")
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be greater than 0.")

        return (1 + nominal_rate / periods_per_year) ** periods_per_year - 1

    @staticmethod
    def present_value_annuity_formula(cash_flow, rate, time_periods):
        if time_periods <= 0:
            raise ValueError("time_periods must be greater than 0.")
        if rate <= -1:
            raise ValueError("rate must be greater than -1.")
        if rate == 0:
            return cash_flow * time_periods

        return (cash_flow / rate) * (1 - (1 / (1 + rate) ** time_periods))

    @staticmethod
    def present_value_growing_annuity(cash_flow, rate, growth_rate, time_periods):
        if time_periods <= 0:
            raise ValueError("time_periods must be greater than 0.")
        if rate <= -1:
            raise ValueError("rate must be greater than -1.")
        if rate <= growth_rate:
            raise ValueError("rate must be greater than growth_rate.")

        return (cash_flow / (rate - growth_rate)) * (
            1 - ((1 + growth_rate) / (1 + rate)) ** time_periods
        )

    @staticmethod
    def future_value_annuity(cash_flow, rate, time_periods):
        if time_periods <= 0:
            raise ValueError("time_periods must be greater than 0.")
        if rate <= -1:
            raise ValueError("rate must be greater than -1.")
        if rate == 0:
            return cash_flow * time_periods

        return cash_flow * (((1 + rate) ** time_periods - 1) / rate)

    @staticmethod
    def present_value_annuity_due(cash_flow, rate, time_periods):
        return TimeValueOfMoney.present_value_annuity_formula(
            cash_flow, rate, time_periods
        ) * (1 + rate)

    @staticmethod
    def present_value_perpetuity(cash_flow, rate):
        if rate <= 0:
            raise ValueError("rate must be greater than 0.")
        return cash_flow / rate

    @staticmethod
    def present_value_growing_perpetuity(next_cash_flow, rate, growth_rate):
        if rate <= growth_rate:
            raise ValueError("rate must be greater than growth_rate.")
        return next_cash_flow / (rate - growth_rate)
