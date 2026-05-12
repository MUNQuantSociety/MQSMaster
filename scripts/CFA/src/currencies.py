class CurrencyConverter:
    """
    A class to handle currency conversion and foreign exchange calculations.
    """

    def __init__(self, exchange_rates):
        """
        Initialize the CurrencyConverter with a dictionary of exchange rates.

        :param exchange_rates: A dictionary where keys are currency codes and values are their exchange rates to a base currency.
        """
        self.exchange_rates = exchange_rates

    def convert(self, amount, from_currency, to_currency):
        """
        Convert an amount from one currency to another.

        :param amount: The amount of money to convert.
        :param from_currency: The currency code of the amount to convert from.
        :param to_currency: The currency code of the amount to convert to.
        :return: The converted amount in the target currency.
        """
        if (
            from_currency not in self.exchange_rates
            or to_currency not in self.exchange_rates
        ):
            raise ValueError("Invalid currency code provided.")
        if amount < 0:
            raise ValueError("amount cannot be negative.")
        if (
            self.exchange_rates[from_currency] <= 0
            or self.exchange_rates[to_currency] <= 0
        ):
            raise ValueError("Exchange rates must be greater than 0.")

        # Convert amount to base currency (assumed to be the first currency in the rates)
        base_amount = amount / self.exchange_rates[from_currency]
        # Convert base currency to target currency
        converted_amount = base_amount * self.exchange_rates[to_currency]
        return converted_amount

    def add_exchange_rate(self, currency, rate):
        """
        Add or update an exchange rate for a currency.

        :param currency: The currency code to add or update.
        :param rate: The exchange rate of the currency to the base currency.
        """
        if rate <= 0:
            raise ValueError("Exchange rate must be greater than 0.")
        self.exchange_rates[currency] = rate

    def remove_exchange_rate(self, currency):
        """
        Remove an exchange rate for a currency.

        :param currency: The currency code to remove.
        """
        if currency in self.exchange_rates:
            del self.exchange_rates[currency]

    @staticmethod
    def forward_exchange_rate(spot_rate, domestic_rate, foreign_rate, time_years):
        if spot_rate <= 0:
            raise ValueError("spot_rate must be greater than 0.")
        if domestic_rate <= -1 or foreign_rate <= -1:
            raise ValueError("interest rates must be greater than -1.")
        if time_years < 0:
            raise ValueError("time_years cannot be negative.")

        return (
            spot_rate
            * ((1 + domestic_rate) ** time_years)
            / ((1 + foreign_rate) ** time_years)
        )
