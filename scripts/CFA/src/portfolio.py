from math import sqrt


class Portfolio:
    def __init__(self, assets=None):
        if assets is None:
            assets = []
        self.assets = assets

    def add_asset(self, asset):
        self.assets.append(asset)

    def remove_asset(self, asset):
        self.assets.remove(asset)

    def total_value(self):
        return sum(asset["value"] for asset in self.assets)

    def expected_return(self):
        total_return = sum(
            asset["value"] * asset["expected_return"] for asset in self.assets
        )
        return total_return / self.total_value() if self.total_value() > 0 else 0

    def risk_assessment(self):
        total_value = self.total_value()
        if total_value <= 0:
            return 0

        # Zero-correlation approximation: portfolio variance = sum((w_i^2) * sigma_i^2)
        total_variance = 0
        for asset in self.assets:
            weight = asset["value"] / total_value
            total_variance += (weight**2) * (asset["risk"] ** 2)
        return total_variance

    def asset_allocation(self):
        allocation = {
            asset["name"]: asset["value"] / self.total_value() for asset in self.assets
        }
        return allocation if self.total_value() > 0 else {}

    @staticmethod
    def arithmetic_average_return(returns):
        if returns is None or len(returns) == 0:
            raise ValueError("returns must be a non-empty list.")
        return sum(returns) / len(returns)

    @staticmethod
    def geometric_average_return(returns):
        if returns is None or len(returns) == 0:
            raise ValueError("returns must be a non-empty list.")

        product = 1.0
        for r in returns:
            if r <= -1:
                raise ValueError("each return must be greater than -1.")
            product *= 1 + r

        return product ** (1 / len(returns)) - 1

    @staticmethod
    def expected_return_probability(returns, probabilities):
        if returns is None or probabilities is None:
            raise ValueError("returns and probabilities cannot be None.")
        if len(returns) == 0 or len(returns) != len(probabilities):
            raise ValueError(
                "returns and probabilities must have equal non-zero length."
            )

        total_probability = sum(probabilities)
        if abs(total_probability - 1.0) > 1e-8:
            raise ValueError("probabilities must sum to 1.")

        return sum(r * p for r, p in zip(returns, probabilities))

    @staticmethod
    def expected_standard_deviation(returns, probabilities):
        mean_return = Portfolio.expected_return_probability(returns, probabilities)
        variance = sum(
            p * (r - mean_return) ** 2 for r, p in zip(returns, probabilities)
        )
        return sqrt(variance)

    @staticmethod
    def historical_standard_deviation(returns):
        if returns is None or len(returns) < 2:
            raise ValueError("returns must contain at least two values.")

        mean_return = Portfolio.arithmetic_average_return(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        return sqrt(variance)

    @staticmethod
    def expected_portfolio_return_two_assets(
        weight_a, expected_return_a, weight_b, expected_return_b
    ):
        if abs((weight_a + weight_b) - 1.0) > 1e-8:
            raise ValueError("weight_a and weight_b must sum to 1.")
        return (weight_a * expected_return_a) + (weight_b * expected_return_b)

    @staticmethod
    def portfolio_weight(holding_i, all_holdings):
        if all_holdings is None or len(all_holdings) == 0:
            raise ValueError("all_holdings must be a non-empty list.")
        total_holdings = sum(all_holdings)
        if total_holdings <= 0:
            raise ValueError("sum(all_holdings) must be greater than 0.")
        return holding_i / total_holdings

    @staticmethod
    def portfolio_standard_deviation_two_assets(
        weight_a,
        sigma_a,
        weight_b,
        sigma_b,
        covariance_ab=None,
        correlation_ab=None,
    ):
        if abs((weight_a + weight_b) - 1.0) > 1e-8:
            raise ValueError("weight_a and weight_b must sum to 1.")

        if covariance_ab is None:
            if correlation_ab is None:
                raise ValueError("provide covariance_ab or correlation_ab.")
            covariance_ab = sigma_a * sigma_b * correlation_ab

        variance = (
            (weight_a**2) * (sigma_a**2)
            + (weight_b**2) * (sigma_b**2)
            + 2 * weight_a * weight_b * covariance_ab
        )
        return sqrt(variance)

    @staticmethod
    def covariance_two_assets(returns_a, returns_b, probabilities):
        if not (returns_a and returns_b and probabilities):
            raise ValueError(
                "returns_a, returns_b, and probabilities must be non-empty."
            )
        if not (len(returns_a) == len(returns_b) == len(probabilities)):
            raise ValueError("returns and probabilities must have equal lengths.")

        mean_a = Portfolio.expected_return_probability(returns_a, probabilities)
        mean_b = Portfolio.expected_return_probability(returns_b, probabilities)
        return sum(
            p * (ra - mean_a) * (rb - mean_b)
            for ra, rb, p in zip(returns_a, returns_b, probabilities)
        )

    @staticmethod
    def correlation_from_covariance(covariance_ab, sigma_a, sigma_b):
        if sigma_a <= 0 or sigma_b <= 0:
            raise ValueError("sigma_a and sigma_b must be greater than 0.")
        return covariance_ab / (sigma_a * sigma_b)

    @staticmethod
    def beta(covariance_with_market, market_variance):
        if market_variance <= 0:
            raise ValueError("market_variance must be greater than 0.")
        return covariance_with_market / market_variance

    @staticmethod
    def portfolio_beta(weights, betas):
        if weights is None or betas is None or len(weights) == 0:
            raise ValueError("weights and betas must be non-empty lists.")
        if len(weights) != len(betas):
            raise ValueError("weights and betas must have the same length.")
        if abs(sum(weights) - 1.0) > 1e-8:
            raise ValueError("weights must sum to 1.")

        return sum(w * b for w, b in zip(weights, betas))

    @staticmethod
    def capm_expected_return(risk_free_rate, beta_j, market_return):
        return risk_free_rate + beta_j * (market_return - risk_free_rate)

    @staticmethod
    def wacc(
        weight_debt,
        cost_debt,
        tax_rate,
        weight_preferred=0.0,
        cost_preferred=0.0,
        weight_common=0.0,
        cost_common=0.0,
        weight_equity=0.0,
        cost_equity=0.0,
    ):
        if not 0 <= tax_rate <= 1:
            raise ValueError("tax_rate must be between 0 and 1.")

        total_weight = weight_debt + weight_preferred + weight_common + weight_equity
        if abs(total_weight - 1.0) > 1e-8:
            raise ValueError("all weights must sum to 1.")

        return (
            weight_debt * cost_debt * (1 - tax_rate)
            + weight_preferred * cost_preferred
            + weight_common * cost_common
            + weight_equity * cost_equity
        )

    @staticmethod
    def sharpe_ratio(expected_return_i, risk_free_rate, sigma_i):
        if sigma_i <= 0:
            raise ValueError("sigma_i must be greater than 0.")
        return (expected_return_i - risk_free_rate) / sigma_i

    @staticmethod
    def cml_expected_return(
        risk_free_rate, expected_market_return, sigma_market, sigma_portfolio
    ):
        if sigma_market <= 0:
            raise ValueError("sigma_market must be greater than 0.")
        return (
            risk_free_rate
            + ((expected_market_return - risk_free_rate) / sigma_market)
            * sigma_portfolio
        )

    @staticmethod
    def asset_beta_from_equity_beta(debt_value, equity_value, equity_beta):
        total_value = debt_value + equity_value
        if total_value <= 0:
            raise ValueError("debt_value + equity_value must be greater than 0.")
        return (equity_value / total_value) * equity_beta

    @staticmethod
    def equity_beta_from_asset_beta(asset_beta, debt_value, equity_value):
        if equity_value <= 0:
            raise ValueError("equity_value must be greater than 0.")
        return (1 + debt_value / equity_value) * asset_beta
