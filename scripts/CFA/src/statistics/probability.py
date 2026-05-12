from scipy.stats import norm


def normal_distribution(x, mean=0, std_dev=1):
    """
    Calculate the probability density function for a normal distribution.

    :param x: Value for which to calculate the PDF
    :param mean: Mean of the distribution
    :param std_dev: Standard deviation of the distribution
    :return: Probability density function value
    """
    if std_dev <= 0:
        raise ValueError("std_dev must be greater than 0.")
    return norm.pdf(x, mean, std_dev)


def cumulative_normal_distribution(x, mean=0, std_dev=1):
    """
    Calculate the cumulative distribution function for a normal distribution.

    :param x: Value for which to calculate the CDF
    :param mean: Mean of the distribution
    :param std_dev: Standard deviation of the distribution
    :return: Cumulative distribution function value
    """
    if std_dev <= 0:
        raise ValueError("std_dev must be greater than 0.")
    return norm.cdf(x, mean, std_dev)


def binomial_probability(n, k, p):
    """
    Calculate the probability of k successes in n trials for a binomial distribution.

    :param n: Number of trials
    :param k: Number of successes
    :param p: Probability of success on an individual trial
    :return: Probability of k successes
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    if k < 0 or k > n:
        raise ValueError("k must be between 0 and n.")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1.")
    from math import comb

    return comb(n, k) * (p**k) * ((1 - p) ** (n - k))


def expected_value(probabilities, values):
    """
    Calculate the expected value of a random variable.

    :param probabilities: List of probabilities
    :param values: List of corresponding values
    :return: Expected value
    """
    if len(probabilities) != len(values):
        raise ValueError("probabilities and values must have the same length.")
    if any(p < 0 or p > 1 for p in probabilities):
        raise ValueError("all probabilities must be between 0 and 1.")
    return sum(p * v for p, v in zip(probabilities, values))


def variance(probabilities, values):
    """
    Calculate the variance of a random variable.

    :param probabilities: List of probabilities
    :param values: List of corresponding values
    :return: Variance
    """
    if len(probabilities) != len(values):
        raise ValueError("probabilities and values must have the same length.")
    mean = expected_value(probabilities, values)
    return sum(p * (v - mean) ** 2 for p, v in zip(probabilities, values))
