from math import exp, factorial

from scipy.stats import binom, norm


def normal_distribution(x, mean, std_dev):
    """
    Calculate the cumulative distribution function for a normal distribution.

    :param x: The value to evaluate
    :param mean: The mean of the distribution
    :param std_dev: The standard deviation of the distribution
    :return: Cumulative probability
    """
    if std_dev <= 0:
        raise ValueError("std_dev must be greater than 0.")
    return norm.cdf(x, mean, std_dev)


def binomial_distribution(n, k, p):
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
    return binom.pmf(k, n, p)


def poisson_distribution(lambd, k):
    """
    Calculate the probability of k events in a fixed interval for a Poisson distribution.

    :param lambd: Average number of events in the interval
    :param k: Actual number of events
    :return: Probability of k events
    """
    if lambd <= 0:
        raise ValueError("lambd must be greater than 0.")
    if k < 0:
        raise ValueError("k must be non-negative.")
    return (lambd**k * exp(-lambd)) / factorial(k)


def uniform_distribution(a, b, x):
    """
    Calculate the probability density function for a uniform distribution.

    :param a: Lower bound of the distribution
    :param b: Upper bound of the distribution
    :param x: Value to evaluate
    :return: Probability density
    """
    if b <= a:
        raise ValueError("b must be greater than a.")
    if a <= x <= b:
        return 1 / (b - a)
    else:
        return 0


def exponential_distribution(lambd, x):
    """
    Calculate the cumulative distribution function for an exponential distribution.

    :param lambd: Rate parameter (1/mean)
    :param x: Value to evaluate
    :return: Cumulative probability
    """
    if lambd <= 0:
        raise ValueError("lambd must be greater than 0.")
    return 1 - exp(-lambd * x) if x >= 0 else 0
