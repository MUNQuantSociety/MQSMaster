class HypothesisTesting:
    """
    A class to perform hypothesis testing methods.
    """

    @staticmethod
    def t_test(sample1, sample2, alpha=0.05):
        """
        Perform a two-sample t-test.

        :param sample1: First sample data
        :param sample2: Second sample data
        :param alpha: Significance level
        :return: t-statistic, p-value, and conclusion
        """
        from scipy import stats

        t_stat, p_value = stats.ttest_ind(sample1, sample2)
        conclusion = "Reject the null hypothesis" if p_value < alpha else "Fail to reject the null hypothesis"
        return t_stat, p_value, conclusion

    @staticmethod
    def z_test(sample_mean, population_mean, population_std, n, alpha=0.05):
        """
        Perform a one-sample z-test.

        :param sample_mean: Mean of the sample
        :param population_mean: Mean of the population
        :param population_std: Standard deviation of the population
        :param n: Sample size
        :param alpha: Significance level
        :return: z-statistic, p-value, and conclusion
        """
        from scipy import stats
        import numpy as np

        z_stat = (sample_mean - population_mean) / (population_std / np.sqrt(n))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        conclusion = "Reject the null hypothesis" if p_value < alpha else "Fail to reject the null hypothesis"
        return z_stat, p_value, conclusion

    @staticmethod
    def chi_square_test(observed, expected, alpha=0.05):
        """
        Perform a chi-square test.

        :param observed: Observed frequencies
        :param expected: Expected frequencies
        :param alpha: Significance level
        :return: chi-square statistic, p-value, and conclusion
        """
        from scipy import stats

        chi2_stat, p_value = stats.chisquare(observed, expected)
        conclusion = "Reject the null hypothesis" if p_value < alpha else "Fail to reject the null hypothesis"
        return chi2_stat, p_value, conclusion

    @staticmethod
    def anova_test(*samples, alpha=0.05):
        """
        Perform a one-way ANOVA test.

        :param samples: Multiple sample data
        :param alpha: Significance level
        :return: F-statistic, p-value, and conclusion
        """
        from scipy import stats

        f_stat, p_value = stats.f_oneway(*samples)
        conclusion = "Reject the null hypothesis" if p_value < alpha else "Fail to reject the null hypothesis"
        return f_stat, p_value, conclusion