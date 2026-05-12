class DescriptiveStatistics:
    """
    A class to perform descriptive statistics calculations.
    """

    @staticmethod
    def mean(data):
        """
        Calculate the mean of a list of numbers.

        :param data: List of numerical values
        :return: Mean of the values
        """
        return sum(data) / len(data) if data else 0

    @staticmethod
    def median(data):
        """
        Calculate the median of a list of numbers.

        :param data: List of numerical values
        :return: Median of the values
        """
        sorted_data = sorted(data)
        n = len(sorted_data)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_data[mid - 1] + sorted_data[mid]) / 2
        else:
            return sorted_data[mid]

    @staticmethod
    def mode(data):
        """
        Calculate the mode of a list of numbers.

        :param data: List of numerical values
        :return: Mode of the values
        """
        from collections import Counter

        if not data:
            return None
        count = Counter(data)
        max_count = max(count.values())
        modes = [k for k, v in count.items() if v == max_count]
        return modes if len(modes) < len(data) else None

    @staticmethod
    def standard_deviation(data, sample=False):
        """
        Calculate the standard deviation of a list of numbers.

        :param data: List of numerical values
        :param sample: Use sample standard deviation when True
        :return: Standard deviation of the values
        """
        if not data:
            return 0
        mean_value = DescriptiveStatistics.mean(data)
        n = len(data)
        denominator = n - 1 if sample else n
        if denominator <= 0:
            raise ValueError(
                "At least two observations are required for sample standard deviation."
            )
        variance = sum((x - mean_value) ** 2 for x in data) / denominator
        return variance**0.5

    @staticmethod
    def variance(data, sample=False):
        """
        Calculate the variance of a list of numbers.

        :param data: List of numerical values
        :param sample: Use sample variance when True
        :return: Variance of the values
        """
        if not data:
            return 0
        mean_value = DescriptiveStatistics.mean(data)
        n = len(data)
        denominator = n - 1 if sample else n
        if denominator <= 0:
            raise ValueError(
                "At least two observations are required for sample variance."
            )
        return sum((x - mean_value) ** 2 for x in data) / denominator
