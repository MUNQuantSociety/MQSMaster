import random
from typing import List


def simple_random_sampling(data: List[float], sample_size: int) -> List[float]:
    """
    Perform simple random sampling from a dataset.

    :param data: The dataset to sample from.
    :param sample_size: The number of samples to draw.
    :return: A list of sampled data points.
    """
    if sample_size < 0:
        raise ValueError("Sample size cannot be negative.")
    if sample_size > len(data):
        raise ValueError("Sample size cannot be greater than the population size.")
    return random.sample(data, sample_size)


def stratified_sampling(
    data: List[float], strata: List[int], sample_size: int
) -> List[float]:
    """
    Perform stratified sampling from a dataset.

    :param data: The dataset to sample from.
    :param strata: A list indicating the strata for each data point.
    :param sample_size: The total number of samples to draw.
    :return: A list of sampled data points.
    """
    if len(data) != len(strata):
        raise ValueError("data and strata must have the same length.")
    if sample_size < 0:
        raise ValueError("sample_size cannot be negative.")
    if sample_size > len(data):
        raise ValueError("Sample size cannot be greater than the population size.")

    stratified_samples = []
    strata_dict = {}

    for i, stratum in enumerate(strata):
        if stratum not in strata_dict:
            strata_dict[stratum] = []
        strata_dict[stratum].append(data[i])

    # Allocate samples proportionally and distribute any remainder.
    sample_sizes = {}
    allocated = 0
    for stratum, values in strata_dict.items():
        stratum_sample_size = int(sample_size * (len(values) / len(data)))
        sample_sizes[stratum] = stratum_sample_size
        allocated += stratum_sample_size

    remainder = sample_size - allocated
    if remainder > 0:
        sorted_strata = sorted(
            strata_dict.items(), key=lambda item: len(item[1]), reverse=True
        )
        for idx in range(remainder):
            sample_sizes[sorted_strata[idx % len(sorted_strata)][0]] += 1

    for stratum, values in strata_dict.items():
        stratified_samples.extend(simple_random_sampling(values, sample_sizes[stratum]))

    return stratified_samples


def systematic_sampling(data: List[float], sample_size: int) -> List[float]:
    """
    Perform systematic sampling from a dataset.

    :param data: The dataset to sample from.
    :param sample_size: The number of samples to draw.
    :return: A list of sampled data points.
    """
    if sample_size < 0:
        raise ValueError("Sample size cannot be negative.")
    if sample_size > len(data):
        raise ValueError("Sample size cannot be greater than the population size.")
    if sample_size == 0:
        return []

    interval = len(data) // sample_size
    return [data[i] for i in range(0, len(data), interval)][:sample_size]
