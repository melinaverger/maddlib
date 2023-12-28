import numpy as np


def opt_bandwidth_order(n0, n1):
    """
    Optimal bandwidth (order) for the histogram used in MADD.
    :param n0: number of samples from distribution 0
    :param n1: number of samples from distribution 1
    :return: optimal bandwidth (order)
    """
    return np.cbrt((n0 + n1 + 2 * np.sqrt(n0 * n1)) / (n0 * n1))


def MADD_k(hist0, hist1, k):
    """
    MADD since bin k
    :param hist0: histogram of distribution 0
    :param hist1: histogram of distribution 1
    :param k: number of bins to skip
    :return: MADD since bin k
    """
    sum_abs_diff = np.sum(np.abs(hist0[k:] - hist1[k:]))
    return sum_abs_diff


def split_sizes(total, proportion):
    """
    Split total samples number into two sizes, according to proportion
    :param total: number of all samples
    :param proportion: proportion of samples in the first group
    :return: sizes of the two groups
    """
    n0 = np.floor(total * proportion).astype(int)
    n1 = total - n0
    return n0, n1
