"""
Define some density functions as examples.
Define their inverse cumulative distribution functions, in order to generate the variables that follow the distributions.
"""

import numpy as np
from scipy.stats import norm, gamma, expon
from . import distribution_tools as dt


def func0(x):
    return 2 * x


def func1(x):
    return 2 - 2 * x


def inv_cdf_f0(x):
    return np.sqrt(x)


def inv_cdf_f1(x):
    return 1 - np.sqrt(1 - x)


def func2(x):
    return np.pi / 2 * np.sin(np.pi * x)


def func3(x):
    return 12 * (x - 0.5) ** 2


def inv_cdf_f2(x):
    return np.arccos(1 - 2 * x) / np.pi


def inv_cdf_f3(x):
    return 0.5 * (1 - np.cbrt(1 - 2 * x))


def x_transform_1(x):
    x = np.array(x)
    return 11 * x


def x_transform_2(x):
    x = np.array(x)
    return 6 * (x - 0.5)


func4 = dt.custom_distribution(gamma(a=4, scale=1).pdf, x_transform_1)
func5 = dt.custom_distribution(norm.pdf, x_transform_2)

inv_cdf_f4 = dt.cdf_to_inverse(dt.pdf_to_cdf(func4))
inv_cdf_f5 = dt.cdf_to_inverse(dt.pdf_to_cdf(func5))

func6 = dt.custom_distribution(lambda x: expon.pdf(x, scale=1 / 10), lambda x: x)
func7 = dt.custom_distribution(lambda x: norm.pdf(x, loc=0.8, scale=0.1), lambda x: x)

inv_cdf_f6 = dt.cdf_to_inverse(dt.pdf_to_cdf(func6))
inv_cdf_f7 = dt.cdf_to_inverse(dt.pdf_to_cdf(func7))

func8 = func4
func9 = dt.custom_distribution(norm.pdf, lambda x: 10 * (x - 0.55))

inv_cdf_f8 = dt.cdf_to_inverse(dt.pdf_to_cdf(func8))
inv_cdf_f9 = dt.cdf_to_inverse(dt.pdf_to_cdf(func9))
