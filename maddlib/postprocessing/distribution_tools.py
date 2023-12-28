"""
This file contains functions that are used to build / manipulate / show distributions or other related functions.
"""

import numpy as np
from scipy.integrate import cumtrapz, quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


"""
Functions when real distribution is known
"""


def custom_distribution(old_dist, x_transform):
    """
    Create a new distribution by transforming the variable of an existing distribution.
    :param old_dist: existing distribution
    :param x_transform: transformation function
    :return: new distribution
    """
    def new_dist(x):
        x_t = x_transform(x)
        y = old_dist(x_t)
        # Set the PDF to zero outside the [0, 1] interval
        ind = (x >= 0) & (x <= 1)
        y = y * ind
        return y

    area = quad(new_dist, 0, 1)[0]

    # Normalize the distribution so the area under the curve is 1
    def new_dist_norm(x):
        x_t = x_transform(x)
        y = old_dist(x_t) / area
        # Set the PDF to zero outside the [0, 1] interval
        ind = (x >= 0) & (x <= 1)
        y = y * ind
        return y

    return new_dist_norm


def pdf_to_cdf(pdf_func, num_points=1000):
    """
    Convert a PDF function to a CDF function.
    :param pdf_func:
    :param num_points:
    :return:
    """
    x = np.linspace(0, 1, num_points)
    pdf_values = pdf_func(x)
    cdf_values = cumtrapz(pdf_values, x, initial=0)
    # Create interpolation function
    cdf_func = interp1d(x, cdf_values, kind='linear', bounds_error=False, fill_value=(0, 1))
    return np.vectorize(cdf_func)


def cdf_to_inverse(cdf_func, num_points=1000):
    """
    Convert a CDF function to an inverse CDF function.
    :param cdf_func:
    :param num_points:
    :return:
    """
    x = np.linspace(0, 1, num_points)
    cdf_values = cdf_func(x)
    # Create interpolation function
    inverse_func = interp1d(cdf_values, x, kind='linear', bounds_error=False, fill_value=(0, 1))
    return np.vectorize(inverse_func)


def generate_samples(inverse_func, num_samples, seed=None):
    """
    Generate samples from a distribution.
    :param inverse_func: inverse CDF function that generates the distribution
    :param num_samples: number of samples to generate
    :param seed:
    :return:
    """
    np.random.seed(seed)
    # Generate uniform samples
    u = np.random.uniform(0, 1, num_samples)
    np.random.seed(None)
    # Generate samples from the distribution
    return inverse_func(u)


"""
Functions when real distribution is unknown
"""


def histo(samples, bins):
    """
    Compute the histogram of a set of samples.
    :param samples: samples
    :param bins: number of bins
    :return: histogram
    """
    res, _ = np.histogram(samples, bins=bins, range=(0, 1))
    return res / len(samples)  # normalize the histogram


def equal_bin_edges_histogram_to_ecdf_function(hist):
    """
    Convert a histogram (whose width of bin are equal) to the empirical CDF function.
    :param hist: histogram
    :return: empirical CDF function
    """
    cdf_values = np.cumsum(hist)
    n_bins = len(hist)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ecdf_function = interp1d(bin_edges[:-1], cdf_values, kind='linear', bounds_error=False, fill_value=(0, 1))

    return ecdf_function


def integrate_abs_diff(f0, f1, a=0, b=1):
    """
    Integrate the absolute difference between two functions
    :param f0: function 0
    :param f1: function 1
    :param a: lower bound
    :param b: upper bound
    :return: integral of the absolute difference between f0 and f1
    """
    return quad(lambda x: abs(f0(x) - f1(x)), a, b)[0]


"""
Plot functions
"""


def plot_function(f, x_range=(-10, 10)):
    """
    Plot the graph of a function.

    Parameters:
    f (function): The function to plot.
    x_range (tuple): The range of x values to use.

    Returns:
    None
    """
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = f(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f.__name__)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Plot of {f.__name__}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_density(f):
    """
    Plot the density function of a distribution.
    :param f: density function
    :return: None
    """
    x = np.linspace(0, 1, 1000)
    y = f(x)
    plt.plot(x, y)
    plt.show()


def plot_densities(title='Functions plot in [0,1]', **kwargs):
    """
    Plot the density functions of multiple distributions.
    :param title: title of the plot
    :param kwargs: dictionary of functions
    :return: None
    """
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 100)
    for name, func in kwargs.items():
        y = func(x)
        ax.plot(x, y, label=name)
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    plt.show()



def plot_histograms(Q0, Q1, nb_bins, legend_groups, density=False, weight=True,
                    xlable_args=None, ylabel_args=None, title_args=None, legend_args=None,
                    show_title=False):
    """
    Plot the histograms of two distributions.
    :param Q0: distribution 0
    :param Q1: distribution 1
    :param nb_bins: number of bins
    :param legend_groups: 2-tuple of str
    :param density:
    :param weight:
    :param xlable_args: dictionary of arguments for the xlabel
    :param ylabel_args: dictionary of arguments for the ylabel
    :param title_args: dictionary of arguments for the title
    :param legend_args: dictionary of arguments for the legend
    :return: None
    """
    args0 = {'x': Q0, 'bins': nb_bins, 'range': (0, 1), 'alpha': 0.5, 'label': legend_groups[0]}
    args1 = {'x': Q1, 'bins': nb_bins, 'range': (0, 1), 'alpha': 0.5, 'label': legend_groups[1]}
    if density:
        args0['density'] = True
        args1['density'] = True
    if weight:
        weights0 = np.ones_like(Q0) / len(Q0)
        weights1 = np.ones_like(Q1) / len(Q1)
        args0['weights'] = weights0
        args1['weights'] = weights1
    plt.hist(**args0)
    plt.hist(**args1)

    default_xlabel_args = {'xlabel': 'Predicted probability', 'fontsize': 16, 'fontweight': 'bold'}
    default_ylabel_args = {'ylabel': 'Proportion', 'fontsize': 16, 'fontweight': 'bold'}
    default_title_args = {'label': 'Histograms', 'fontsize': 16, 'fontweight': 'bold'}
    default_legend_args = {'fontsize': 12}

    if xlable_args is not None:
        default_xlabel_args = default_xlabel_args | xlable_args
    plt.xlabel(**default_xlabel_args)
    if ylabel_args is not None:
        default_ylabel_args = default_ylabel_args | ylabel_args
    plt.ylabel(**default_ylabel_args)
    if title_args is not None:
        default_title_args = default_title_args | title_args
    if show_title == True:
        plt.title(**default_title_args)
    if legend_args is not None:
        default_legend_args = default_legend_args | legend_args
    plt.legend(**default_legend_args, prop={'weight':'bold'})

    plt.grid(True)
    plt.show()
