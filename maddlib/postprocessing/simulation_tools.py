import pickle
from . import distribution_tools as dt


def save_variable(variable, filename):
    """
    Save a variable to a file.
    :param variable: variable to be saved
    :param filename: filename
    :return: None
    """
    with open(filename, 'wb') as f:
        pickle.dump(variable, f)


def load_variable(filename):
    """
    Load a variable from a file.
    :param filename: filename
    :return: variable
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def generate_Q(n0, n1, inv_cdf_f0, inv_cdf_f1, seed=None):
    """
    Generate samples from two distributions
    :param n0: number of samples from distribution 0
    :param n1: number of samples from distribution 1
    :param inv_cdf_f0: inverse cumulative distribution function of distribution 0
    :param inv_cdf_f1: inverse cumulative distribution function of distribution 1
    :param seed: random seed
    :return: samples from two distributions
    """
    if seed is None:
        Q0 = dt.generate_samples(inverse_func=inv_cdf_f0, num_samples=n0, seed=seed)
        Q1 = dt.generate_samples(inverse_func=inv_cdf_f1, num_samples=n1, seed=seed)
    else:
        Q0 = dt.generate_samples(inverse_func=inv_cdf_f0, num_samples=n0, seed=seed)
        Q1 = dt.generate_samples(inverse_func=inv_cdf_f1, num_samples=n1, seed=seed + 1)
    return Q0, Q1
