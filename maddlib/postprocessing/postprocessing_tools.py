import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import concurrent.futures
from tqdm import tqdm
import plotly.graph_objects as go

from . import distribution_tools as dt
from . import MADD_tools as mt
from . import simulation_tools as st


"""
Basic functions
"""


def cdf_lambda(cdf0, cdf1, lambda_):
    """
    Calculate the cdf of the mixture distribution of lambda_ * cdf0 + (1 - lambda_) * cdf1
    :param cdf0: cumulative distribution function of one distribution (not necessarily distribution 0)
    :param cdf1: cumulative distribution function of another distribution (not necessarily distribution 1)
    :param lambda_: proportion of distribution 0 in the mixture distribution
    :return: cdf of the mixture distribution
    """

    def cdf(y_pred):
        return (1 - lambda_) * cdf0(y_pred) + lambda_ * cdf1(y_pred)

    return np.vectorize(cdf)


def predict_lambda(T0, T1, cdf0, cdf1, cdff, lambda_):
    """
    Predict the label of the data points using the mixture distributions of
    lambda_ * cdf0 + (1 - lambda_) * cdff and lambda_ * cdf1 + (1 - lambda_) * cdff
    :param T0: F_0(Q0)
    :param T1: F_1(Q1)
    :param cdf0: cumulative distribution function of distribution 0
    :param cdf1: cumulative distribution function of distribution 1
    :param cdff: cumulative distribution function of global distribution
    :param lambda_: proportion of distribution 0/1 in the mixture distribution
    :return: predicted probabilities by the mixture distributions
    """
    # calculate the inverse cdf of the mixture distributions
    cdf0_l = cdf_lambda(cdf0, cdff, lambda_)
    inv_cdf0_l = dt.cdf_to_inverse(cdf0_l)

    cdf1_l = cdf_lambda(cdf1, cdff, lambda_)
    inv_cdf1_l = dt.cdf_to_inverse(cdf1_l)

    # predict the labels by the mixture distributions with the formula F_new^{-1}(F_old(Q0))
    y0_pred = inv_cdf0_l(T0)
    y1_pred = inv_cdf1_l(T1)
    y_pred = np.concatenate([y0_pred, y1_pred])

    return y_pred


"""
Accuracy loss functions
"""


def binary_cross_entropy(y_true, y_pred):
    """
    Calculate the binary cross entropy loss
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: binary cross entropy loss
    """
    # make sure y_pred is not 0 or 1
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # calculate the loss
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # return the mean loss
    return np.mean(loss)


def binary_cross_entropy_lambda(y_true, T0, T1, cdf0, cdf1, cdff, lambda_):
    """
    Calculate the binary cross entropy loss of the mixture distributions
    :param y_true: true labels
    :param T0: F_0(Q0)
    :param T1: F_1(Q1)
    :param cdf0: cumulative distribution function of distribution 0
    :param cdf1: cumulative distribution function of distribution 1
    :param cdff: cumulative distribution function of global distribution
    :param lambda_: proportion of distribution 0/1 in the mixture distribution
    :return: binary cross entropy loss of the mixture distributions
    """
    y_pred = predict_lambda(T0, T1, cdf0, cdf1, cdff, lambda_)
    return binary_cross_entropy(y_true, y_pred)


def quadratic_loss(y_true, y_pred):
    """
    Calculate the quadratic loss
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: quadratic loss
    """
    return np.mean((y_true - y_pred) ** 2)


def quadratic_loss_lambda(y_true, T0, T1, cdf0, cdf1, cdff, lambda_):
    """
    Calculate the quadratic loss of the mixture distributions
    :param y_true: true labels
    :param T0: F_0(Q0)
    :param T1: F_1(Q1)
    :param cdf0: cumulative distribution function of distribution 0
    :param cdf1: cumulative distribution function of distribution 1
    :param cdff: cumulative distribution function of global distribution
    :param lambda_: proportion of distribution 0/1 in the mixture distribution
    :return: quadratic loss of the mixture distributions
    """
    y_pred = predict_lambda(T0, T1, cdf0, cdf1, cdff, lambda_)
    return quadratic_loss(y_true, y_pred)


def accuracy_percentage_loss(y_true, y_pred):
    """
    Calculate the accuracy percentage loss
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: accuracy percentage loss
    """
    return np.mean(y_true != y_pred)


def accuracy_percentage_loss_lambda(y_true, T0, T1, cdf0, cdf1, cdff, seuil, lambda_):
    """
    Calculate the accuracy percentage loss of the mixture distributions
    :param y_true: true labels
    :param T0: F_0(Q0)
    :param T1: F_1(Q1)
    :param cdf0: cumulative distribution function of distribution 0
    :param cdf1: cumulative distribution function of distribution 1
    :param cdff: cumulative distribution function of global distribution
    :param seuil: threshold to predict the labels
    :param lambda_: proportion of distribution 0/1 in the mixture distribution
    :return: accuracy percentage loss of the mixture distributions
    """
    y_pred = predict_lambda(T0, T1, cdf0, cdf1, cdff, lambda_) >= seuil
    return accuracy_percentage_loss(y_true, y_pred)


"""
MADD loss functions
"""


def real_MADD_lambda(T0, T1, cdf0, cdf1, cdff, nb_bins, lambda_):
    """
    Calculate the MADD of the mixture distributions
    :param T0: F_0(Q0)
    :param T1: F_1(Q1)
    :param cdf0: cumulative distribution function of distribution 0
    :param cdf1: cumulative distribution function of distribution 1
    :param cdff: cumulative distribution function of global distribution
    :param nb_bins: number of bins to calculate the MADD
    :param lambda_: proportion of distribution 0/1 in the mixture distribution
    :return: MADD of the mixture distributions
    """
    y_pred = predict_lambda(T0, T1, cdf0, cdf1, cdff, lambda_)
    n0 = len(T0)
    new_Q0 = y_pred[:n0]
    new_Q1 = y_pred[n0:]
    return mt.MADD_k(dt.histo(new_Q0, nb_bins), dt.histo(new_Q1, nb_bins), k=0)


def theoretical_MADD_lambda(init_madd, lambda_):
    """
    Calculate the theoretical MADD of the mixture distributions, which the real MADD will approach when the number of
    samples and the number of bins go to infinity
    :param init_madd: MADD when lambda = 0
    :param lambda_: proportion of distribution 0/1 in the mixture distribution
    :return: theoretical MADD of the mixture distributions
    """
    return (1 - lambda_) * init_madd


"""
Key functions
"""


def object_func(accu_loss_settings, madd_settings, global_settings, lambda_, theta_):
    """
    Calculate the objective function, which is a weighted sum of the loss function and the MADD, potentially with
    a normalization factor
    :param accu_loss_settings: settings of the loss function
    :param madd_settings: settings of the MADD
    :param global_settings: settings of the global distribution
    :param lambda_: proportion of distribution 0/1 in the mixture distribution
    :param theta_: proportion of importance of MADD in the objective function
    :return: total loss
    """
    lambda_ = np.asarray(lambda_)
    theta_ = np.asarray(theta_)

    # print(accu_loss_settings['params'])
    accu_loss = accu_loss_settings['func'](**(accu_loss_settings['params'] | {'lambda_': lambda_}))
    madd = madd_settings['func'](**(madd_settings['params'] | {'lambda_': lambda_}))

    if not global_settings['has_threshold']:
        norm_fact = global_settings['norm_fact']
        total_loss = (1 - theta_) * norm_fact * accu_loss + theta_ * madd
    else:
        total_loss = (1 - theta_) * accu_loss + theta_ * 0.5 * madd

    return total_loss, accu_loss, madd


def post_processing_precomputed(Q0, Q1, nb_bins):
    """
    Pre-compute the data used in the post-processing
    :param Q0: samples from distribution 0
    :param Q1: samples from distribution 1
    :param nb_bins: number of bins used to compute the histograms
    :return: pre-computed data
    """
    n0 = len(Q0)
    n1 = len(Q1)
    hist_Q0 = dt.histo(Q0, nb_bins)
    hist_Q1 = dt.histo(Q1, nb_bins)
    hist_Q = (n0 * hist_Q0 + n1 * hist_Q1) / (n0 + n1)

    cdf_Q0 = dt.equal_bin_edges_histogram_to_ecdf_function(hist_Q0)
    cdf_Q1 = dt.equal_bin_edges_histogram_to_ecdf_function(hist_Q1)
    cdf_Q = dt.equal_bin_edges_histogram_to_ecdf_function(hist_Q)

    T0 = cdf_Q0(Q0)
    T1 = cdf_Q1(Q1)

    # MADD when lambda = 0
    init_madd = mt.MADD_k(hist_Q0, hist_Q1, 0)

    precomputed_data = {
        'n0': n0,
        'n1': n1,
        'hist_Q0': hist_Q0,
        'hist_Q1': hist_Q1,
        'hist_Q': hist_Q,
        'cdf0': cdf_Q0,
        'cdf1': cdf_Q1,
        'cdff': cdf_Q,
        'T0': T0,
        'T1': T1,
        'init_madd': init_madd
    }

    return precomputed_data


def post_processing_settings(precomputed_data, y_true, nb_bins, loss_type, real_madd, auto_rescale, seuil=None):
    """
    Settings of the loss, accuracy and global functions
    :param precomputed_data: precomputed data
    :param y_true: true labels
    :param nb_bins: number of bins used to compute the histograms
    :param loss_type: loss function type
    :param real_madd: boolean, whether to use the real MADD or not
    :param auto_rescale: boolean, whether to rescale the loss function and MADD or not
    :param seuil: threshold used to predict the labels
    :return: settings of the loss, accuracy and global functions
    """
    n0, n1, hist_Q0, hist_Q1, hist_Q, cdf_Q0, cdf_Q1, cdf_Q, T0, T1, init_madd = precomputed_data.values()

    if loss_type == 'quadratic':
        loss_func = quadratic_loss_lambda
    elif loss_type == 'binary_cross_entropy':
        loss_func = binary_cross_entropy_lambda
    elif loss_type == 'accuracy_percentage':
        loss_func = accuracy_percentage_loss_lambda
    else:
        raise ValueError('loss_type must be quadratic, binary_cross_entropy or accuracy_percentage')

    accu_loss_settings = {
        'func': loss_func,
        'params': {
            'y_true': y_true,
            'T0': T0,
            'T1': T1,
            'cdf0': cdf_Q0,
            'cdf1': cdf_Q1,
            'cdff': cdf_Q,
        }
    }

    if seuil is None:
        global_settings = {'has_threshold': False}
    else:
        accu_loss_settings['params']['seuil'] = seuil
        global_settings = {'has_threshold': True}

    if real_madd:
        madd_settings = {
            'func': real_MADD_lambda,
            'params': {
                'T0': T0,
                'T1': T1,
                'cdf0': cdf_Q0,
                'cdf1': cdf_Q1,
                'cdff': cdf_Q,
                'nb_bins': nb_bins,
            }
        }
    else:
        madd_settings = {
            'func': theoretical_MADD_lambda,
            'params': {
                'init_madd': init_madd
            }
        }

    if auto_rescale:
        loss_lambda_1 = accu_loss_settings['func'](**(accu_loss_settings['params'] | {'lambda_': 1}))
        global_settings['norm_fact'] = init_madd / loss_lambda_1
    else:
        global_settings['norm_fact'] = 1

    return accu_loss_settings, madd_settings, global_settings


def calculate_losses(f, theta_is_fixed=False):
    """
    Compute the object function values on a grid of lambda and theta values
    :param f: object function
    :param theta_is_fixed:
    :return: Lambda, Theta, Z
    """
    if theta_is_fixed:
        lambda_values = np.linspace(0, 1, 1000)

        # Create a progress pool executor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(f, lambda_=lambda_): lambda_ for lambda_ in lambda_values}
            # Create a progress bar
            progress = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Calculating", ncols=75)
            # Collect the results
            total_loss = []
            accu_loss = []
            madd = []
            lambdas = []
            for future in progress:
                lambdas.append(futures[future])
                a, b, c = future.result()
                total_loss.append(a)
                accu_loss.append(b)
                madd.append(c)
            order = np.argsort(lambdas)
            total_loss = np.array(total_loss)[order]
            accu_loss = np.array(accu_loss)[order]
            madd = np.array(madd)[order]

            return lambda_values, total_loss, accu_loss, madd
    else:
        lambda_values = np.linspace(0, 1, 100)
        theta_values = np.linspace(0, 1, 100)

        Lambda, Theta = np.meshgrid(lambda_values, theta_values)
        total_loss = np.empty_like(Lambda)
        accu_loss = np.empty_like(Lambda)
        madd = np.empty_like(Lambda)

        # Create a progress pool executor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # for each (i, j) coordinate, start a new process to compute the function value
            futures = {executor.submit(f, lambda_=Lambda[i, j], theta_=Theta[i, j]): (i, j) for i in
                       range(Lambda.shape[0])
                       for j in range(Lambda.shape[1])}

            # Create a progress bar
            progress = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Calculating", ncols=75)

            # Collect the results
            for future in progress:
                i, j = futures[future]
                total_loss[i, j], accu_loss[i, j], madd[i, j] = future.result()

        return Lambda, Theta, total_loss, accu_loss, madd


def post_processing(Q0, Q1, y_true, nb_bins, loss_type, real_madd, auto_rescale, seuil=None):
    """
    Post-processing of the results Q0 and Q1
    :param Q0: samples from distribution 0
    :param Q1: samples from distribution 1
    :param y_true: true labels
    :param nb_bins: number of bins used to compute the histograms
    :param loss_type: loss function type
    :param auto_rescale: boolean, whether to rescale the loss function and MADD or not
    :param seuil: threshold used to predict the labels
    :return: coordinates of total loss function to plot, accuracy losses when lambda = 0 and 1, MADD when lambda = 0
    """
    precomputed_data = post_processing_precomputed(Q0, Q1, nb_bins)
    accu_loss_settings, madd_settings, global_settings \
        = post_processing_settings(precomputed_data, y_true, nb_bins, loss_type, real_madd, auto_rescale, seuil)

    loss_lambda_0 = accu_loss_settings['func'](**(accu_loss_settings['params'] | {'lambda_': 0}))
    loss_lambda_1 = accu_loss_settings['func'](**(accu_loss_settings['params'] | {'lambda_': 1}))

    obj_partial = partial(object_func, accu_loss_settings=accu_loss_settings, madd_settings=madd_settings,
                          global_settings=global_settings)

    Lambda, Theta, total_loss, accu_loss, madd = calculate_losses(obj_partial)
    data = precomputed_data | {'loss_lambda_0': loss_lambda_0, 'loss_lambda_1': loss_lambda_1,
                               'accu_loss': accu_loss, 'madd': madd}
    return Lambda, Theta, total_loss, data


def post_processing_fixed_theta(Q0, Q1, y_true, nb_bins, theta, loss_type, real_madd, auto_rescale, seuil=None):
    precomputed_data = post_processing_precomputed(Q0, Q1, nb_bins)
    accu_loss_settings, madd_settings, global_settings \
        = post_processing_settings(precomputed_data, y_true, nb_bins, loss_type, real_madd, auto_rescale, seuil)

    loss_lambda_0 = accu_loss_settings['func'](**(accu_loss_settings['params'] | {'lambda_': 0}))
    loss_lambda_1 = accu_loss_settings['func'](**(accu_loss_settings['params'] | {'lambda_': 1}))

    obj_partial = partial(object_func, accu_loss_settings=accu_loss_settings, madd_settings=madd_settings,
                          global_settings=global_settings, theta_=theta)

    lambda_values, total_loss, accu_loss, madd = calculate_losses(obj_partial, theta_is_fixed=True)

    data = precomputed_data | {'loss_lambda_0': loss_lambda_0, 'loss_lambda_1': loss_lambda_1,
                               'lambda_values': lambda_values, 'total_loss': total_loss, 'accu_loss': accu_loss,
                               'madd': madd}
    return data


"""
Simulation functions
"""


def simulation_precomputed(n0, n1, inv_cdf_0, inv_cdf_1, loss_type, start_seed, seuil=None):
    Q0 = dt.generate_samples(inv_cdf_0, n0, seed=start_seed)
    Q1 = dt.generate_samples(inv_cdf_1, n1, seed=start_seed + 1)

    if seuil is None:
        if loss_type == 'quadratic':
            y0_true = Q0
            y1_true = Q1
        elif loss_type == 'binary_cross_entropy':
            np.random.seed(start_seed + 2)
            y0_true = np.random.binomial(1, Q0)
            np.random.seed(start_seed + 3)
            y1_true = np.random.binomial(1, Q1)
        else:
            raise ValueError('loss_type must be quadratic or binary_cross_entropy')
    else:
        if loss_type == 'accuracy_percentage':
            np.random.seed(start_seed + 2)
            y0_true = np.random.binomial(1, Q0)
            np.random.seed(start_seed + 3)
            y1_true = np.random.binomial(1, Q1)
            # y0_true = np.asarray(Q0 >= seuil, dtype=int)
            # y1_true = np.asarray(Q1 >= seuil, dtype=int)
        else:
            raise ValueError('loss_type must be accuracy_percentage')
    np.random.seed(None)
    y_true = np.concatenate([y0_true, y1_true])
    return Q0, Q1, y_true


def simulation(n0, n1, inv_cdf_0, inv_cdf_1, nb_bins, loss_type, auto_rescale, real_madd, start_seed, seuil=None):
    """
    Simulate the data and do the post-processing
    :param n0: number of samples from distribution 0
    :param n1: number of samples from distribution 1
    :param inv_cdf_0: inverse cumulative distribution function of distribution 0
    :param inv_cdf_1: inverse cumulative distribution function of distribution 1
    :param nb_bins: nb_bins: number of bins used to compute the histograms
    :param loss_type: loss function type
    :param auto_rescale: boolean, whether to rescale the loss function and MADD or not
    :param start_seed: beginning seed
    :param seuil: threshold used to predict the labels
    :return: results of the post-processing
    """
    Q0, Q1, y_true = simulation_precomputed(n0, n1, inv_cdf_0, inv_cdf_1, loss_type, start_seed, seuil)

    return post_processing(Q0, Q1, y_true, nb_bins, loss_type, real_madd, auto_rescale, seuil)


def simulation_fixed_theta(n0, n1, inv_cdf_0, inv_cdf_1, nb_bins, theta, loss_type, auto_rescale, real_madd, start_seed,
                           seuil=None):
    Q0, Q1, y_true = simulation_precomputed(n0, n1, inv_cdf_0, inv_cdf_1, loss_type, start_seed, seuil)

    return post_processing_fixed_theta(Q0, Q1, y_true, nb_bins, theta, loss_type, real_madd, auto_rescale, seuil)


"""
Application and verification functions
"""


def fair_improved_predicton(data, lambda_):
    """
    Improved prediction function for the fair classifier
    :param Q0: samples from distribution 0
    :param Q1: samples from distribution 1
    :param nb_bins: number of bins used to compute the histograms
    :param lambda_: lambda parameter
    :return: new prediction
    """
    T0, T1, cdf_Q0, cdf_Q1, cdf_Q, madd = data['T0'], data['T1'], data['cdf0'], data['cdf1'], data['cdff'], data['madd']

    return predict_lambda(T0, T1, cdf_Q0, cdf_Q1, cdf_Q, lambda_)


def verify_application(data, Q0_second, Q1_second, y_true_second, nb_bins, theta, loss_type, real_madd, auto_rescale,
                       seuil=None):
    precomputed_data = post_processing_precomputed(Q0_second, Q1_second, nb_bins)
    cdf_Q0, cdf_Q1, cdf_Q = data['cdf0'], data['cdf1'], data['cdff']
    precomputed_data['cdf0'], precomputed_data['cdf1'], precomputed_data['cdff'] = cdf_Q0, cdf_Q1, cdf_Q
    accu_loss_settings, madd_settings, global_settings \
        = post_processing_settings(precomputed_data, y_true_second, nb_bins, loss_type, real_madd, auto_rescale, seuil)

    loss_lambda_0 = accu_loss_settings['func'](**(accu_loss_settings['params'] | {'lambda_': 0}))
    loss_lambda_1 = accu_loss_settings['func'](**(accu_loss_settings['params'] | {'lambda_': 1}))

    obj_partial = partial(object_func, accu_loss_settings=accu_loss_settings, madd_settings=madd_settings,
                          global_settings=global_settings, theta_=theta)

    lambda_values, total_loss, accu_loss, madd = calculate_losses(obj_partial, theta_is_fixed=True)

    data = precomputed_data | {'loss_lambda_0': loss_lambda_0, 'loss_lambda_1': loss_lambda_1,
                               'lambda_values': lambda_values, 'total_loss': total_loss, 'accu_loss': accu_loss,
                               'madd': madd}
    return data


"""
Theoretical proof functions
"""


def MADD_convergence_test(f0, f1, list_nb_samples):
    real_inv_cdf_0 = dt.cdf_to_inverse(dt.pdf_to_cdf(f0))
    real_inv_cdf_1 = dt.cdf_to_inverse(dt.pdf_to_cdf(f1))
    curent_seed = 0

    result = []
    lambda_values = np.linspace(0, 1, 1000)
    for nb_samples in list_nb_samples:
        n0, n1 = mt.split_sizes(nb_samples, 0.5)
        Q0, Q1 = st.generate_Q(n0, n1, real_inv_cdf_0, real_inv_cdf_1, curent_seed)
        curent_seed += 1
        nb_bins = np.floor(1 / (0.1 * mt.opt_bandwidth_order(n0, n1))).astype(int)
        precomputed_data = post_processing_precomputed(Q0, Q1, nb_bins)
        cdf_Q0, cdf_Q1, cdf_Q, T0, T1, init_madd = \
            precomputed_data['cdf0'], precomputed_data['cdf1'], precomputed_data['cdff'], precomputed_data['T0'], \
                precomputed_data['T1'], precomputed_data['init_madd']

        madds_real = []

        partial_madd = partial(real_MADD_lambda, T0=T0, T1=T1, cdf0=cdf_Q0, cdf1=cdf_Q1, cdff=cdf_Q, nb_bins=nb_bins)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(partial_madd, lambda_=lambda_): lambda_ for lambda_ in lambda_values}
            # Create a progress bar
            progress = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Calculating", ncols=75)
            # Collect the results
            lambdas = []
            for future in progress:
                lambdas.append(futures[future])
                madds_real.append(future.result())
            order = np.argsort(lambdas)
            madds_real = np.array(madds_real)[order]

        result.append(madds_real)
    return result


"""
Plot function
"""


def plot_func_3D(Theta, Lambda, Z):
    """
    Plot the 3D object function
    :param Theta:
    :param Lambda:
    :param Z:
    :return:
    """
    surface = go.Surface(
        x=Theta,
        y=Lambda,
        z=Z,
        colorscale='Viridis',
        opacity=0.8,
        contours={
            # Show height contours for z axis
            "x": {"show": True, "start": 0, "end": 1, "size": 0.2, "color": "white"}
        }
    )
    fig = go.Figure(data=[surface])
    fig.update_layout(scene=dict(xaxis_title='theta', yaxis_title='lambda', zaxis_title='total loss'),
                      title='3D loss function')
    return fig


def plot_func_2D(Lambda, total_loss, accu_loss, madd,
                 ylim=None, xlabel_args=None, ylabel_args=None, title_args=None, legend_args=None,
                 show_title=False):
    """
    Plot the 2D object functions
    :param Lambda:
    :param total_loss:
    :param accu_loss:
    :param madd:
    :param ylim:
    :param xlabel_args: dictionary of arguments for the xlabel
    :param ylabel_args: dictionary of arguments for the ylabel
    :param title_args: dictionary of arguments for the title
    :param legend_args: dictionary of arguments for the legend
    :return:
    """
    plt.plot(Lambda, total_loss, label='Objective function')
    plt.plot(Lambda, accu_loss, label='Accuracy loss')
    plt.plot(Lambda, madd, label='Fairness loss')
    if ylim is not None:
        plt.ylim(ylim)

    default_xlabel_args = {'xlabel': 'lambda', 'fontsize': 16, 'fontweight': 'bold'}
    default_ylabel_args = {'ylabel': 'loss', 'fontsize': 16, 'fontweight': 'bold'}
    default_title_args = {'label': 'Effect of lambda on \nthe objective function, accuracy loss and fairness loss', 'fontsize': 16, 'fontweight': 'bold'}
    default_legend_args = {'fontsize': 12}

    if xlabel_args is not None:
        default_xlabel_args = default_xlabel_args | xlabel_args
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
    legend = plt.legend(**default_legend_args)
    for text in legend.get_texts():
        text.set_fontweight('bold')

    plt.show()
