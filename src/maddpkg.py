import numpy as np
import pandas as pd


def separate_pred_proba(X, ypp, sf):
    """Separates X and y according to a specific binary sensitive feature.

    Parameters
    ----------
    X : pd.DataFrame
        The feature set
    ypp : np.ndarray
        The predicted probabilities of positive predictions
    sf : str
        The name of the sensitive feature

    Returns
    -------
    couple np.ndarray
        The couple of (y_sf0, y_sf1)
    """
    X_sf0 = X[X[sf] == 0]
    X_sf1 = X[X[sf] == 1]
    y_sf0 = ypp[X_sf0.index]
    y_sf1 = ypp[X_sf1.index]
    return (y_sf0, y_sf1)


def normalized_density_vector(pred_proba, e):
    """Computes the density vector (\D_{G_0} or \D_{G_1}).
    
    Parameters
    ----------
    pred_proba : np.ndarray of shape (n, 1)
        The predicted probabilities of positive predictions from a model
    e : float
        The probability sampling parameter
    
    Returns
    -------
    np.ndarray
        The density vector
    """
    if not (0 < e < 1):
        raise Exception("The value of argument e should be between 0 and 1 excluded.")
    elif abs(e) < 10**(-5):
        # cannot guarantee the results due to numerical approximation
        raise Exception("The value of argument e is too small.")
    
    nb_decimals = (lambda e : len(np.format_float_positional(e).split(".")[1]))(e)
    nb_components = (lambda e : int(1 // e) + 2)(e)

    PP_rounded = np.around(pred_proba, decimals=nb_decimals)

    density_vector = np.zeros(nb_components)
    proba_values = np.linspace(0, 1, nb_components)

    for i in range(len(proba_values)):
        compar = proba_values[i]
        count = 0
        for x in PP_rounded:
            if abs(x - compar) <= e/10:
                count = count + 1
        density_vector[i] = count
    
    normalized_density_vec = density_vector / np.sum(density_vector)

    return normalized_density_vec


def MADD(X_test, pred_proba, sf, e):

    D_G0 = normalized_density_vector()
    D_G1 = normalized_density_vector()
    return np.sum(np.abs(D_G0 - D_G1))
