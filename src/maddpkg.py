import numpy as np
import pandas as pd


def separate_pred_proba(X, pred_proba, sf):
    """Separates the predicted probabilities according to the two groups of a specified binary sensitive feature.

    Parameters
    ----------
    X : pd.DataFrame
        The feature set
    pred_proba : np.ndarray of shape (n, 1)
        The predicted probabilities of positive predictions
    sf : str
        The name of the binary sensitive feature

    Returns
    -------
    couple of np.ndarray
        The couple of predicted probabilities separated (pred_proba_sf0, pred_proba_sf1)
    """
    X_sf0 = X[X[sf] == 0]
    X_sf1 = X[X[sf] == 1]
    pred_proba_sf0 = pred_proba[X_sf0.index]
    pred_proba_sf1 = pred_proba[X_sf1.index]
    return (pred_proba_sf0, pred_proba_sf1)


def normalized_density_vector(pred_proba_sfi, e):
    """Computes the density vector for one group (\D_{G_0} or \D_{G_1}).
    
    Parameters
    ----------
    pred_proba_sfi : np.ndarray of shape (n, 1)
        The predicted probabilities of positive predictions for one group
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

    PP_rounded = np.around(pred_proba_sfi, decimals=nb_decimals)

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
    """Computes the MADD.
    
    Parameters
    ----------
    X_test : pd.DataFrame
        The test set
    pred_proba : np.ndarray of shape (n, 1)
        The predicted probabilities of positive predictions 
    sf: str
        The name of the binary sensitive feature
    e: float
        The probability sampling parameter
    
    Returns
    -------
    np.ndarray
        The density vector
    """
    nb_decimals = (lambda e : len(np.format_float_positional(e).split(".")[1]))(e)
    pred_proba_sf0, pred_proba_sf1 = separate_pred_proba(X_test, pred_proba, sf)
    D_G0 = normalized_density_vector(pred_proba_sf0, e)
    D_G1 = normalized_density_vector(pred_proba_sf1, e)
    return round(np.sum(np.abs(D_G0 - D_G1)), nb_decimals)
