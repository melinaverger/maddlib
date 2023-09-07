import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
    if not (0 < e <= 1):
        raise Exception("The value of argument e should be between ]0, 1].")
    elif abs(e) < 10**(-6):
        # cannot guarantee the results due to numerical approximation
        raise Exception("The value of argument e is too small.")
    
    nb_bins = int(np.floor(1/e))
    density_vector = np.histogram(pred_proba_sfi, bins=nb_bins, range=(0,1), density=False)[0]
    return  density_vector / np.sum(density_vector)


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
    pred_proba_sf0, pred_proba_sf1 = separate_pred_proba(X_test, pred_proba, sf)
    D_G0 = normalized_density_vector(pred_proba_sf0, e)
    D_G1 = normalized_density_vector(pred_proba_sf1, e)
    return np.sum(np.abs(D_G0 - D_G1))


def madd_plot(pred_proba_sf0, pred_proba_sf1, sf, e, model_name):
    """Plots a visual approximation of the MADD.

    Parameters
    ----------
    pred_proba_sf0 : np.ndarray of shape (n, 1)
        The predicted probabilities of positive predictions for group 0
    pred_proba_sf1 : np.ndarray of shape (n, 1)
        The predicted probabilities of positive predictions for group 1
    sf: str
        The name of the binary sensitive feature
    e: float
        The probability sampling parameter
    model_name: str
        The name of the model that outputs the predicted probabilities
    
    Returns
    -------
    Axes
        The Axes of the subplot

    """

    nb_bins = int(np.floor(1/e))

    # Arbitrary choices of colors
    if sf == "gender":
        color_gp1 = "mediumaquamarine"
        color_gp0 = "lightcoral"
    elif sf == "imd_band" or sf == "poverty":
        color_gp1 = "gold"
        color_gp0 = "dimgray"
    elif sf == "disability":
        color_gp1 = "mediumpurple"
        color_gp0 = "lightskyblue"
    elif sf == "age_band" or sf == "age":
        color_gp1 = "salmon"
        color_gp0 = "seagreen"
    else:  # random colors
        color_gp1 = (np.random.random(), np.random.random(), np.random.random())
        color_gp0 = (np.random.random(), np.random.random(), np.random.random())

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.5), constrained_layout=True)  # figsize=(12, 4) for better visualization
    fig.supxlabel("Predicted probabilities  [0 ; 1]", fontsize=16, fontweight='bold')

    # plot D_G0
    ax0 = sns.histplot(ax=axes[0], data=pred_proba_sf1, kde=False, stat="proportion", color=color_gp1, bins=np.linspace(0,1,nb_bins))
    ax0.set_xlim(0, 1)
    ax0.set_ylabel("Density", fontsize=16, fontweight='bold')

    # plot D_G1
    ax1 = sns.histplot(ax=axes[1], data=pred_proba_sf0, kde=False, stat="proportion", color=color_gp0, bins=np.linspace(0,1,nb_bins))
    ax1.set_xlim(0, 1)
    ax1.set_yticklabels([]) # turn off y ticks labels
    ax1.yaxis.set_visible(False)

    # plot the density estimates
    ax2 = sns.kdeplot(ax=axes[2], data=pred_proba_sf1, color=color_gp1, label=sf + ": 1")
    ax2 = sns.kdeplot(ax=axes[2], data=pred_proba_sf0, color=color_gp0, label=sf + ": 0")
    ax2.set_ylabel("Density", fontsize=16, fontweight='bold')
    ax2.set_xlim(0, 1)
    
    plt.legend(bbox_to_anchor = (1.65, 0.5), loc='center right', prop={'weight':'bold'})
    ax1.set_title(f"{model_name}", loc="center", fontsize=16, fontweight='bold')
