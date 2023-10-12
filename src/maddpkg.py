import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import optimal_bandwidth


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


def MADD(h, X_test=None, pred_proba=None, sf=None, pred_proba_sf0=None, pred_proba_sf1=None, min_nb_points=50):
    """Computes the MADD.
    
    Parameters
    ----------
    h : float or str
        The bandwidth (previously called the probability sampling parameter)
    X_test : pd.DataFrame
        The test set
    pred_proba : np.ndarray of shape (n, 1)
        The predicted probabilities of positive predictions (all)
    sf: str
        The name of the binary sensitive feature
    pred_proba_sf0 : numpy.ndarray of shape (n, 1)
        The predicted probabilities of positive predictions of group 0
    pred_proba_sf1 : numpy.ndarray of shape (n, 1)
        The predicted probabilities of positive predictions of group 1
    
    Returns
    -------
    float
        The MADD value
    """
    if (X_test is not None) and (pred_proba is not None):
        if sf is None:
            raise Exception("sf should be given (it sould be the column name of the sensitive feature).")
        else:
            pred_proba_sf0, pred_proba_sf1 = separate_pred_proba(X_test, pred_proba, sf)
    
    if (X_test is None) and (pred_proba is None):
        if (pred_proba_sf0 is None) or (pred_proba_sf1 is None):
            raise Exception("Both preb_proba_sf0 and preb_proba_sf1 should be given.")
    
    if h == "auto":
        Lh = optimal_bandwidth.generate_bandwidths(500)
        Lmadd = [MADD(hi, X_test, pred_proba, sf, pred_proba_sf0, pred_proba_sf1) for hi in Lh]
        interval = optimal_bandwidth.find_stable_interval(Lh, Lmadd, min_nb_points=min_nb_points)
        return interval["madd average"]

    else:
        D_G0 = normalized_density_vector(pred_proba_sf0, h)
        D_G1 = normalized_density_vector(pred_proba_sf1, h)
        return np.sum(np.abs(D_G0 - D_G1))


def madd_plot(h, pred_proba_sf0, pred_proba_sf1, legend_groups, title, figsize=(12, 4)):
    """Plots a visual approximation of the MADD.

    Parameters
    ----------
    h : float
        The bandwidth (previously called the probability sampling parameter)
    pred_proba_sf0 : np.ndarray of shape (n, 1)
        The predicted probabilities of positive predictions of group 0
    pred_proba_sf1 : np.ndarray of shape (n, 1)
        The predicted probabilities of positive predictions of group 1
    legend_groups: str or 2-tuple
        The name of the binary sensitive feature or the names of the two groups in a 2-tuple
    title: str
        The title of the graph (it could be the name of the model that outputs the predicted probabilities)
    
    Returns
    -------
    None
    """

    nb_bins = int(np.floor(1/h))

    # Arbitrary choices of colors
    if legend_groups == "gender":
        color_gp1 = "mediumaquamarine"
        color_gp0 = "lightcoral"
    elif legend_groups == "imd_band" or legend_groups == "poverty":
        color_gp1 = "gold"
        color_gp0 = "dimgray"
    elif legend_groups == "disability":
        color_gp1 = "mediumpurple"
        color_gp0 = "lightskyblue"
    elif legend_groups == "age_band" or legend_groups == "age":
        color_gp1 = "salmon"
        color_gp0 = "seagreen"
    else:  # random colors
        color_gp1 = (np.random.random(), np.random.random(), np.random.random())
        color_gp0 = (np.random.random(), np.random.random(), np.random.random())

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    fig.supxlabel("Predicted probabilities  [0 ; 1]", fontsize=16, fontweight='bold')

    # plot D_G0
    ax0 = sns.histplot(ax=axes[0], data=pred_proba_sf1, kde=False, stat="proportion", color=color_gp1, bins=np.linspace(0,1,nb_bins))
    ax0.set_xlim(0, 1)
    ax0.set_ylabel("Proportion", fontsize=16, fontweight='bold')

    # plot D_G1
    ax1 = sns.histplot(ax=axes[1], data=pred_proba_sf0, kde=False, stat="proportion", color=color_gp0, bins=np.linspace(0,1,nb_bins))
    ax1.set_xlim(0, 1)
    ax1.set_yticklabels([]) # turn off y ticks labels
    ax1.yaxis.set_visible(False)

    # plot the density estimates
    if type(legend_groups) is str:
        ax2 = sns.kdeplot(ax=axes[2], data=pred_proba_sf1, color=color_gp1, label=legend_groups + ": 1")
        ax2 = sns.kdeplot(ax=axes[2], data=pred_proba_sf0, color=color_gp0, label=legend_groups + ": 0")
    elif (type(legend_groups) is tuple) and (len(legend_groups) == 2):
        ax2 = sns.kdeplot(ax=axes[2], data=pred_proba_sf1, color=color_gp1, label=legend_groups[1])
        ax2 = sns.kdeplot(ax=axes[2], data=pred_proba_sf0, color=color_gp0, label=legend_groups[0])
    ax2.set_ylabel("Density", fontsize=16, fontweight='bold')
    ax2.set_xlim(0, 1)
    
    plt.legend(bbox_to_anchor = (0.8, 1.25), loc='upper right', prop={'weight':'bold'})
    ax1.set_title(f"{title}", loc="center", fontsize=16, fontweight='bold', y=1.1)
