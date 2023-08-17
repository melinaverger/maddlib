import pandas as pd

def separate_sets(X, y, sf):
    """Separates X and y according to a specific binary sensitive feature.

    Parameters
    ----------
    X : pd.DataFrame
        The feature set
    y : pd.DataFrame
        The labels of the feature set X
    sf : str
        The name of the sensitive feature

    Returns
    -------
    four-tuple of pd.DataFrame
        The tuple of (X_sf0, X_sf1, y_sf0, y_sf1)
    """
    X_sf0 = X[X[sf] == 0]
    X_sf1 = X[X[sf] == 1]
    y_sf0 = y.loc[X_sf0.index]
    y_sf1 = y.loc[X_sf1.index]
    return (X_sf0, X_sf1, y_sf0, y_sf1)

def normalized_density_vector():
    pass

def MADD():
    # call separate_sets and normalized_density_vectors
    pass
