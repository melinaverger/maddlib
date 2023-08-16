import pandas as pd

def separate_test_sets(X_test, y_test, sf):
    """Separates X_test and y_test according to a specific binary sensitive feature.

    Parameters
    ----------
    X_test : pd.DataFrame
        The test set
    y_test : pd.DataFrame
        The labels of the test set
    sf : str
        The name of the sensitive feature

    Returns
    -------
    four-tuple of pd.DataFrame
        The tuple of (X_test_sf0, X_test_sf1, y_test_sf0, y_test_sf1)
    """
    X_test_sf0 = X_test[X_test[sf] == 0]
    X_test_sf1 = X_test[X_test[sf] == 1]
    y_test_sf0 = y_test.loc[X_test_sf0.index]
    y_test_sf1 = y_test.loc[X_test_sf1.index]
    return (X_test_sf0, X_test_sf1, y_test_sf0, y_test_sf1)
