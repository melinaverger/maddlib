import numpy as np
import matplotlib.pyplot as plt

def calculate_std(y, start, end):
    """Calculates the standard deviation in the index interval [start, end] of y.
    
    Parameters
    ----------
    y : List or np.ndarray of shape (n, 1)
        The list of data in which to calculate the standard deviation
    start : int
        The initial index
    end : int
        The final index

    Returns
    -------
    float
        The standard deviation
    """
    if type(start) is int and type(end) is int:
        if start > end:
            raise Exception("start should be inferior than end.")
        if end >= len(y):
            raise Exception("end should not be greater than len(y) - 1.")
    else:
        raise Exception("start and end should be integers.")
    
    return np.std(y[start:end+1])


def find_stable_interval(h_list, madd_list, n0=None, n1=None, min_interval_length=None, min_nb_points=50):
    """Finds the bandwidth interval within which the MADD results are stable.
    Calls calculate_std().
    
    Parameters
    ----------
    h_list : List or np.ndarray of shape (n, 1)
        The list of bandwidth values
    madd_list : List or np.ndarray of shape (n, 1)
        The list of MADD results
    n0 : int
        The number of samples in the group 0
    n1 : int
        The number of samples in the group 1
    min_interval_length : float
        The minimum length for the bandwidth interval to consider
    min_nb_points : int
        The minimum number of points to consider in the bandwidth interval
    
    Returns
    -------
    dict
        Dictionary of the results
    """
    # if n0 and n1 are given, min_interval_length is automatically calculated (even if given)
    # otherwise, min_interval_length is set either to the given value or to a default value
    if type(n0) is int and type(n1) is int:
        order = ( (n0**(1/2) + n1**(1/2)) / (n0*n1)**(1/2) )**(2/3)
        min_interval_length = order * (1 - 0.1) / 2
    elif n0 is None or n1 is None:
        if min_interval_length is None:
            min_interval_length = 0.05
    else:
        raise Exception('n0 and n1 arguments must be both None or both int.')

    # initialize the variables
    min_std = float('inf')
    indexes = (0, 0)
    max_h = h_list[-1]

    for i in range(0, len(madd_list)):
        
        x_start = h_list[i]
        
        # if the x_start is too big, we stop the loop
        if x_start > max_h - min_interval_length:
            break
        
        # find the index of the greatest x that is smaller than x_stop
        x_min_sup = x_start + min_interval_length
        index_min_sup = np.searchsorted(h_list, x_min_sup, side='right') - 1
        
        # if the number of points in the interval is too small, we continue the loop
        if index_min_sup - i < min_nb_points:
            continue
        
        # we calculate the std of the interval [i, j]
        for j in range(index_min_sup, len(madd_list)):
            std = calculate_std(madd_list, i, j)
            if std < min_std:
                min_std = std
                indexes = (i, j)

    result = {'min interval length': min_interval_length,
              'indexes': indexes, 
              'h interval': "[{start}, {end}]".format(start=round(h_list[indexes[0]], 3),
                                                      end=round(h_list[indexes[1]+1], 3)), 
              'min madd std': min_std, 
              'madd average': np.average(madd_list[indexes[0] : indexes[1]+1])
             }
    return result


def plot_stable_interval(h_list, madd_list, show_stable=True, indexes=None, show_order=False, n0=None, n1=None, zoom="None", legend=True):
    """Plots MADD results according to the bandwidth.
    
    Parameters
    ----------
    h_list : List or np.ndarray of shape (n, 1)
        The list of bandwidth values
    madd_list : List or np.ndarray of shape (n, 1)
        The list of MADD results
    show_stable : boolean
        To show the stable interval or not
    indexes : tuple
        The indexes (initial, final) of the stable interval (to compute with find_stable_interval())
    show_order : boolean
        To show informational bandwidth values or not
    n0 : int
        The number of samples in the group 0
    n1 : int
        The number of samples in the group 1
    zoom : str or tuple
        To specify some zoom options
    legend : boolean
        To show the legend or not
    
    Returns
    -------
    None
    """
    if show_stable is True:
        if indexes is None:
            raise Exception("indexes argument should be given when show_stable=True.")
    
    if show_order is True:
        if n0 is None or n1 is None:
            raise Exception('n0 and n1 arguments must be given when show_order=True.')
    
    if zoom != "None":
        if n0 is None or n1 is None:
            raise Exception('n0 and n1 arguments must be given when zoom is not "None".')
        elif zoom == "stable":
            if indexes is None:
                raise Exception('indexes argument should be given when zoom is "stable".')

    if type(n0) is int and type(n0) is int:
        order = ( (n0**(1/2) + n1**(1/2)) / (n0*n1)**(1/2) )**(2/3)

    plt.step(h_list, madd_list, where="pre", color="black")
    
    if show_stable:
        plt.axvline(x=h_list[indexes[0]], color="g", linestyle="--")
        plt.axvline(x=h_list[indexes[1]], color="g", linestyle="--")
        plt.axhline(y=np.average(madd_list[indexes[0]:indexes[1]+1]), color="g", linestyle="--", label="MADD average in the stable interval ({0})".format(round(np.average(madd_list[indexes[0] : indexes[1]+1]), 2)))
        if legend is True:
            plt.legend()

    if zoom == "None":
        plt.xlim(0, np.max(h_list))
        if show_order:
            plt.axvline(x= 0.05 * order, color="r", linestyle="--", alpha=0.5, label="h inferior order ({0})".format(round(0.05 * order, 2)))
            plt.axvline(x= 0.1 * order, color="r", linestyle="--", alpha=0.7, label="h middle order ({0})".format(round(0.1 * order, 2)))
            plt.axvline(x= order , color="r", linestyle="--", label="h superior order ({0})".format(round(order, 2)))
            if legend is True:
                plt.legend()
    elif zoom == "sup order":
        plt.xlim(0, order+0.001)
        if show_order:
            plt.axvline(x= 0.05 * order, color="r", linestyle="--", alpha=0.5, label="h inferior order ({0})".format(round(0.05 * order, 2)))
            plt.axvline(x= 0.1 * order, color="r", linestyle="--", alpha=0.7, label="h middle order ({0})".format(round(0.1 * order, 2)))
            plt.axvline(x= order , color="r", linestyle="--", label="h superior order ({0})".format(round(order, 2)))
            if legend is True:
                plt.legend()
    elif zoom == "middle order":
        plt.xlim(0, 0.1 * order+0.001)
        if show_order:
            plt.axvline(x= 0.05 * order, color="r", linestyle="--", alpha=0.5, label="h inferior order ({0})".format(round(0.05 * order, 2)))
            plt.axvline(x= 0.1 * order, color="r", linestyle="--", alpha=0.7, label="h middle order ({0})".format(round(0.1 * order, 2)))
            if legend is True:
                plt.legend()
    elif zoom == "inf order":
        plt.xlim(0, 0.05 * order+0.001)
        if show_order:
            plt.axvline(x= 0.05 * order, color="r", linestyle="--", alpha=0.5, label="h inferior order ({0})".format(round(0.05 * order, 2)))
            if legend is True:
                plt.legend()
    elif zoom == "stable":
        plt.xlim(h_list[indexes[0]]-0.001, h_list[indexes[1]]+0.001)
        if show_order:
            if h_list[indexes[0]] <= 0.05 * order <= h_list[indexes[1]]:
                plt.axvline(x= 0.05 * order, color="r", linestyle="--", alpha=0.5, label="h inferior order ({0})".format(round(0.05 * order, 2)))
            if h_list[indexes[0]] <= 0.1 * order <= h_list[indexes[1]]:
                     plt.axvline(x= 0.1 * order, color="r", linestyle="--", alpha=0.7, label="h middle order ({0})".format(round(0.1 * order, 2)))
            if h_list[indexes[0]] <= order <= h_list[indexes[1]]:
                    plt.axvline(x= order , color="r", linestyle="--", label="h superior order ({0})".format(round(order, 2)))
            if legend is True:
                plt.legend()
    elif type(zoom) is tuple:
        plt.xlim(zoom[0], zoom[1])
        if show_order:
            if zoom[0] <= 0.05 * order <= zoom[1]:
                plt.axvline(x= 0.05 * order, color="r", linestyle="--", alpha=0.5, label="h inferior order ({0})".format(round(0.05 * order, 2)))
            if zoom[0] <= 0.1 * order <= zoom[1]:
                     plt.axvline(x= 0.1 * order, color="r", linestyle="--", alpha=0.7, label="h middle order ({0})".format(round(0.1 * order, 2)))
            if zoom[0] <= order <= zoom[1]:
                    plt.axvline(x= order , color="r", linestyle="--", label="h superior order ({0})".format(round(order, 2)))
            if legend is True:
                plt.legend()
    else:
        raise ValueError('zoom must be "None", "sup order", "middle order", "inf order", "stable" or a tuple.')

    plt.ylim(-0.1, 2)
    plt.xlabel("h")
    plt.ylabel("MADD")
    plt.show()