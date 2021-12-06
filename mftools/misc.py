"""Miscellaneous functions.
"""


import numpy as np


def remove_outliers(x, n_std=2):
    """Remove outliers, i.e. values, x_i, which lie outside the range :math:`x_i +- std(x) * n_std`.

    Parameters
    ----------
    x : ndarray
        Input data.
    n_std : int
        Number of standard deviations to retain data within (2, by default).

    Returns
    -------
    ndarray
        Data with outliers removed. Returned as 1d array regardless of input shape.
    """

    if x.ndim > 1:
        x = x.reshape(-1)

    return x[abs(x - np.mean(x)) < n_std * np.std(x)]
