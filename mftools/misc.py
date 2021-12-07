"""Miscellaneous functions.
"""


import numpy as np


def remove_outliers(x, n_std=2):
    r"""Remove outliers, i.e. values, :math:`x_i`, which lie outside the range

    .. math::
        x_i \pm \text{std}(x) * n_{\text{std}},

    where :math:`n_{\text{std}}` denotes the number of standard deviations specified.

    Parameters
    ----------
    x : ndarray
        Input data.
    n_std : int
        Number of standard deviations to retain data within (2, by default).

    Returns
    -------
    ndarray
        Data with outliers removed. Returned as 1D array regardless of input shape.
    """

    if x.ndim > 1:
        x = x.reshape(-1)

    return x[abs(x - np.mean(x)) < n_std * np.std(x)]
