"""Functions for detemining volume fraction. Each function takes a binary image as input.
"""

import numpy as np


def vol_frac(image):
    """Calculate volume fraction of white space in binary image.

    Parameters
    ----------
    image : ndarray
        Binary image.

    Returns
    -------
    vf : float
        Percentage white space in image.
    """

    vf = 100 * (np.sum(image) / image.size)

    return vf
