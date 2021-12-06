"""Methods for binarising grayscale images.
"""

import numpy as np
from skimage.filters import threshold_niblack, threshold_sauvola, threshold_otsu
from sklearn.cluster import KMeans


def niblack(image, window_size=15, k=0.2):
    """Return binary image such that all pixels with a value > :math:`T` are set to 1 and otherwise set to 0, where
    :math:`T` is given by

    .. math::
        T = \\mu(x,y) - k * \\sigma(x,y),

    where :math:`\\mu(x,y)` and :math`\\sigma(x,y)` are the mean and standard deviation, respectively, of the square
    neighbourhood with side length window_size, centred at pixel :math:`(x,y)` and :math:`k` is a tunable scale factor.

    Parameters
    ----------

    image : ndarray
        Input image. Must be grayscale.
    window_size : int, optional
        Side length of sliding square neighbourhood. Must be odd (15, by default).
    k : float, optional
        Scale factor for standard deviation in Niblack formula (0.2, by default).

    Returns
    -------
    image_binary : ndarray
        Binarised image.
    """

    T = threshold_niblack(image, window_size, k)
    image_binary = image > T

    return image_binary


def sauvola(image, window_size=15, k=0.2, R=None):
    """Return binary image such that all pixels with a value > :math:`T` are set to 1 and otherwise set to 0, where
    :math:`T` is given by

    .. math::
        T = \\mu(x,y)*(1 + k * ((\\sigma(x,y)/R) - 1)),

    where :math:`\\mu(x,y)` and :math:`\\sigma(x,y)` are the mean and standard deviation, respectively, of the square
    neighbourhood with side length window_size, centred at pixel :math:`(x,y)`, :math:`k` is a tunable scale factor and
    :math:`R` is the dynamic range of standard deviation.

    Parameters
    ----------

    image : ndarray
        Input image. Must be grayscale.
    window_size : int, optional
        Side length of sliding square neighbourhood. Must be odd (15, by default).
    k : float, optional
        Scale factor for standard deviation in Sauvola formula (0.2, by default).
    R : float, optional
        Dynamic range of standard deviation. If None (default), R is set to half of the image dtype range.

    Returns
    -------
    image_binary : ndarray
        Binarised image.
    """

    T = threshold_sauvola(image, window_size, k, R)
    image_binary = image > T

    return image_binary


def otsu(image, n_bins=256, T_shift=0):
    """Return binary image such that all pixels with a value > :math:`T` are set to 1 and otherwise set to 0, where
    :math:`T` is defined globally by Otsu's method.

    Parameters
    ----------
    image : ndarray
        Input image. Must be grayscale.
    n_bins : int, optional
        Number of histogram bins for computing global threshold (256, by default).
    T_shift : float
        Alteration to make to detected threshold, :math:`T` (0, by default).

    Returns
    -------
    image_binary : ndarray
        Binarised image.
    """

    T = threshold_otsu(image, n_bins)
    image_binary = image > T + T_shift

    return image_binary


def k_means(image, n_clusters=2):
    """Return segmented image determined via :math:`k`-means clustering, where the number of segmented regions is
    n_clusters. Setting n_clusters = 2 (default) returns a binarised image.

    Parameters
    ----------
    image : ndarray
        Input image. Must be grayscale.
    n_clusters : int, optional
        Number of clusters to segment (2, by default).

    Returns
    -------
    image_seg : ndarray
        Segmented image.
    """

    image_vector = image.reshape(-1, 1)
    image_seg = np.zeros(image.shape)

    k_means = KMeans(n_clusters, random_state=0).fit(image_vector)
    codebook = k_means.cluster_centers_
    labels = k_means.predict(image_vector)
    label_idx = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_seg[i][j] = codebook[labels[label_idx]]
            label_idx += 1

    return image_seg
