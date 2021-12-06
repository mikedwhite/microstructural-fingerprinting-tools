"""Functions for determining micrograph scaling.
"""


def determine_scaling(image, bar_length_um, bar_frac):
    """Determine um per pixels scaling for an image provided the bar length in um and the fraction of the image for
    which it occupies.

    Parameters
    ----------
    image : ndarray
        Input image. Must be grayscale.
    bar_length_um : float
        Length of scale bar (um).
    bar_frac : float
        Fraction of the image width occupied by the scale bar.

    Returns
    -------
    um_per_px : float
        Scaling (um per px).
    """

    # Convert bar length from um to pixels
    bar_length_px = bar_frac * image.shape[1]

    # Determine conversion
    um_per_px = bar_length_um/bar_length_px

    return um_per_px
