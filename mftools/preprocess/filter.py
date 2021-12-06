"""Homomorphic filters which map from image domain to log domain to Fourier domain before application of filter.
Mappings are inverted before the filtered image is returned.
"""

import numpy as np
from skimage import img_as_float


def hmf_gauss(image, sigma=1., alpha=0., beta=1., pass_type='low'):
    """Apply emphasised Gaussian homomorphic filter of the form

    .. math::
        H_{emphasis} = \\alpha + \\beta * H,

    where :math:`H` is a Gaussian filter with standard deviation :math:`\\sigma`.

    Parameters
    ----------
    image : ndarray
        Input image. Must be grayscale.
    sigma : float, optional
        Gaussian filter standard deviation (1.0, by default).
    alpha : float, optional
        Gaussian filter offset (0.0, by default).
    beta : float, optional
        Gaussian filter scaling (1.0, by default).
    pass_type : str, optional
        'low' (default)
            Applies a low-pass filter.
        'high'
            Applies a high-pass filter.

    Returns
    -------
    image : ndarray
        Grayscale filtered image normalised to the range [0,1].
    """

    # Define original image size
    m, n = image.shape[0], image.shape[1]

    # Convert to float
    image = img_as_float(image)

    # Take log
    image = np.log(1 + image)

    # Pad image with replicated image edges
    image = np.pad(image, ((np.int(m/2),), (np.int(n/2),)), mode='edge')

    # Construct Gaussian filter
    M, N = 2*m, 2*n
    X, Y = np.meshgrid(np.linspace(0, N, N+1), np.linspace(0, M, M+1))
    X_centre, Y_centre = np.ceil((N+1)/2), np.ceil((M+1)/2)
    hmf = np.exp(-((X-X_centre)**2+(Y-Y_centre)**2)/(2*sigma**2))
    if pass_type == 'high':
        hmf = 1 - hmf

    # Rearrange filter
    hmf = np.fft.fftshift(hmf)

    # Offset and scale filter
    hmf = alpha + beta*hmf

    # Apply Fourier transform to image
    image = np.fft.fft2(image, s=(M + 1, N + 1))

    # Apply filter and inverse fourier transform
    image = np.real(np.fft.ifft2(hmf * image))

    # Crop filtered image to original size
    image = image[int(np.ceil(M/2)-m/2):int(np.ceil(M/2)+m/2),
                  int(np.ceil(N/2)-n/2):int(np.ceil(N/2)+n/2)]

    # Take exponential
    image = np.exp(image) - 1

    # Normalise image to lie in range 0 to 1
    image -= np.amin(image)
    image /= np.amax(image)

    return image
