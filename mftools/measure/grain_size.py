"""Automatic grain size measurement functions following [1]. Each function takes a binary image as input.

References
----------
.. [1] ASTM E1382−97 (2015) Standard Test Methods for Determining Average Grain Size Using Semiautomatic and Automatic
       Image Analysis
"""

import numpy as np
import scipy.ndimage as ndi
from skimage import measure
from skimage.segmentation import clear_border

from mftools.misc import remove_outliers


def l_over_a(image: np.ndarray, um_per_px: float) -> list:
    """Perform grain boundary length/area method.

    Parameters
    ----------
    image : ndarray
        Binary image.
    um_per_px : float
        Scaling (um per pixel).

    Returns
    -------
    loa : list of float
        List containing length over area for each grain (excluding grains which intersect the edge of the image field).

    References
    ----------
    .. [1] ASTM E1382−97 (2015) Standard Test Methods for Determining Average Grain Size Using Semiautomatic and
           Automatic Image Analysis
    """

    # Exclude grains intersecting edge of image field
    image = clear_border(image)

    # Label individual grains
    labels, num_grains = ndi.label(image)

    # Create storage for measurements
    loa = []

    # Iterate through each grain
    for n in range(num_grains):

        # Skip n=0 (corresponds to grain boundary) and include n=num_grains
        n += 1

        # Set all other grains to 0 and current grain to 1
        current_grain = labels == n

        # Calculate grain perimeter
        perimeter = measure.perimeter(current_grain)
        perimeter *= um_per_px

        # Calculate grain area
        area = np.sum(current_grain)
        area *= um_per_px**2

        # Calculate length over area
        loa += [perimeter/area]

    return loa


def intersection_count(image: np.ndarray, um_per_px: float, n_scans: int, min_length: float) -> list:
    """Perform intersection count method.

    Parameters
    ----------
    image : ndarray
        Binary image.
    um_per_px : float
        Scaling (um per pixel).
    n_scans : int
        Number of randomly orientated line scans to perform.
    min_length : float
        Minimum scan line length to include in measurements (um).

    Returns
    -------
    ic : list of float
        List of intersection counts over true length of scan lines.

    References
    ----------
    .. [1] ASTM E1382−97 (2015) Standard Test Methods for Determining Average Grain Size Using Semiautomatic and
           Automatic Image Analysis
    """

    # Create storage for returns
    ic = []

    # Initialise counter
    counter = 0

    while counter < n_scans:

        # Create storage for scan line start and end coordinates
        x = []

        # Generate two random integers from 0 to 3 without replacement (defines side selections for scan line start and
        # end coordinates)
        sides = np.random.choice(4, 2, replace=False)

        # Generate random coordinates for a point which lies on each side
        for side in sides:
            if side == 0:
                x.append(int(np.random.uniform(0, image.shape[0])))
                x.append(0)
            elif side == 1:
                x.append(0)
                x.append(int(np.random.uniform(0, image.shape[1])))
            elif side == 2:
                x.append(int(np.random.uniform(0, image.shape[0])))
                x.append(image.shape[1])
            else:
                x.append(image.shape[0])
                x.append(int(np.random.uniform(0, image.shape[1])))

        # Compute intensity profile of scan line
        line_profile = measure.profile_line(image, (x[0],x[1]), (x[2],x[3]))

        # Calculate length of scan line
        line_length = np.sqrt((x[0]-x[2])**2 + (x[1]-x[3])**2) * um_per_px

        # Skip measurement if scan line length is below min_length
        if line_length < min_length:
            continue

        # Count number of grain boundary intersections along scan line
        ic_temp = np.sum(np.abs(np.diff(line_profile)))

        # Compute number of intersections over scan line length
        ic.append(ic_temp/line_length)

        # Add 1 to counter
        counter += 1

    return ic


def chord_length(image: np.ndarray, um_per_px: float, n_scans: int,
                 min_length: float) -> list:
    """Perform intercept (chord) length method with randomly orientated scan lines.

    Parameters
    ----------
    image : ndarray
        Binary image.
    um_per_px : float
        Scaling (um per pixel).
    n_scans : int
        Number of randomly orientated line scans to perform.
    min_length : float
        Minimum scan line length to include in measurements (um).

    Returns
    -------
    cl : list of float
        List of chord lengths.

    References
    ----------
    .. [1] ASTM E1382−97 (2015) Standard Test Methods for Determining Average Grain Size Using Semiautomatic and
           Automatic Image Analysis
    """

    # Exclude grains intersecting edge of image field
    # image = clear_border(image)

    # Create storage for returns
    cl = []

    # Initialise counter
    counter = 0

    while counter < n_scans:

        # Create storage for scan line start and end coordinates
        x = []

        # Generate two random integers from 0 to 3 without replacement (defines side selections for scan line start and
        # end coordinates)
        sides = np.random.choice(4, 2, replace=False)

        # Generate random coordinates for a point which lies on each side
        for side in sides:
            if side == 0:
                x.append(int(np.random.uniform(0, image.shape[0])))
                x.append(0)
            elif side == 1:
                x.append(0)
                x.append(int(np.random.uniform(0, image.shape[1])))
            elif side == 2:
                x.append(int(np.random.uniform(0, image.shape[0])))
                x.append(image.shape[1])
            else:
                x.append(image.shape[0])
                x.append(int(np.random.uniform(0, image.shape[1])))

        # Compute intensity profile of scan line
        line_profile = measure.profile_line(image, (x[0],x[1]), (x[2],x[3]))

        # Calculate length of scan line
        line_length = np.sqrt((x[0]-x[2])**2 + (x[1]-x[3])**2) * um_per_px

        # Skip measurement if scan line length is below min_length
        if line_length < min_length:
            continue

        # Determine grain initial and end indices along scan line
        line_profile_dx = np.diff(line_profile)
        idx_init = np.argwhere(line_profile_dx > 0)
        idx_end = np.argwhere(line_profile_dx < 0)

        # Remove excess indices
        if idx_init.size > idx_end.size:
            idx_init = idx_init[:idx_end.size]
        elif idx_init.size < idx_end.size:
            idx_end = idx_end[:idx_init.size]

        # Ensure first index corresponds to an initial index
        try:
            while idx_init[0] > idx_end[0]:
                idx_end = idx_end[1:]
                idx_init = idx_init[:-1]

            # Calculate difference between index vectors
            idx_diff = np.abs(idx_init - idx_end)

            # Convert differences to distance in um and store mean chord length
            cl_temp = idx_diff / line_profile.size * line_length

            # Remove outliers
            try:
                cl_temp = remove_outliers(cl_temp, n_std=1)
            except:
                continue

            # Store mean chord length
            # cl += [np.mean(cl_temp)]
            cl.extend(cl_temp)

            # Update counter
            counter += 1

        except IndexError:
            continue

    return cl


def chord_length_uniform(image : np.ndarray, um_per_px: float,
                         n_scans: int) -> list:
    """Perform intercept (chord) length method with uniform grid of scan lines.

    Parameters
    ----------
    image : ndarray
        Binary image.
    um_per_px : float
        Scaling (um per pixel).
    n_scans : int
        Number of line scans in both the horizontal and vertical direction.

    Returns
    -------
    cl : list of float
        List of chord lengths.
    """

    # Create storage for returns
    cl = []
    n_alpha_grains_x = []
    n_alpha_grains_y = []
    total_alpha_length_x = []
    total_alpha_length_y = []

    for n in range(n_scans):

        # Perform horizontal scan
        src = (int(n * image.shape[0] / n_scans), 0)
        dst = (int(n * image.shape[0] / n_scans), image.shape[1])
        scan_line = measure.profile_line(image, src, dst)

        # Calculate length of scan line
        line_length = scan_line.size * um_per_px

        # Calculate length of total intersection with alpha grains
        total_alpha_length_x += [np.sum(scan_line) * um_per_px]

        # Determine number of alpha grains detected by scan line
        n_alpha_grains_x += [np.sum(np.diff(scan_line) > 0)]

        # Perform vertical scam
        src = (0, int(n * image.shape[1] / n_scans))
        dst = (image.shape[0], int(n * image.shape[1] / n_scans))
        scan_line = measure.profile_line(image, src, dst)

        # Calculate length of scan line
        line_length = scan_line.size * um_per_px

        # Calculate length of total intersection with alpha grains
        total_alpha_length_y += [np.sum(scan_line) * um_per_px]

        # Determine number of alpha grains detected by scan line
        n_alpha_grains_y += [np.sum(np.diff(scan_line) > 0)]

    lambda_x = np.mean(np.array(n_alpha_grains_x))
    lambda_y = np.mean(np.array(n_alpha_grains_y))
    L_x = np.mean(np.array(total_alpha_length_x))
    L_y = np.mean(np.array(total_alpha_length_y))

    cl = 1 / np.sqrt((lambda_x/L_x)**2 + (lambda_y/L_y)**2)

    return cl


def grain_count(image: np.ndarray, um_per_px: float) -> int:
    """Perform intersection count method. (REQUIRES OMMISSION OF GRAINS CROSSING THE BOUNDARY OF THE IMAGE FIELD)

    Parameters
    ----------
    image : ndarray
        Binary image.
    um_per_px : float
        Scaling (um per pixel).

    Returns
    -------
    gc : int
        Number of grains per um^2.

    References
    ----------
    .. [1] ASTM E1382−97 (2015) Standard Test Methods for Determining Average Grain Size Using Semiautomatic and
           Automatic Image Analysis
    """

    # Label and count grains
    _, num_grains = ndi.label(image)

    # Area of image field (um^2)
    image_area = image.shape[0]*image.shape[1]*um_per_px**2

    # Number of grains per um^2
    gc = num_grains/image_area

    return gc


def grain_areas(image: np.ndarray, um_per_px: float) -> list:
    """Perform individual grain area method.

    Parameters
    ----------
    image : ndarray
        Binary image.
    um_per_px : float
        Scaling (um per pixel).

    Returns
    -------
    ga : list of float
        List of grain areas.

    References
    ----------
    .. [1] ASTM E1382−97 (2015) Standard Test Methods for Determining Average Grain Size Using Semiautomatic and
           Automatic Image Analysis
    """

    # Exclude grains intersecting edge of image field
    image = clear_border(image)

    # Label individual grains
    labels, num_grains = ndi.label(image)

    # Create storage for measurements
    ga = []

    # Iterate through each grain
    for n in range(num_grains):

        # Skip n=0 (corresponds to grain boundary) and include n=num_grains
        n += 1

        # Set all other grains to 0 and current grain to 1
        current_grain = labels == n

        # Calculate grain area
        area = np.sum(current_grain)
        area *= um_per_px**2

        # Calculate length over area
        ga += [area]

    return ga
