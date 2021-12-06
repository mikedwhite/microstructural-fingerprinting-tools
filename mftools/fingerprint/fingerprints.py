import numpy as np
from progress.bar import IncrementalBar
from skimage import io
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def get_dict(input_path, micro_list, feature_generator, cnn=None, red=False):
    """Get dictionary of features from list of input images.

    Parameters
    ----------
    input_path : str
        Path to image data.
    micro_list : list
        List of image filenames.
    feature_generator : mftools.generate_features
        Feature generation method. Options are available in the generate_features module.
    cnn : str or None
        None (default)
            Feature generator is not based on CNN architecture.
        'alexnet'
            Generates CNN features from AlexNet architecture.
        'vgg'
            Generates CNN features from VGG architecture.
    red : bool
        False (default)
            Full feature stack is used for fingerprint construction.
        True
            Feature stack is reduced via PCA to shape :math:`(d, d)`, where :math:`d` is the dimension of each feature
            vector.

    Returns
    -------
    dict : ndarray
        Dictionary of features with shape :math:`(N, d)`, where :math:`N` is the number of features and d is the length
        of each feature.
    """

    nimage = len(micro_list)
    dict = []

    with IncrementalBar('Generating dictionary', max=nimage, suffix='%(percent).1f%% - %(eta)ds') as bar:

        for n in range(nimage):
            image = io.imread(f'{input_path}micrographs/{micro_list[n]}')
            if cnn is None:
                xfeat = feature_generator(image)
            else:
                xfeat = feature_generator(image, cnn)

            if red is True:
                if xfeat.shape[0] > xfeat.shape[1]:
                    J = xfeat.shape[1]
                    pca = PCA(n_components=J)
                    xfeat = np.transpose(pca.fit_transform(np.transpose(xfeat)))
                else:
                    print('Number of features must be greater than dimension of feature vectors')
                    return 1

            if n == 0:
                dict = np.copy(xfeat)
            else:
                dict = np.vstack((dict, xfeat))

            bar.next()

    return dict


def learn_labeling(dict, nclust):
    """Train cluster model on dictionary of features.

    Parameters
    ----------
    dict : ndarray
        Dictionary of features with shape :math:`(N, d)`, where :math:`N` is the number of features and :math:`d` is
        the length of each feature.
    nclust : int
        Number of clusters to assign when training :math:`k`-means model.

    Returns
    -------
    kmeans : sklearn.cluster.kmeans
        Trained :math:`k`-means cluster model corresponding to dictionary input.
    """

    print('Learning cluster labels')
    kmeans = KMeans(n_clusters=nclust)
    kmeans.fit(dict)

    return kmeans


def single_image_fingerprint_h0(xfeat, kmeans):
    """Generate fingerprint :math:`H_0` from :math:`k`-means predictions of cluster centres from single image features.

    Parameters
    ----------
    xfeat : ndarray
        Dictionary of features with shape :math:`(N, d)`, where :math:`N` is the number of features and :math:`d` is the
        length of each feature.
    kmeans : sklearn.cluster.kmeans
        :math:`k`-means cluster model corresponding to dictionary of features from the whole dataset.

    Returns
    -------
    fingerprint : ndarray
        :math:`H_0` fingerprint.
    """

    xlab = kmeans.predict(xfeat)
    nclust = kmeans.cluster_centers_.shape[0]
    fingerprint, _ = np.histogram(xlab, bins=nclust)
    fingerprint = fingerprint / np.sum(fingerprint)

    return fingerprint


def single_image_fingerprint_h1(xfeat, kmeans):
    """Generate fingerprint :math:`H_1` from :math:`k`-means predictions of cluster centres from single image features.

    Parameters
    ----------
    xfeat : ndarray
        Dictionary of features with shape :math:`(N, d)`, where :math:`N` is the number of features and :math:`d` is the
        length of each feature.
    kmeans : sklearn.cluster.kmeans
        :math:`k`-means cluster model corresponding to dictionary of features from the whole dataset.

    Returns
    -------
    fingerprint : ndarray
        :math:`H_1` fingerprint.
    """

    nvec, dimv = xfeat.shape
    xlab = kmeans.predict(xfeat)
    nclust = kmeans.cluster_centers_.shape[0]
    fingerprint = np.zeros((nclust, dimv))
    counter = np.zeros((nclust), dtype=int)

    for kvec in range(nvec):
        counter[np.int64(xlab[kvec])] = counter[np.int64(xlab[kvec])] + 1
        fingerprint[np.int64(xlab[kvec]), :] = fingerprint[np.int64(xlab[kvec]), :] + xfeat[kvec, :]
    for kclust in range(nclust):
        if counter[kclust] > 0:
            fingerprint[kclust, :] = fingerprint[kclust, :] / counter[kclust]

    return np.reshape(fingerprint, -1)


def single_image_fingerprint_h1v(xfeat, kmeans):
    """Generate fingerprint :math:`H_1^v` from :math:`k`-means predictions of cluster centres from single image
    features.

    Parameters
    ----------
    xfeat : ndarray
        Dictionary of features with shape :math:`(N, d)`, where :math:`N` is the number of features and :math:`d` is the
        length of each feature.
    kmeans : sklearn.cluster.kmeans
        :math:`k`-means cluster model corresponding to dictionary of features from the whole dataset.

    Returns
    -------
    fingerprint : ndarray
        :math:`H_1^v` fingerprint.
    """

    nvec, dimv = xfeat.shape
    xlab = kmeans.predict(xfeat)
    nclust = kmeans.n_clusters
    fingerprint = np.zeros([nclust, dimv])
    centers = kmeans.cluster_centers_

    for kclust in range(nclust):
        if np.sum(xlab == kclust) > 0:
            fingerprint[kclust] = np.sum(xfeat[xlab == kclust, :] - centers[kclust], axis=0)

    fingerprint = fingerprint.flatten()
    fingerprint = np.sign(fingerprint) * np.sqrt(np.abs(fingerprint))
    fingerprint = fingerprint / np.sqrt(np.dot(fingerprint, fingerprint))

    return fingerprint


def single_image_fingerprint_h2(xfeat, kmeans):
    """Generate fingerprint :math:`H_2` from :math:`k`-means predictions of cluster centres from single image features.

    Parameters
    ----------
    xfeat : ndarray
        Dictionary of features with shape :math:`(N, d)`, where :math:`N` is the number of features and :math:`d` is the
        length of each feature.
    kmeans : sklearn.cluster.kmeans
        :math:`k`-means cluster model corresponding to dictionary of features from the whole dataset.

    Returns
    -------
    fingerprint : ndarray
        :math:`H_2` fingerprint.
    """

    nvec, dimv = xfeat.shape
    xlab = kmeans.predict(xfeat)
    nclust = kmeans.cluster_centers_.shape[0]
    fingerprint = np.zeros((nclust, dimv, dimv))

    fingerprint_h1 = np.reshape(single_image_fingerprint_h1(xfeat, kmeans), (nclust, dimv))

    for kclust in range(nclust):
        args = np.argwhere(xlab == kclust)[:, 0]
        if args.shape[0] > 0:
            J_k = args.shape[0]
            for arg in args:
                fingerprint[kclust, :, :] = fingerprint[kclust, :, :] +\
                                            np.outer(xfeat[arg, :] - fingerprint_h1[kclust, :],
                                                     xfeat[arg, :] - fingerprint_h1[kclust, :])
            fingerprint[kclust, :, :] = (1 / J_k) * fingerprint[kclust, :, :]

    fingerprint_diag = np.zeros((nclust, dimv))
    for kclust in range(nclust):
        fingerprint_diag[kclust, :] = fingerprint[kclust, :, :].diagonal()

    fingerprint_h2 = np.reshape(fingerprint_diag, -1)
    # fingerprint_h1 = np.reshape(fingerprint_h1, -1)
    # fingerprint = np.hstack((fingerprint_h1, fingerprint_h2))

    return fingerprint_h2


def single_image_fingerprint_h2v(xfeat, kmeans):
    """Generate fingerprint :math:`H_2^v` from :math:`k`-means predictions of cluster centres from single image features.

    Parameters
    ----------
    xfeat : ndarray
        Dictionary of features with shape :math:`(N, d)`, where :math:`N` is the number of features and :math:`d` is the
        length of each feature.
    kmeans : sklearn.cluster.kmeans
        :math:`k`-means cluster model corresponding to dictionary of features from the whole dataset.

    Returns
    -------
    fingerprint : ndarray
        :math:`H_2^v` fingerprint.
    """

    nvec, dimv = xfeat.shape
    xlab = kmeans.predict(xfeat)
    nclust = kmeans.cluster_centers_.shape[0]
    centers = kmeans.cluster_centers_
    fingerprint = np.zeros((nclust, dimv, dimv))

    fingerprint_h1 = np.reshape(single_image_fingerprint_h1(xfeat, kmeans), (nclust, dimv))

    for kclust in range(nclust):
        args = np.argwhere(xlab == kclust)[:, 0]
        if args.shape[0] > 0:
            J_k = args.shape[0]
            for arg in args:
                fingerprint[kclust, :, :] = fingerprint[kclust, :, :] +\
                                            np.outer((xfeat[arg, :] - centers[kclust]) /
                                                     np.sqrt(np.dot(fingerprint_h1[kclust, :] - centers[kclust],
                                                                    fingerprint_h1[kclust, :] - centers[kclust])),
                                                     (xfeat[arg, :] - centers[kclust]) /
                                                     np.sqrt(np.dot(fingerprint_h1[kclust, :] - centers[kclust],
                                                                    fingerprint_h1[kclust, :] - centers[kclust])))
            fingerprint[kclust, :, :] = (1 / J_k) * fingerprint[kclust, :, :]

    # Reduce by diagonalizing
    # fingerprint_diag = np.zeros((nclust, dimv))
    # for kclust in range(nclust):
    #     fingerprint_diag[kclust, :] = fingerprint[kclust, :, :].diagonal()

    fingerprint_h2v = np.reshape(fingerprint, -1)
    # fingerprint_h1v = single_image_fingerprint_h1v(xfeat, kmeans)
    # fingerprint_h12v = np.hstack((fingerprint_h1v, fingerprint_h2v))

    return fingerprint_h2v


def get_fingerprint(image, kmeans, feature_generator, order='h0', cnn=None, red=False):
    """Call a specified function for extracting :math:`H_l` fingerprints, where :math:`l` denotes the order of the
    fingerprint.

    Parameters
    ----------
    image : ndarray
        Image data.
    kmeans : sklearn.cluster.kmeans
        :math:`k`-means cluster model corresponding to dictionary of features from the whole dataset.
    feature_generator : mftools.generate_features
        Feature generation method. Options are available in the generate_features module.
    order : str
        'h0' (default)
            Return :math:`H_0` fingerprint.
        'h1'
            Return :math:`H_1` fingerprint.
        'h1v'
            Return :math:`H_1^v` fingerprint.
        'h2'
            Return :math:`H_2` fingerprint.
        'h2v'
            Return :math:`H_2^v` fingerprint.
    cnn : str or None
        None (default)
            Feature generator is not based on CNN architecture.
        'alexnet'
            Generates CNN features from AlexNet architecture.
        'vgg'
            Generates CNN features from VGG architecture.
    red : bool
        False (default)
            Full feature stack is used for fingerprint construction.
        True
            Feature stack is reduced via PCA to shape :math:`(d, d)`, where :math:`d` is the dimension of each feature
            vector.

    Returns
    -------
    fingerprint : ndarray
        Fingerprint of specified order.
    """


    if cnn is None:
        xfeat = feature_generator(image)
    else:
        xfeat = feature_generator(image, cnn)

    if red is True:
        if xfeat.shape[0] > xfeat.shape[1]:
            J = xfeat.shape[1]
            pca = PCA(n_components=J)
            xfeat = np.transpose(pca.fit_transform(np.transpose(xfeat)))
        else:
            print('Number of features must be greater than dimension of feature vectors')
            return 1

    if order == 'h0':
        fingerprint = single_image_fingerprint_h0(xfeat, kmeans)
    elif order == 'h1':
        fingerprint = single_image_fingerprint_h1(xfeat, kmeans)
    elif order == 'h1v':
        fingerprint = single_image_fingerprint_h1v(xfeat, kmeans)
    elif order == 'h2':
        fingerprint = single_image_fingerprint_h2(xfeat, kmeans)
    elif order == 'h2v':
        fingerprint = single_image_fingerprint_h2v(xfeat, kmeans)

    else:
        print('Order must be `h0`, `h1`, `h2`, `h1v`, or `h2v`')
        return 1

    return fingerprint


def get_fingerprints_vbow(input_path, micro_list, kmeans, feature_generator, order='h0', cnn=None, red=False):
    """Generate fingerprints for whole images based on :math:`H_{l}` framework, where :math:`l` denotes the order of the
    fingerprint.

    Parameters
    ----------
    input_path : str
        Path to image data.
    micro_list : list
        List of image filenames.
    kmeans : sklearn.cluster.kmeans
        :math:`k`-means cluster model corresponding to dictionary of features from the whole dataset.
    feature_generator : mftools.generate_features
        Feature generation method. Options are available in the generate_features module.
    order : str
        'h0' (default)
            Return :math:`H_0` fingerprint.
        'h1'
            Return :math:`H_1` fingerprint.
        'h1v'
            Return :math:`H_1^v` fingerprint.
        'h2'
            Return :math:`H_2` fingerprint.
        'h2v'
            Return :math:`H_2^v` fingerprint.
    cnn : str or None
        None (default)
            Feature generator is not based on CNN architecture.
        'alexnet'
            Generates CNN features from AlexNet architecture.
        'vgg'
            Generates CNN features from VGG architecture.
    red : bool
        False (default)
            Full feature stack is used for fingerprint construction.
        True
            Feature stack is reduced via PCA to shape :math:`(d, d)`, where :math:`d` is the dimension of each feature
            vector.

    Returns
    -------
    fingerprints : ndarray
        Array of fingerprints with shape :math:`(N, d)`, where :math:`N` is the number of input images and :math:`d` is
        the length of each fingerprint.
    """

    nimage = len(micro_list)
    fingerprints = []

    with IncrementalBar('Generating fingerprints', max=nimage, suffix='%(percent).1f%% - %(eta)ds') as bar:

        for n in range(nimage):
            image = io.imread(f'{input_path}micrographs/{micro_list[n]}')
            if n == 0:
                fingerprints = get_fingerprint(image, kmeans, feature_generator, order, cnn, red)
            else:
                fingerprints = np.vstack((fingerprints, get_fingerprint(image, kmeans, feature_generator, order, cnn, red)))

            bar.next()  # update progress bar

    return fingerprints


def get_fingerprints_cnn(input_path, micro_list, feature_generator, cnn='alexnet'):
    """Generate fingerprints for whole images based on CNN features alone.

    Parameters
    ----------
    input_path : str
        Path to image data.
    micro_list : list
        List of image filenames.
    feature_generator : mftools.generate_features
        Feature generation method. Options are available in the generate_features module.
    cnn : str
        'alexnet' (default)
            Generates CNN features from AlexNet architecture.
        'vgg'
            Generates CNN features from VGG architecture.

    Returns
    -------
    fingerprints : ndarray
        Array of fingerprints with shape :math:`(N, d)`, where :math:`N` is the number of input images and :math:`d` is
        the length of each fingerprint.
    """

    nimage = len(micro_list)
    fingerprints = []

    with IncrementalBar('Generating fingerprints', max=nimage, suffix='%(percent).1f%% - %(eta)ds') as bar:

        for n in range(nimage):
            image = io.imread(f'{input_path}micrographs/{micro_list[n]}')
            xfeat = feature_generator(image, cnn)
            if n == 0:
                fingerprints = np.copy(xfeat)
            else:
                fingerprints = np.vstack((fingerprints, xfeat))

            bar.next()  # update progress bar

    return fingerprints
