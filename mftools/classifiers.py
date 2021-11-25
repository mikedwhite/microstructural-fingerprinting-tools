import graphlearning as gl
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_svm(xtrain, xttest, ytrain, yttest, kernel='linear'):
    """Train support vector machine (SVM) on training data, validate on test data and compute accuracy score for the
    validation.

    Parameters
    ----------
    xtrain : ndarray
        Array of fingerprints with shape (n_train, d), where n_train is the number of fingerprints within the training
        set and d is the length of each fingerprint.
    xttest : ndarray
        List of labels corresponding to xtrain, with shape (n_train, ).
    ytrain : ndarray
        Array of fingerprints with shape (n_test, d), where n_train is the number of fingerprints within the test set
        and d is the length of each fingerprint.
    xttest : ndarray
        List of labels corresponding to xttest, with shape (n_test, ).

    Returns
    -------
    accuracy : float
        Accuracy score when validating SVM on test data. Has range [0, 1].
    """

    print('Training SVM')
    svm = SVC(kernel=kernel, C=1.0, gamma='auto').fit(xtrain, ytrain)
    ypred = svm.predict(xttest)
    accuracy = accuracy_score(yttest, ypred)

    return accuracy


def train_rf(xtrain, xttest, ytrain, yttest):
    """Train random forest on training data, validate on test data and compute accuracy score for the validation.

    Parameters
    ----------
    xtrain : ndarray
        Array of fingerprints with shape (n_train, d), where n_train is the number of fingerprints within the training
        set and d is the length of each fingerprint.
    xttest : ndarray
        List of labels corresponding to xtrain, with shape (n_train, ).
    ytrain : ndarray
        Array of fingerprints with shape (n_test, d), where n_train is the number of fingerprints within the test set
        and d is the length of each fingerprint.
    xttest : ndarray
        List of labels corresponding to xttest, with shape (n_test, ).

    Returns
    -------
    accuracy : float
        Accuracy score when validating random forest on test data. Has range [0, 1].
    """

    print('Training Random Forest')
    clf = RandomForestClassifier(n_estimators=10000, max_depth=10).fit(xtrain, ytrain)
    ypred = clf.predict(xttest)
    accuracy = accuracy_score(yttest, ypred)

    return accuracy


def train_ul(xtrain, xttest, yttest, nclass, method='kmeans'):
    """Perform unsupervised learning (UL) on training data and compute accuracy score. Supports k-means and spectral
    clustering via the `method` parameter.

    Parameters
    ----------
    xtrain : ndarray
        Array of fingerprints with shape (n_train, d), where n_train is the number of fingerprints within the training
        set and d is the length of each fingerprint.
    xttest : ndarray
        List of labels corresponding to xtrain, with shape (n_train, ).
    xttest : ndarray
        List of labels corresponding to xttest, with shape (n_test, ).
    nclass : int
        Number of classes to split data into.
    method : str
        'kmeans' (deafult)
            k-means clustering.
        'spectral'
            Spectral clustering.

    Returns
    -------
    accuracy : float
        Accuracy score when validating UL on test data. Has range [0, 1].
    """

    print('Training unsupervised')
    scaler = StandardScaler()
    yttest = np.array(yttest)
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=nclass)
        kmeans.fit(scaler.fit_transform(xttest))
        ytrain_pred = kmeans.predict(scaler.fit_transform(xtrain))
        yttest_pred = kmeans.predict(scaler.fit_transform(xttest))
    elif method == 'spectral':
        kmeans = SpectralClustering(n_clusters=nclass, random_state=0, affinity='nearest_neighbors')
        ytrain_pred = kmeans.fit_predict(scaler.fit_transform(xtrain))
        yttest_pred = kmeans.fit_predict(scaler.fit_transform(xttest))
    else:
        print('method must be set as either `kmeans` or `spectral`.')
        return 1

    lab_truth = np.array(range(nclass))
    lab_map = np.array(([0, 1, 2],
                        [0, 2, 1],
                        [1, 0, 2],
                        [1, 2, 0],
                        [2, 0, 1],
                        [2, 1, 0]))

    accuracy_list = np.zeros(lab_map.shape[0])
    for m in range(lab_map.shape[0]):
        ytrain_pred_mapped = np.zeros(ytrain_pred.shape[0])
        for n in range(lab_truth.shape[0]):
            args = np.argwhere(ytrain_pred == n)
            ytrain_pred_mapped[args] = lab_map[m, n]

        yttest_pred_mapped = np.zeros(yttest_pred.shape[0])
        for n in range(lab_truth.shape[0]):
            args = np.argwhere(yttest_pred == n)
            yttest_pred_mapped[args] = lab_map[m, n]

        accuracy_list[m] = accuracy_score(yttest, yttest_pred_mapped)

    accuracy = np.max(accuracy_list)

    return accuracy


def train_ssl(xtrain, xttest, ytrain, yttest, frac_data):
    """Propagate labels via semi-supervised learning (SSL) and compute accuracy score.

    Parameters
    ----------
    xtrain : ndarray
        Array of fingerprints with shape (n_train, d), where n_train is the number of fingerprints within the training
        set and d is the length of each fingerprint.
    xttest : ndarray
        List of labels corresponding to xtrain, with shape (n_train, ).
    ytrain : ndarray
        Array of fingerprints with shape (n_test, d), where n_train is the number of fingerprints within the test set
        and d is the length of each fingerprint.
    xttest : ndarray
        List of labels corresponding to xttest, with shape (n_test, ).
    frac_data : float
        Fraction of ytrain used to initialise label propagation. Must have range (0, 1).

    Returns
    -------
    acc_laplace : float
        Accuracy score when validating SSL on test data via laplace learning. Has range [0, 1].
    acc_poisson : float
        Accuracy score when validating SSL on test data via poisson learning. Has range [0, 1].
    """

    print('Training semi-supervised')
    ytrain = np.array(ytrain)
    yttest = np.array(yttest)
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xttest = scaler.fit_transform(xttest)
    idx = np.random.permutation(ytrain.size)
    num_ind = np.int64(frac_data * idx.size)
    idx_train = np.asarray(range(num_ind), dtype=int)
    xdata = np.concatenate((xtrain[idx[0: num_ind], :], xtrain[idx[num_ind:], :], xttest), axis=0)
    ydata = np.concatenate((ytrain[idx[0: num_ind]], ytrain[idx[num_ind:]], yttest))

    neigh = NearestNeighbors(n_neighbors=10)
    neigh.fit(xdata)
    W = neigh.kneighbors_graph(xdata).toarray()

    labels_laplace = gl.graph_ssl(W, idx_train, ydata[0: num_ind], algorithm='laplace')
    labels_poisson = gl.graph_ssl(W, idx_train, ydata[0: num_ind], algorithm='poisson')

    acc_laplace = accuracy_score(ydata[num_ind:], labels_laplace[num_ind:])
    acc_poisson = accuracy_score(ydata[num_ind:], labels_poisson[num_ind:])

    return acc_laplace, acc_poisson