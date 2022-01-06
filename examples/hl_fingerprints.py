"""Enables SIFT H012V, SURF H012V, CNN H012V.

USAGE: Execute this script from within the `examples` directory.
Comment/uncomment params as desired.
There should be no more than one assignment per parameter.
Run via terminal for progress bars.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_ubyte
from sklearn.metrics.pairwise import chi2_kernel

from mftools.assess.classify import train_svm, train_rf, train_ul, train_ssl
from mftools.fingerprint.fingerprints import get_dict, learn_labeling, get_fingerprints_hl, reduce_fingerprints
from mftools.fingerprint.generate_features import (generate_feature_surf, generate_feature_sift,
                                                   generate_feature_cnn_featdict)

from data_proc import cross_validation_split_dataset1, cross_validation_split_dataset2

if __name__ == '__main__':

    params = {
        # -- File paths --------------------------------------------
        # 'input_path': 'data/dataset1/',
        'input_path': 'data/dataset2/',
        'output_path': 'out/',

        # -- Number of classes -------------------------------------
        # 'nclass': 2,
        'nclass': 3,

        # -- Number of clusters ------------------------------------
        'nclust': 10,

        # -- Feature extraction algorithm --------------------------
        # 'feature_generator': generate_feature_surf,
        'feature_generator': generate_feature_sift,
        # 'feature_generator': generate_feature_cnn_featdict,

        # -- CNN architecture (should be None if using SIFT/SURF) --
        'cnn': None,
        # 'cnn': 'alexnet',
        # 'cnn': 'vgg',

        # -- Fingerprint order -------------------------------------
        'order': 'h0',
        # 'order': 'h1',
        # 'order': 'h1v',
        # 'order': 'h2',
        # 'order': 'h2v',

        # -- PCA reduction options ---------------------------------
        # 'red_feat': True,
        'red_feat': False,
        # 'red_fing': True,
        'red_fing': False,
        'ncomponents': None,
        # 'ncomponents': 256,

        # -- Classifier parameters ---------------------------------
        'niter': 10,
        # 'svm_kernel': 'linear',
        'svm_kernel': chi2_kernel,
        'ssl_ratio': .05
    }

    INPUT_PATH = params['input_path']
    OUTPUT_PATH = params['output_path']
    NCLASS = params['nclass']
    NCLUST = params['nclust']
    FEATURE_GENERATOR = params['feature_generator']
    CNN = params['cnn']
    ORDER = params['order']
    RED_FEAT = params['red_feat']
    RED_FING = params['red_fing']
    NCOMPONENTS = params['ncomponents']
    NITER = params['niter']
    SSL_RATIO = params['ssl_ratio']
    SVM_KERNEL = params['svm_kernel']

    ACCURACY_svm = np.zeros(NITER)
    ACCURACY_rf = np.zeros(NITER)
    ACCURACY_ulk = np.zeros(NITER)
    ACCURACY_uls = np.zeros(NITER)
    ACCURACY_lap = np.zeros(NITER)
    ACCURACY_poi = np.zeros(NITER)

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    with open(f'{INPUT_PATH}micro_list.pkl', 'rb') as f:
        micro_list = pickle.load(f)
    with open(f'{INPUT_PATH}label_list.pkl', 'rb') as f:
        label_list = pickle.load(f)

    dict = get_dict(INPUT_PATH, micro_list, FEATURE_GENERATOR, CNN, RED_FEAT)
    np.save(f'{OUTPUT_PATH}dictionaries/dict.npy', dict)
    dict = np.load(f'{OUTPUT_PATH}dictionaries/dict.npy')

    kmeans = learn_labeling(dict, NCLUST)
    pickle.dump(kmeans, open(f'{OUTPUT_PATH}dictionaries/clust_model.sav', 'wb'))
    with open(f'{OUTPUT_PATH}dictionaries/clust_model.sav', 'rb') as f:
        kmeans = pickle.load(f)

    fingerprints = get_fingerprints_hl(INPUT_PATH, micro_list, kmeans, FEATURE_GENERATOR, ORDER, CNN, RED_FEAT)
    np.save(f'{OUTPUT_PATH}fingerprints/fingerprints_hl.npy', fingerprints)
    fingerprints = np.load(f'{OUTPUT_PATH}fingerprints/fingerprints_hl.npy')

    if RED_FING:
        fingerprints_reduced = reduce_fingerprints(fingerprints)
        np.save(f'{OUTPUT_PATH}fingerprints/fingerprints_hl_red{NCOMPONENTS}.npy',
                fingerprints)
        fingerprints = np.load(f'{OUTPUT_PATH}fingerprints/fingerprints_hl_red{NCOMPONENTS}.npy')

    micro_list_train_stack, micro_list_test_stack, label_list_train_stack, label_list_test_stack = \
        cross_validation_split_dataset2(micro_list, label_list, NITER)

    for ITER in range(NITER):
        print(f'Iteration {ITER + 1}/{NITER}')

        micro_list_train = micro_list_train_stack[ITER, :].tolist()
        label_list_train = label_list_train_stack[ITER, :].tolist()
        micro_list_test = micro_list_test_stack[ITER, :].tolist()
        label_list_test = label_list_test_stack[ITER, :].tolist()

        xtrain = fingerprints[micro_list_train, :]
        xtest = fingerprints[micro_list_test, :]

        accuracy_svm = train_svm(xtrain, xtest, label_list_train, label_list_test, kernel=SVM_KERNEL)
        accuracy_rf = train_rf(xtrain, xtest, label_list_train, label_list_test)
        accuracy_ulk = train_ul(xtrain, xtest, label_list_test, NCLASS, method='kmeans')
        accuracy_uls = train_ul(xtrain, xtest, label_list_test, NCLASS, method='spectral')
        accuracy_lap, accuracy_poi = train_ssl(xtrain, xtest, label_list_train, label_list_test, SSL_RATIO)

        print(f'SVM accuracy score: {accuracy_svm:.3f}')
        print(f'Random forest accuracy score: {accuracy_rf:.3f}')
        print(f'k-means unsupervised accuracy score: {accuracy_ulk:.3f}')
        print(f'Spectral unsupervised accuracy score: {accuracy_uls:.3f}')
        print(f'SSL Laplace accuracy score: {accuracy_lap:.3f}')
        print(f'SSL Poisson accuracy score: {accuracy_poi:.3f}')

        ACCURACY_svm[ITER] = accuracy_svm
        ACCURACY_rf[ITER] = accuracy_rf
        ACCURACY_ulk[ITER] = accuracy_ulk
        ACCURACY_uls[ITER] = accuracy_uls
        ACCURACY_lap[ITER] = accuracy_lap
        ACCURACY_poi[ITER] = accuracy_poi

    print(params)
    print(f'CV SVM accuracy score: {np.mean(ACCURACY_svm):.3f} +- {np.std(ACCURACY_svm):.4f}')
    print(f'CV Random forest accuracy score: {np.mean(ACCURACY_rf):.3f} +- {np.std(ACCURACY_rf):.4f}')
    print(f'CV k-means unsupervised accuracy score: {np.mean(ACCURACY_ulk):.3f} +- {np.std(ACCURACY_ulk):.4f}')
    print(f'CV spectral unsupervised accuracy score: {np.mean(ACCURACY_uls):.3f} +- {np.std(ACCURACY_uls):.4f}')
    print(f'CV SSL Laplace accuracy score: {np.mean(ACCURACY_lap):.3f} +- {np.std(ACCURACY_lap):.4f}')
    print(f'CV SSL Poisson accuracy score: {np.mean(ACCURACY_poi):.3f} +- {np.std(ACCURACY_poi):.4f}')
