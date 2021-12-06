"""Enables CNN flattened final convolution layer and CNN max-pooled final convolution layer.

USAGE: Execute this script from within the `examples` directory.
Run via terminal for progress bars.
"""


import os
import pickle

import numpy as np

from mftools.assess.classify import train_svm, train_rf, train_ul, train_ssl
from mftools.fingerprint.fingerprints import get_fingerprints_cnn
from mftools.fingerprint.generate_features import generate_feature_cnn_flatten, generate_feature_cnn_maxpool

from data_proc_dataset2 import cross_validation_split


if __name__ == '__main__':

    params = {'input_path': 'data/dataset2/',
              'output_path': 'out/',
              'nclass': 3,
              # 'feature_generator': generate_feature_cnn_flatten,
              'feature_generator': generate_feature_cnn_maxpool,
              'cnn': 'alexnet',
              # 'cnn': 'vgg',
              'niter': 10,
              'ssl_ratio': .05}

    INPUT_PATH = params['input_path']
    OUTPUT_PATH = params['output_path']
    NCLASS = params['nclass']
    FEATURE_GENERATOR = params['feature_generator']
    CNN = params['cnn']
    NITER = params['niter']
    SSL_RATIO = params['ssl_ratio']

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

    fingerprints = get_fingerprints_cnn(INPUT_PATH, micro_list, FEATURE_GENERATOR, CNN)
    np.save(f'{OUTPUT_PATH}/cnn_fingerprints.npy', fingerprints)
    fingerprints = np.load(f'{OUTPUT_PATH}/cnn_fingerprints.npy')

    micro_list_train_stack, micro_list_test_stack, label_list_train_stack, label_list_test_stack =\
        cross_validation_split(micro_list, label_list, NITER)

    for ITER in range(NITER):

        print(f'Iteration {ITER+1}/{NITER}')

        micro_list_train = micro_list_train_stack[ITER, :].tolist()
        label_list_train = label_list_train_stack[ITER, :].tolist()
        micro_list_test = micro_list_test_stack[ITER, :].tolist()
        label_list_test = label_list_test_stack[ITER, :].tolist()

        xtrain = fingerprints[micro_list_train, :]
        xtest = fingerprints[micro_list_test, :]

        accuracy_svm = train_svm(xtrain, xtest, label_list_train, label_list_test)
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
    print(f'SVM accuracy score: {np.mean(ACCURACY_svm):.3f} +- {np.std(ACCURACY_svm):.4f}')
    print(f'Random forest accuracy score: {np.mean(ACCURACY_rf):.3f} +- {np.std(ACCURACY_rf):.4f}')
    print(f'k-means unsupervised accuracy score: {np.mean(ACCURACY_ulk):.3f} +- {np.std(ACCURACY_ulk):.4f}')
    print(f'Spectral unsupervised accuracy score: {np.mean(ACCURACY_uls):.3f} +- {np.std(ACCURACY_uls):.4f}')
    print(f'SSL Laplace accuracy score: {np.mean(ACCURACY_lap):.3f} +- {np.std(ACCURACY_lap):.4f}')
    print(f'SSL Poisson accuracy score: {np.mean(ACCURACY_poi):.3f} +- {np.std(ACCURACY_poi):.4f}')
