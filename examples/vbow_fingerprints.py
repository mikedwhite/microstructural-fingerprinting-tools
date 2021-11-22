"""Enables SIFT H012V, SURF H012V, CNN H012V.

USAGE: Execute this script from within the `examples` directory.
Run via terminal for progress bars.
"""


import os
import pickle

import numpy as np
from sklearn.metrics.pairwise import chi2_kernel

from mftools.classifiers import train_svm, train_rf, train_ul, train_ssl
from mftools.fingerprints import get_dict, learn_labeling, get_fingerprints_vbow
from mftools.generate_features import generate_feature_surf, generate_feature_sift, generate_feature_cnn_featdict

from data_proc_dataset2 import cross_validation_split


if __name__ == '__main__':

    params = {'input_path': 'data/dataset2/',
              'output_path': 'out/',
              'nclass': 3,
              'nclust': 10,
              'feature_generator': generate_feature_surf,
              # 'feature_generator': generate_feature_sift,
              # 'feature_generator': generate_feature_cnn_featdict,
              'cnn': None,
              # 'cnn': 'alexnet',
              # 'cnn': 'vgg',
              # 'red': True,
              'red': False,
              'order': 'h0',
              # 'order': 'h1',
              # 'order': 'h1v',
              # 'order': 'h2',
              # 'order': 'h2v',
              'niter': 10,
              'ssl_ratio': .05,
              'svm_kernel': 'linear'
              # 'svm_kernel': chi2_kernel
              }

    INPUT_PATH = params['input_path']
    OUTPUT_PATH = params['output_path']
    NCLASS = params['nclass']
    NCLUST = params['nclust']
    FEATURE_GENERATOR = params['feature_generator']
    CNN = params['cnn']
    RED = params['red']
    ORDER = params['order']
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

    dict = get_dict(INPUT_PATH, micro_list, FEATURE_GENERATOR, CNN, RED)
    kmeans = learn_labeling(dict, NCLUST)
    pickle.dump(kmeans, open(f'{OUTPUT_PATH}/clust_model.sav', 'wb'))
    with open(f'{OUTPUT_PATH}/clust_model.sav', 'rb') as f:
        kmeans = pickle.load(f)

    fingerprints = get_fingerprints_vbow(INPUT_PATH, micro_list, kmeans, FEATURE_GENERATOR, ORDER, CNN, RED)
    np.save(f'{OUTPUT_PATH}/vbow_fingerprints.npy', fingerprints)
    fingerprints = np.load(f'{OUTPUT_PATH}/vbow_fingerprints.npy')

    micro_list_train_stack, micro_list_ttest_stack, label_list_train_stack, label_list_ttest_stack =\
        cross_validation_split(micro_list, label_list, NITER)

    for ITER in range(NITER):

        print(f'Iteration {ITER+1}/{NITER}')

        micro_list_train = micro_list_train_stack[ITER, :].tolist()
        label_list_train = label_list_train_stack[ITER, :].tolist()
        micro_list_ttest = micro_list_ttest_stack[ITER, :].tolist()
        label_list_ttest = label_list_ttest_stack[ITER, :].tolist()

        xtrain = fingerprints[micro_list_train, :]
        xttest = fingerprints[micro_list_ttest, :]

        accuracy_svm = train_svm(xtrain, xttest, label_list_train, label_list_ttest, kernel=SVM_KERNEL)
        accuracy_rf = train_rf(xtrain, xttest, label_list_train, label_list_ttest)
        accuracy_ulk = train_ul(xtrain, xttest, label_list_ttest, NCLASS, method='kmeans')
        accuracy_uls = train_ul(xtrain, xttest, label_list_ttest, NCLASS, method='spectral')
        accuracy_lap, accuracy_poi = train_ssl(xtrain, xttest, label_list_train, label_list_ttest, SSL_RATIO)

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
