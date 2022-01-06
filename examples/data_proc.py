import numpy as np


def cross_validation_split_dataset1(micro_list, label_list, niter):
    r"""Split :math:`N` micrographs into train/test sets for :math:`k`-fold cross-validation, where :math:`k` is
    equivalent to the parameter **niter**.

    Parameters
    ----------
    micro_list : list
        List of micrograph file names for the whole dataset.
    label_list : list
        List of labels corresponding to micrograph class labels.
    niter : int
        Number of train/test split iterations to perform.

    Returns
    -------
    micro_list_train_stack : ndarray
        Lists of micrograph filenames to comprise training sets stacked into an array of shape
        (**niter**, :math:`N`(**niter** - 1)/**niter**)
    micro_list_test_stack : ndarray
        Lists of micrograph filenames to comprise test sets stacked into an array of shape
        (**niter**, :math:`N`/**niter**)
    label_list_train_stack : ndarray
        Array of class labels corresponding to **micro_list_train_stack**
    label_list_test_stack : ndarray
        Array of class labels corresponding to **micro_list_test_stack**
    """

    micro_list = np.array(micro_list)
    label_list = np.array(label_list)
    nimage = label_list.size
    ntest = int(np.round(nimage / niter))
    ntrain = int(nimage - ntest)

    shuffle_order_bimod = np.random.choice(20, 20, replace=False)
    shuffle_order_lamel = np.random.choice(20, 20, replace=False) + 20

    micro_list_train_stack = np.zeros(([niter, ntrain]))
    label_list_train_stack = np.zeros(([niter, ntrain]))
    micro_list_test_stack = np.zeros(([niter, ntest]))
    label_list_test_stack = np.zeros(([niter, ntest]))

    for n in range(niter):
        test_ind = np.concatenate((shuffle_order_bimod[n*2:(n+1)*2], shuffle_order_lamel[n*2:(n+1)*2]))
        label_list_test_stack[n, :] = label_list[test_ind]
        all_ind = range(nimage)
        train_ind = np.array([element for element in all_ind if element not in test_ind])
        label_list_train_stack[n, :] = label_list[train_ind]
        if n == 0:
            micro_list_test_stack = test_ind
            micro_list_train_stack = train_ind
        else:
            micro_list_test_stack = np.vstack((micro_list_test_stack, test_ind))
            micro_list_train_stack = np.vstack((micro_list_train_stack, train_ind))

    for n in range(niter):
        shuffle_order_train = np.random.choice(ntrain, ntrain, replace=False)
        shuffle_order_test = np.random.choice(ntest, ntest, replace=False)
        micro_list_train_stack[n, :] = micro_list_train_stack[n, shuffle_order_train]
        label_list_train_stack[n, :] = label_list_train_stack[n, shuffle_order_train]
        micro_list_test_stack[n, :] = micro_list_test_stack[n, shuffle_order_test]
        label_list_test_stack[n, :] = label_list_test_stack[n, shuffle_order_test]

    return micro_list_train_stack, micro_list_test_stack, label_list_train_stack, label_list_test_stack


def cross_validation_split_dataset2(micro_list, label_list, niter):
    r"""Split :math:`N` micrographs into train/test sets for :math:`k`-fold cross-validation, where :math:`k` is
    equivalent to the parameter **niter**.

    Parameters
    ----------
    micro_list : list
        List of micrograph file names for the whole dataset.
    label_list : list
        List of labels corresponding to micrograph class labels.
    niter : int
        Number of train/test split iterations to perform.

    Returns
    -------
    micro_list_train_stack : ndarray
        Lists of micrograph filenames to comprise training sets stacked into an array of shape
        (**niter**, :math:`N`(**niter** - 1)/**niter**)
    micro_list_test_stack : ndarray
        Lists of micrograph filenames to comprise test sets stacked into an array of shape
        (**niter**, :math:`N`/**niter**)
    label_list_train_stack : ndarray
        Array of class labels corresponding to **micro_list_train_stack**
    label_list_test_stack : ndarray
        Array of class labels corresponding to **micro_list_test_stack**
    """

    micro_list = np.array(micro_list)
    label_list = np.array(label_list)
    nimage = label_list.size
    ntest = int(np.round(nimage / niter))
    ntrain = int(nimage - ntest)

    shuffle_order_carbn = np.random.choice(200, 200, replace=False)
    shuffle_order_pearl = np.random.choice(200, 200, replace=False) + 200
    shuffle_order_spher = np.random.choice(200, 200, replace=False) + 400

    micro_list_train_stack = np.zeros(([niter, ntrain]))
    label_list_train_stack = np.zeros(([niter, ntrain]))
    micro_list_test_stack = np.zeros(([niter, ntest]))
    label_list_test_stack = np.zeros(([niter, ntest]))

    for n in range(niter):
        test_ind = np.concatenate((shuffle_order_carbn[n*20:(n+1)*20], shuffle_order_pearl[n*20:(n+1)*20],
                                    shuffle_order_spher[n*20:(n+1)*20]))
        label_list_test_stack[n, :] = label_list[test_ind]
        all_ind = range(nimage)
        train_ind = np.array([element for element in all_ind if element not in test_ind])
        label_list_train_stack[n, :] = label_list[train_ind]
        if n == 0:
            micro_list_test_stack = test_ind
            micro_list_train_stack = train_ind
        else:
            micro_list_test_stack = np.vstack((micro_list_test_stack, test_ind))
            micro_list_train_stack = np.vstack((micro_list_train_stack, train_ind))

    for n in range(niter):
        shuffle_order_train = np.random.choice(ntrain, ntrain, replace=False)
        shuffle_order_test = np.random.choice(ntest, ntest, replace=False)
        micro_list_train_stack[n, :] = micro_list_train_stack[n, shuffle_order_train]
        label_list_train_stack[n, :] = label_list_train_stack[n, shuffle_order_train]
        micro_list_test_stack[n, :] = micro_list_test_stack[n, shuffle_order_test]
        label_list_test_stack[n, :] = label_list_test_stack[n, shuffle_order_test]

    return micro_list_train_stack, micro_list_test_stack, label_list_train_stack, label_list_test_stack
