import pickle
import numpy as np


def cross_validation_split(micro_list, label_list, niter):

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
