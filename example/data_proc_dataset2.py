import pickle
import numpy as np


def cross_validation_split(micro_list, label_list):

    micro_list = np.array(micro_list)
    label_list = np.array(label_list)

    shuffle_order_carbn = np.random.choice(200, 200, replace=False)
    shuffle_order_pearl = np.random.choice(200, 200, replace=False) + 200
    shuffle_order_spher = np.random.choice(200, 200, replace=False) + 400

    micro_list_train_stack = np.zeros(([10, 540]))
    label_list_train_stack = np.zeros(([10, 540]))
    micro_list_ttest_stack = np.zeros(([10, 60]))
    label_list_ttest_stack = np.zeros(([10, 60]))

    for n in range(10):
        ttest_ind = np.concatenate((shuffle_order_carbn[n*20:(n+1)*20], shuffle_order_pearl[n*20:(n+1)*20],
                                    shuffle_order_spher[n*20:(n+1)*20]))
        label_list_ttest_stack[n, :] = label_list[ttest_ind]
        all_ind = range(600)
        train_ind = np.array([element for element in all_ind if element not in ttest_ind])
        label_list_train_stack[n, :] = label_list[train_ind]
        if n == 0:
            micro_list_ttest_stack = ttest_ind
            micro_list_train_stack = train_ind
        else:
            micro_list_ttest_stack = np.vstack((micro_list_ttest_stack, ttest_ind))
            micro_list_train_stack = np.vstack((micro_list_train_stack, train_ind))

    for n in range(10):
        shuffle_order_train = np.random.choice(540, 540, replace=False)
        shuffle_order_ttest = np.random.choice(60, 60, replace=False)
        micro_list_train_stack[n, :] = micro_list_train_stack[n, shuffle_order_train]
        label_list_train_stack[n, :] = label_list_train_stack[n, shuffle_order_train]
        micro_list_ttest_stack[n, :] = micro_list_ttest_stack[n, shuffle_order_ttest]
        label_list_ttest_stack[n, :] = label_list_ttest_stack[n, shuffle_order_ttest]

    return micro_list_train_stack, micro_list_ttest_stack, label_list_train_stack, label_list_ttest_stack
