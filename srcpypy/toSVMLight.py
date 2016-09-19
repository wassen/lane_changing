#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import join
import numpy as np
import repo_env

import csv

csv.field_size_limit(10000000000)

keywords = ('f_train', 'l_train', 'f_test', 'l_test')

data = np.load(join(repo_env.DATA_DIR, 'train_test_feature_label.npz'))

f_train, l_train, f_test, l_test = [data[keyword] for keyword in keywords]

# svmlight_train = []
# total = l_train.shape[0]
# for i, (x, y) in enumerate(zip(fv_train, l_train)):
#     print('train', str(round(i / total * 100, 2)) + "%")
#     svmlight_train.append([str(y)])
#     for j, xj in enumerate(x):
#         svmlight_train[i].append(str(j + 1) + ":" + str(xj))
#
# with open("data/train_for_svm.csv", 'w') as f:
#     writer = csv.writer(f, lineterminator='\n', delimiter=' ')
#     writer.writerows(svmlight_train)
#
# svmlight_test = []
# total = l_test.shape[0]
# for i, (x, y) in enumerate(zip(fv_test, l_test)):
#     print('test', str(round(i / total * 100, 2)) + "%")
#     svmlight_test.append([str(y)])
#     for j, xj in enumerate(x):
#         svmlight_test[i].append(str(j + 1) + ":" + str(xj))
#
# with open("data/test_for_svm.dat", 'w') as f:
#     writer = csv.writer(f, lineterminator='\n', delimiter=' ')
#     writer.writerows(svmlight_test)

def toSVMLight2(file_name, X, y):
    import csv

    row = [[str(l)] + [str(j + 1) + ":" + str(fj) for j, fj in enumerate(f)] for (f, l) in zip(X, y)]

    print('complete converting')

    with open(file_name, 'w') as file_handler:
        c_writer = csv.writer(file_handler, lineterminator='\n', delimiter=' ')
        c_writer.writerows(row)


toSVMLight2(join(repo_env.DATA_DIR, "train_for_svm.dat"), f_train, l_train)
toSVMLight2(join(repo_env.DATA_DIR, "test_for_svm.dat"), f_test, l_test)
print(f_test.shape)
print(l_test.shape)
print(f_train.shape)
print(l_train.shape)
