#! /usr/bin/env python
# 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import repo_env
from os.path import join


DIR = '../srcpypy/test_for_svm.dat.210.pred'

with open(DIR) as file_handle:
    # saigonokuugyouwokesu filter(len)
    f = filter(len, file_handle.read().split('\n'))
    result = np.array([float(row) for row in f])
    
data = np.load(join(repo_env.DATA_DIR, 'train_test_feature_label.npz'))
label = data['l_test']

target_names = ['left_LC', 'straight', 'right_LC']

print(classification_report(label, result, target_names=target_names))

print(confusion_matrix(label, result))

print(precision_recall_fscore_support(label, result))

# jidoushuturyoku

