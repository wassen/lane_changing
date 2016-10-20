#!/usr/bin/env python
# -*- coding: utf-8 -*-

import repo_env
import os
from os.path import join

tool_path = join(repo_env.SVM_TOOLS_DIR, 'fselect_lin.py')
train_path = join(repo_env.DATA_DIR, 'train_for_svm.dat')
test_path = join(repo_env.DATA_DIR, 'test_for_svm.dat')

print(" ".join([tool_path, train_path, test_path]))

os.system(" ".join([tool_path, train_path, test_path]))

# fselectを改造
# PATHの設定実行時のディレクトリに出力するくっそお行儀の悪いスクリプトですわ。
