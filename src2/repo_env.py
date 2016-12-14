#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

REPOSITORY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
DATA_PATH_6000 = os.path.join(REPOSITORY_DIR, 'data/Original/6000/')
DATA_PATH_9000 = os.path.join(REPOSITORY_DIR, 'data/Original/9000/')
TMP_DIR = os.path.join(REPOSITORY_DIR, 'tmp')
DATA_DIR = os.path.join(REPOSITORY_DIR, 'data')
LIBRARY_DIR = os.path.join(REPOSITORY_DIR, 'lib')
SVM_TOOLS_DIR = os.path.join(LIBRARY_DIR, 'libsvm', 'tools')
OLD_9000_DIR = os.path.join(DATA_DIR, "Original", "9000_old")
MOD_9000_DIR = os.path.join(DATA_DIR, "Original", "9000_mod")



def path(*paths):
    return os.path.join(REPOSITORY_DIR, *paths)

# これexistOK指定しなかったらどうなんだ、第一引数はどっちで処理されるんだ
# 実際のExist_OKの実装を参考にしよう
def make_dirs(*dir,**kwargs):

    if len(dir) == 0:
        raise
    if 'exist_ok' in kwargs:
        exist_ok = kwargs['exist_ok']
    else:
        exist_ok = False
    dir_path = os.path.join(REPOSITORY_DIR, *dir)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif exist_ok == True:
        return
    else:
        os.makedirs(dir_path)
