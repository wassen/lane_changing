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
