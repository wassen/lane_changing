#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

REPOSITORY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
TMP_DIR = os.path.join(REPOSITORY_DIR, 'tmp')
DATA_DIR = os.path.join(REPOSITORY_DIR, 'data')
DATA_PATH_6000 = os.path.join(DATA_DIR, '6000')
DATA_PATH_9000 = os.path.join(DATA_DIR, '9000')
OUTPUT_DIR = os.path.join(REPOSITORY_DIR, 'out')
LIBRARY_DIR = os.path.join(REPOSITORY_DIR, 'lib')
SVM_TOOLS_DIR = os.path.join(LIBRARY_DIR, 'libsvm', 'tools')
