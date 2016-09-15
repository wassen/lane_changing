#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import repo_env
import driving_data as dd

for behavior in dd.behavior_list:
    arrays = dd.behavior_key_nparrays_value[behavior]
    print([(array.shape, type(array)) for array in arrays.values()])