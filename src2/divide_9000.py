#!/usr/bin/env python
# -*- coding: utf-8 -*-

import driving_data as dd
import ddTools
import repo_env
import os
import numpy as np
from os.path import join
repo_env.make_dirs('data','divide_9000', exist_ok=True)

for behavior in dd.behavior_list:

    lc = dd.behavior_and_drivingdata[behavior]['roa']['lc']
    start_indexes = ddTools.start_index(lc)
    r_count = 0
    l_count = 0

    for i in range(len(start_indexes['str'])):
        start_index_str = start_indexes['str'][i]
        if i == len(start_indexes['str']) - 1:
            next_start_index_str = 9999999999
        else:
            next_start_index_str = start_indexes['str'][i+1]

        if ddTools.next_lc(start_index_str, start_indexes)[1] == 'right':
            repo_env.make_dirs('data', 'divide_9000', behavior, 'right', str(r_count) , exist_ok=True)
            for tYpe in ('drv','roa','sur'):
                dd.behavior_and_drivingdata[behavior][tYpe][start_index_str:next_start_index_str].to_pickle(
                    join(repo_env.DATA_DIR, 'divide_9000', behavior, 'right', str(r_count), tYpe),
                )
            r_count += 1
        elif ddTools.next_lc(start_index_str, start_indexes)[1] == 'left':
            repo_env.make_dirs('data', 'divide_9000', behavior, 'left', str(l_count), exist_ok=True)
            for tYpe in ('drv', 'roa', 'sur'):
                dd.behavior_and_drivingdata[behavior][tYpe][start_index_str:next_start_index_str].to_pickle(
                    join(repo_env.DATA_DIR, 'divide_9000', behavior, 'left', str(l_count), tYpe),
                )
            l_count += 1

