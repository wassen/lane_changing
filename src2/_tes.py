#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import listdir as ls
from os.path import join

import repo_env
#import driving_data as dd

import numpy as np
import os
# np.save(os.path.join(repo_env.TMP_DIR, 'lc'),
#        [dd.behavior_and_drivingdata[behavior]['roa']['lc'] for behavior in dd.behavior_list])
# np.save(os.path.join(repo_env.TMP_DIR, 'blist'), dd.behavior_list)
a = np.load(os.path.join(repo_env.TMP_DIR, 'lc.npy'))
blist = np.load(os.path.join(repo_env.TMP_DIR, 'blist.npy'))
import ddTools

for c, b in zip(a, blist):
    print(b, len(ddTools.start_index(c)['right']), len(ddTools.start_index(c)['left']))

import csv

with open('some.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for c, b in zip(a, blist):
        print(b, len(ddTools.start_index(c)['right']), len(ddTools.start_index(c)['left']))
        l = [b, len(ddTools.start_index(c)['right']), len(ddTools.start_index(c)['left'])] 
        writer.writerow(l)
