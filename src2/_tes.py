#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
print(sys.version_info)

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


l = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,]


def start_index2(l):
    def onetotwo(n):
        if n == 0:
            return 0
        elif n == 1:
            return 2
        elif n == -1:
            return 1
        else:
            raise
    l = np.array(l)
    l = map(onetotwo, l)
    l = l - np.append(0,l[:-1])
    if l[0] == 0:
        l[0] = -10
    return {'str':  list(np.where(l < 0)[0]),
            'right':list(np.where(l == 2)[0]),
            'left': list(np.where(l == 1)[0]),}

# for c, b in zip(a, blist):
#     print(b, start_index2(c)['right'] == list(ddTools.start_index(c)['right']), start_index2(c)['left']== list(ddTools.start_index(c)['left']))

print(l)
print(start_index2(l))
exp = start_index2(l)
ans = {'str':[0,15,25,35,45], 'right':[10,30,40], 'left':[20]}

print(exp == ans)

def next_lc(num, start_index):

    def min_with_st(ite, st):
        l = filter(lambda x: x > st, ite)
        if len(l) == 0:
            return float("inf")
        else:
            return min(l)
    r = min_with_st(start_index['right'], num)
    l = min_with_st(start_index['left'], num)

    if r < l:
        return r, 'right'
    elif l < r:
        return l, 'left'
    else:
        return float('inf'), 'None'



for i in range(50):
    print(i, next_lc(i,ans))

'9000_str_to_right'
