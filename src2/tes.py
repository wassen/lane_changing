#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import listdir as ls
from os.path import join

import repo_env


class KeySortDict(dict):
    def __init__(self, d):
        dict.__init__(self, d)

    def keys(self):
        return sorted(dict(self).keys())

    def values(self):
        return [self[key] for key in self.keys()]


# def subjecttask_dict():
#     subjects = ls(repo_env.DATA_PATH_6000)
#     return {subject: ls(join(repo_env.DATA_PATH_6000, subject)) for subject in subjects}

# def subjecttask_list():
#     subjects = sorted(ls(repo_env.DATA_PATH_6000))
#     return [subject + task for subject in subjects for task in sorted(ls(join(repo_env.DATA_PATH_6000, subject)))]
#
# def sur_to_cars

import constants as C

# lc = [0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,-1,-1,-1,-1,-1,-1,0,0,1,1,0,1,-1,0,0,]
# r_count = 0
# l_count = 0
# del_index = []
# for i, lc in enumerate(lc):
#
#     if lc == 0:
#         r_count = 0
#         l_count = 0
#     elif lc == 1:
#         r_count += 1
#
#     if r_count > 3:
#         l_count = 0
#         del_index.append(i)
#     elif lc == -1:
#         l_count += 1
#
#     if l_count > 3:
#         r_count = 0
#         del_index.append(i)
#     else:
#         print('lcに0,1,-1の文字が含まれています。')
#
# print(del_index)
# # 差集合(残すindex)を計算してnumpyで扱えるリスト化
# LC_use_index = list(set(range(len(LC))) - set(del_index))
# # TODO よく考えてみると、変数名の規則に則ってないな。
# LC = LC[LC_use_index]
# dt30 = dt30[LC_use_index]
#
lc = [0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,-1,-1,-1,-1,-1,-1,0,0,1,1,0,1,-1,1,-1,1,-1,1,1,-1,1,0,0,]
rlc_state = 0
llc_state = 0
del_index = []
for i, lc in enumerate(lc):

    if lc == 0:
        rlc_state = 0
        llc_state = 0
    elif lc == 1:
        rlc_state += 1
        llc_state = 0
    elif lc == -1:
        llc_state += 1
        rlc_state = 0
    else:
        print('lcに0,1,-1の文字が含まれています。')
    if rlc_state > 3:
        del_index.append(i)
    elif llc_state > 3:
        del_index.append(i)


print(del_index)


#roadDataと7フレームずらして対応付け
# >---PRED = 7
# >---LC = LC[PRED:]
# >---dt30 = dt30[:-1*PRED]
# >---#zantei 0の配列を足してから消す方法とか、
# >---if not X.size:
# >--->---X = dt30
# >--->---Y = LC
# >---else:
# >--->---X = np.r_[X,dt30]
# >--->---Y = np.r_[Y,LC]
#
# if(X.shape[0] != Y.shape[0]):
# >---print('サンプル数が一致していません')
#
# np.save('data/fv.npy',X)
# np.save('data/l.npy',Y)
