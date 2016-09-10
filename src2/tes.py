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

def subjecttask_list():
    subjects = sorted(ls(repo_env.DATA_PATH_6000))
    return [subject + task for subject in subjects for task in sorted(ls(join(repo_env.DATA_PATH_6000, subject)))]
