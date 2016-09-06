#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import listdir as ls
from os.path import join

import repo_env

class keySortDict(dict):
    def __init__(self, d):
        dict.__init__(self, d)

    def keys(self):
        return sorted(dict(self).keys())

    def values(self):
        return [self[key] for key in self.keys()]

def asdf():
    subjects = ls(repo_env.DATA_PATH_6000)
    return {subject: ls(join(repo_env.DATA_PATH_6000, subject)) for subject in subjects}
