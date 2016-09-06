#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
import os
from os import listdir as ls
from os.path import join
import repo_env
import re

ls_in_9000 = ls(repo_env.DATA_PATH_9000)
mod_dir = join(repo_env.DATA_DIR, "Original", "9000_mod")

def copy_files_to_mod_dir(ls_in_kind, kind, kind_name):
    # re.searchでエラーが出るなら、DS_Store他ゴミファイルを削除する。
    for file in ls_in_kind:
        subject = re.search("[fm]96\d\d", file).group(0)
        task = re.search("[AB]\[?ex\d\]?", file).group(0)
        task = re.sub("[\[\]]", '', task)

        mod_sub_task_dir = join(mod_dir, subject, task)
        os.makedirs(mod_sub_task_dir, exist_ok=True)
        shutil.copy(join(repo_env.DATA_PATH_9000, kind, file),
                    join(mod_sub_task_dir, "{0}{1}-{2}.csv".format(subject, task, kind_name)))

for kind in ls_in_9000:
    if "Behavior" in kind:
        kind_name = "HostV_DrvInfo"
        ls_in_beh = ls(join(repo_env.DATA_PATH_9000, kind))
        copy_files_to_mod_dir(ls_in_beh, kind, kind_name)
    elif "Road" in kind:
        kind_name = "HostV_RoadInfo"
        ls_in_roa = ls(join(repo_env.DATA_PATH_9000, kind))
        copy_files_to_mod_dir(ls_in_roa, kind, kind_name)
    elif "Surround" in kind:
        kind_name = "SurVehicleInfo"
        ls_in_sur = ls(join(repo_env.DATA_PATH_9000, kind))
        copy_files_to_mod_dir(ls_in_sur, kind, kind_name)
    else:
        print("FOOk_the_DS_Store")


