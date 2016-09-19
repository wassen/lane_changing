#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from os import listdir as ls
from os.path import join

import numpy as np
import pandas as pd
import repo_env


# define method
def rename_columns(self, name_list):
    return self.rename(columns={column: name for column, name in zip(self.columns, name_list)})
def droplatlon(self):
    try:
        return self.drop(['lat[deg]', 'lon[deg]'], axis=1)
    except:
        return self


pd.DataFrame.rename_columns = rename_columns
pd.DataFrame.droplatlon = droplatlon


# subject->被験者No(f6300等)
# task->被験者に割り当てられたタスク(nc5はタスクなし。mseは楽曲の検索。特に考慮に入れず)
# behavior ある被験者のひとつ分の運転行動。subject+task

# def __get_cars(sur_row):
#     sur_row = np.array(sur_row)
#     cars = sur_row.reshape(int(sur_row.shape[0] / 4), 4).tolist()
#     return filter(lambda car: not all([item == 0 for item in car]), cars)
#
#
# def __to_eachcar(sur):
#     return [__get_cars(sur_row) for sur_row in sur]


# こんな感じで9000とどっちも取得したいけど、9,000でカオスと化してるディレクトリ構成でできるか？
# for behavior in behaviors:
#     for info in infos:
#         path = __get_path(6000, drv, behavior, info)
#         pd.read_csv(path, ...)


def subject_task_list(data_dir):
    subjects = sorted(ls(data_dir))
    return [subject + task for subject in subjects for task in sorted(ls(join(data_dir, subject)))]


class DataType:
    drv = "HostV_DrvInfo.csv"
    roa = "HostV_RoadInfo.csv"
    sur = "SurVehicleInfo.csv"
    type_name_list = drv, roa, sur


class TypeInfo:
    # シングルトンで実装したい。がめんどい
    def __init__(self):
        pass

    @classmethod
    def get_dataframe_from_csv(cls, path):
        df = pd.read_csv(path,
                         encoding='shift-jis',
                         header=0,
                         # names=drv_names,
                         # dtype='float16'
                         )
        return df


class DrvTypeInfo(TypeInfo):
    type_name = "drv"
    names = ['time', 'brake', 'gas', 'vel', 'steer', 'accX', 'accY', 'accZ']
    drops = ['time', ]

    @classmethod
    def get_dataframe_from_csv(cls, drv_path):

        return TypeInfo \
            .get_dataframe_from_csv(drv_path) \
            .droplatlon() \
            .dropna(axis=1, how="all") \
            .rename_columns(cls.names) \
            .drop(cls.drops, axis=1)


class RoaTypeInfo(TypeInfo):
    type_name = "roa"
    names = ['vel', 'host_lane', 'lc', 'host_rad', 'lane_number', 'FRT', 'BRT', 'D_fr(m)',
             'D_fl(m)', 'D_br(m)', 'D_bl(m)', 'Rad_fr(m)', 'CC_frx(m)', 'CC_fry(m)',
             'Rad_fl(m)', 'CC_flx(m)', 'CC_fly(m)', 'Rad_br(m)', 'CC_brx(m)', 'CC_bry(m)',
             'Rad_bl(m)', 'CC_blx(m)', 'CC_bly(m)']

    @classmethod
    def get_dataframe_from_csv(cls, roa_path):
        return TypeInfo \
            .get_dataframe_from_csv(roa_path) \
            .rename_columns(cls.names)


class SurTypeInfo(TypeInfo):
    type_name = "sur"
    # nameは不定


def __get_3paths_from_behavior(behavior, number_dir):
    """

    :param behavior f6372nc5 etc:
    :param number_dir use DATA_PATH_6000 or DATA_PATH_9000 from repo_env:
    :return drv_path, roa_path, sur_path:
    """
    subject = re.search('^[fm]\d\d\d\d', behavior).group(0)
    task = re.search('(mse|nc\d|[AB]ex\d)$', behavior).group(0)
    dir = join(number_dir, subject, task)
    return [join(dir, (subject + task + "-" + typename)) for typename in DataType.type_name_list]


def __read_csv():
    global behavior_list, behavior_key_nparrays_value
    behavior6000_list = subject_task_list(repo_env.DATA_PATH_6000)
    behavior9000_list = subject_task_list(repo_env.DATA_PATH_9000)
    behavior_list = subject_task_list(repo_env.DATA_PATH_6000) + subject_task_list(repo_env.DATA_PATH_9000)
    data_path_list = [repo_env.DATA_PATH_6000 for _ in behavior6000_list] + [repo_env.DATA_PATH_9000 for _ in
                                                                             behavior9000_list]

    for data_path, behavior in zip(data_path_list, behavior_list):
        three_paths = __get_3paths_from_behavior(behavior, data_path)
        type_infos = (DrvTypeInfo, RoaTypeInfo, SurTypeInfo)

        def dataframe_from_paths_and_typeinfos(three_paths, type_infos):
            """
            :param three_paths:
            :param type_infos:
            :return {'drv':drv_npmat, 'roa':roa_npmat, 'sur':sur_npmat}:
            """
            # dataframes = {type_info.type_name: type_info.get_dataframe_from_csv(path)
            #         for path, type_info in zip(three_paths, type_infos)}
            return {type_info.type_name: type_info.get_dataframe_from_csv(path)
                    for path, type_info in zip(three_paths, type_infos)}

        def __equalize_size(df_dict):
            row_size = min([df_dict[t.type_name].shape[0] for t in type_infos])
            return {t.type_name:df_dict[t.type_name][:row_size] for t in type_infos}

        df_dict = dataframe_from_paths_and_typeinfos(three_paths, type_infos)

        behavior_key_nparrays_value[behavior] = __equalize_size(df_dict)


if __name__ == '__main__':
    pass
    # expect = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2], [7, 8, 9, 0]]
    # actual = __get_cars([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 0, 0, 0, 0, 7, 8, 9, 0])
    # print(expect == actual)

else:
    # importするたび読み込んでたら使いにくい。小分けに読み込むとか、読み込むようのメソッド用意するとか。
    print("importing driving_data...")
    global behavior_list, behavior_key_nparrays_value
    behavior_list = []
    behavior_key_nparrays_value = {}

    # 改めてreadcsvしたほうが断然早いが、pypyでやればなんとかなるか
    # file_path = os.path.join(repo_env.DATA_DIR, '69data.npz')

    # if os.path.exists(file_path):
    #     load = np.load(file_path)
    #     behavior_list = load['behavior_list']
    #     # pypyではここでバグる。dictをシリアライズできないっぽい感じ nijuuhairetudemomuridatta
    #     behavior_key_nparrays_value = load['behavior_key_nparrays_value']
    #     load.close()
    # else:
    # 後ほど変数に代入する形に変える
    __read_csv()
    # np.savez_compressed(file_path,
    #          behavior_list=behavior_list,
    #          behavior_key_nparrays_value=behavior_key_nparrays_value)

    print('complete importing')
