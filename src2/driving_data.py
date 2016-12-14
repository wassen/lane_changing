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

def __get_cars(sur_row):
    sur_row = np.array(sur_row)
    cars = sur_row.reshape(int(sur_row.shape[0] / 4), 4).tolist()
    return filter(lambda car: not all([item == 0 for item in car]), cars)

def to_eachcar(sur):
    return [__get_cars(sur_row) for sur_row in sur]
# remoteで消えたのなぜ？
class Label():
    left_lanechanging = -2
    begin_left_lanechange = -1
    go_straight = 0
    begin_right_lanechange = 1
    right_lanechanging = 2
    braking_and_go_straight = 3
    gaspedal_and_go_straight = 4


def dividelabelwithbrakeandgas(label, brake, gas, threshold=5):
    '''
    直進ラベル(0)をブレーキ踏力のしきい値から0と3に分ける
    '''
    # TODO 二行に分ける必要なし。brakeかつアクセルとかいうイミフな状況は考えなくていいか。
    brake_filtered_label = [Label.braking_and_go_straight if b >= threshold and l == Label.go_straight else l for
                            l, b in
                            zip(label, brake)]
    return [Label.gaspedal_and_go_straight if g >= 2 and l == Label.go_straight else l for l, g in
            zip(brake_filtered_label, gas)]

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


class Info:
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


class DrvInfo(Info):
    type_name = "drv"
    names = ['time', 'brake', 'gas', 'vel', 'steer', 'accX', 'accY', 'accZ']
    drops = ['time', ]

    @classmethod
    def get_dataframe_from_csv(cls, drv_path):

        return Info \
            .get_dataframe_from_csv(drv_path) \
            .droplatlon() \
            .dropna(axis=1, how="all") \
            .rename_columns(cls.names) \
            .drop(cls.drops, axis=1)


class RoaInfo(Info):
    type_name = "roa"
    names = ['vel', 'host_lane', 'lc', 'host_rad', 'lane_number', 'FRT', 'BRT', 'D_fr(m)',
             'D_fl(m)', 'D_br(m)', 'D_bl(m)', 'Rad_fr(m)', 'CC_frx(m)', 'CC_fry(m)',
             'Rad_fl(m)', 'CC_flx(m)', 'CC_fly(m)', 'Rad_br(m)', 'CC_brx(m)', 'CC_bry(m)',
             'Rad_bl(m)', 'CC_blx(m)', 'CC_bly(m)']

    @classmethod
    def get_dataframe_from_csv(cls, roa_path):
        return Info \
            .get_dataframe_from_csv(roa_path) \
            .rename_columns(cls.names)


class SurInfo(Info):
    type_name = "sur"
    # nameは不定


def __get_3info_from_behavior(behavior, number_dir):
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
    behavior_and_drivingdata = {}
    behavior_list = subject_task_list(repo_env.DATA_PATH_9000)
    data_path = repo_env.DATA_PATH_9000

    for behavior in behavior_list:
        three_paths = __get_3info_from_behavior(behavior, data_path)
        info_list = (DrvInfo, RoaInfo, SurInfo)

        def dataframe_from(three_paths, info_list):
            """
            :param three_paths:
            :param info_list:
            :return {'drv':drv_npmat, 'roa':roa_npmat, 'sur':sur_npmat}:
            """
            # dataframes = {type_info.type_name: type_info.get_dataframe_from_csv(path)
            #         for path, type_info in zip(three_paths, info_list)}
            return {type_info.type_name: type_info.get_dataframe_from_csv(path)
                    for path, type_info in zip(three_paths, info_list)}

        def __equalize_size(df_dict):
            row_size = min([df_dict[t.type_name].shape[0] for t in info_list])
            return {t.type_name:df_dict[t.type_name][:row_size] for t in info_list}

        df_dict = dataframe_from(three_paths, info_list)

        behavior_and_drivingdata[behavior] = __equalize_size(df_dict)
    return behavior_list, behavior_and_drivingdata


if __name__ == '__main__':
    pass
    # expect = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2], [7, 8, 9, 0]]
    # actual = __get_cars([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 0, 0, 0, 0, 7, 8, 9, 0])
    # print(expect == actual)

else:
    # importするたび読み込んでたら使いにくい。小分けに読み込むとか、読み込むようのメソッド用意するとか。
    print("importing driving_data...")
    global behavior_list, behavior_and_drivingdata
    behavior_list, behavior_and_drivingdata = __read_csv()


    print('complete importing')
