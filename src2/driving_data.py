#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import repo_env


def __get_cars(sur_row):
    sur_row = np.array(sur_row)
    cars = sur_row.reshape(int(sur_row.shape[0] / 4), 4).tolist()
    return filter(lambda car: not all([item == 0 for item in car]), cars)


def __to_eachcar(sur):
    return [__get_cars(sur_row) for sur_row in sur]


# こんな感じで9000とどっちも取得したいけど、9,000でカオスと化してるディレクトリ構成でできるか？
# for behavior in behaviors:
#     for info in infos:
#         path = __get_path(6000, drv, behavior, info)
#         pd.read_csv(path, ...)

def __read_6000():
    dataDicts = []

    tmpList = sorted(os.listdir(repo_env.DATA_PATH_6000))

    print('6000番台読込中')

    behavior_names = []
    for i, subject in enumerate(tmpList):
        for task in sorted(os.listdir(os.path.join(repo_env.DATA_PATH_6000, subject))):
            behavior_names.append(subject + task)
            drvDF = pd.read_csv(
                os.path.join(repo_env.DATA_PATH_6000, subject, task, subject + task + '-HostV_DrvInfo.csv'),
                encoding='shift-jis', header=0,
                names=['time', 'brake', 'gas', 'vel', 'steer', 'accX', 'accY', 'accZ', 'NaN'],
                dtype='float16')
            drvDF = drvDF.drop(['time', 'NaN'], axis=1)

            roaDF = pd.read_csv(
                os.path.join(repo_env.DATA_PATH_6000, subject, task, subject + task + '-HostV_RoadInfo.csv'),
                encoding='shift-jis', header=0,
                names=['host_v(km/h)', 'host_#L', 'LC', 'host_rad(m)', 'LN', 'FRT', 'BRT', 'D_fr(m)',
                       'D_fl(m)', 'D_br(m)', 'D_bl(m)', 'Rad_fr(m)', 'CC_frx(m)', 'CC_fry(m)',
                       'Rad_fl(m)', 'CC_flx(m)', 'CC_fly(m)', 'Rad_br(m)', 'CC_brx(m)', 'CC_bry(m)',
                       'Rad_bl(m)', 'CC_blx(m)', 'CC_bly(m)'],
                dtype={'LC': 'int8'})
            # roaDF = roaDF['LC']
            surDF = pd.read_csv(os.path.join(repo_env.DATA_PATH_6000, subject, task, subject +
                                             task + '-SurVehicleInfo.csv'), encoding='shift-jis', header=0,
                                dtype='float16')

            surDF = __to_eachcar(surDF.as_matrix())
            dataDict = {'drv': drvDF, 'roa': roaDF, 'sur': surDF}
            dataDicts.append(dataDict)
    return dataDicts, behavior_names


def __read_9000():
    dataDicts = []
    behavior_names=[]

    print('9000番台読込中')
    for i, item in enumerate(sorted(os.listdir(repo_env.DATA_PATH_9000))):
        for j, data in enumerate(sorted(os.listdir(os.path.join(repo_env.DATA_PATH_9000, item)))):
            if i == 0:
                behavior_names.append(data)
                drvDF = pd.read_csv(os.path.join(
                    repo_env.DATA_PATH_9000, item, data), encoding='shift-jis', header=0, dtype='float16')
                drvDF = drvDF.drop(
                    ['time[sec]', 'lat[deg]', 'lon[deg]'], axis=1)
                drvDF = drvDF.dropna()
                dataDicts.append({'drv': drvDF.as_matrix()})
            elif i == 1:
                roaDF = pd.read_csv(os.path.join(
                    repo_env.DATA_PATH_9000, item, data), encoding='shift-jis', header=0, dtype={'LC': 'int8'})
                # roaDF = roaDF['LC']
                dataDicts[j]['roa'] = roaDF.as_matrix()
            elif i == 2:
                surDF = pd.read_csv(os.path.join(
                    repo_env.DATA_PATH_9000, item, data), encoding='shift-jis', header=0, dtype='float16')
                dataDicts[j]['sur'] = surDF.as_matrix()
    return dataDicts, behavior_names


if __name__ == '__main__':
    expect = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2], [7, 8, 9, 0]]
    actual = __get_cars([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 0, 0, 0, 0, 7, 8, 9, 0])
    print(expect == actual)

else:
    global d6000, d9000, behavior_names6000, behavior_names9000
    file_path = os.path.join(repo_env.DATA_DIR, '69data.npz')
    if os.path.exists(file_path):
        load = np.load(file_path)
        d6000 = load['d6000']
        d9000 = load['d9000']
        behavior_names6000 = load['behavior_names6000']
        behavior_names9000 = load['behavior_names9000']
        load.close()
    else:
        d6000, behavior_names6000 = __read_6000()
        d9000, behavior_names9000 = __read_9000()
        np.savez(os.path.join(repo_env.DATA_DIR, '69data.npz'), d6000=d6000, d9000=d9000,
                 behavior_names6000=behavior_names6000, behavior_names9000=behavior_names9000)
