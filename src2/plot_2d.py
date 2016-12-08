#!/usr/bin/env python
# -*- coding: utf-8 -*-

import repo_env
from os.path import join
from os import listdir as ls
import numpy as np
import ddTools as dT
import pandas as pd
import matplotlib.pyplot as plt
z = [0]
green_dot, = plt.plot(z, "go", markersize=3)
red_dot, = plt.plot(z, "ro", markersize=3)
# plt.clf()
plt.close()

types = ('drv', 'roa', 'sur')


def load_right_divide_9000():

    def load(behavior, lc_series_num, type):
        return pd.read_pickle(join(repo_env.DATA_DIR, 'divide_9000', behavior, 'right', lc_series_num, type))

    l = []
    for behavior in sorted(ls(join(repo_env.DATA_DIR, 'divide_9000'))):
        for lc_series_num in sorted(ls(join(repo_env.DATA_DIR, 'divide_9000', behavior, 'right'))):
                types = ('drv', 'roa', 'sur')
                l.append({tYpe: load(behavior, lc_series_num, tYpe) for tYpe in types})
    return l

features = []
for j,data in enumerate(load_right_divide_9000()):
    start_index = dT.start_index(data['roa']['lc'])['right'][0]
    if start_index < 100:
        first_of_array = 0
        first_of_color_palette = 100 - start_index
    else:
        first_of_array = start_index - 100
        first_of_color_palette = 0

    drv_10_sec = data['drv'][first_of_array:start_index]
    roa_10_sec = data['roa'][first_of_array:start_index]
    sur_10_sec = dT.add_accel(data['sur'][first_of_array:start_index])
    print (len(drv_10_sec), len(roa_10_sec), len(sur_10_sec))

    columns = ['label',
               'gas', 'brake', 'steer',
               'front_center_distance','front_center_relvy', 'front_center_ttcy',
               'rear_right_distance', 'rear_right_ttcy', 'rear_right_relvy']
    if j == 3:
        pass

    for i, (drv, roa, sur) in enumerate(zip(drv_10_sec.iterrows(),roa_10_sec.iterrows(),sur_10_sec)):

        feature = []
        drv=drv[1]

        cars = dT.get_cars(sur)
        f_c_car = dT.specific_nearest_car(cars, dT.front_center)
        r_r_car = dT.specific_nearest_car(cars, dT.rear_right)

        def feature_if_exist(car, feature):
            if len(car) == 0:
                return None
            elif feature == 'vy':
                return car[3]
            else:
                return dT.to_feature(car, feature)

        feature.append("{}frame_before".format(100 - i))
        # 逆数取る特徴を定義
        feature.append(drv['gas'])
        feature.append(drv['brake'])
        feature.append(drv['steer'])
        feature.append(feature_if_exist(f_c_car, 'vy'))
        feature.append(feature_if_exist(r_r_car, 'vy'))
        feature.append(feature_if_exist(f_c_car, dT.Features.Distance))
        feature.append(feature_if_exist(r_r_car, dT.Features.Distance))

        f_c_car_ttcy = feature_if_exist(f_c_car, dT.Features.TimeToCollisionY)
        r_r_car_ttcy = feature_if_exist(r_r_car, dT.Features.TimeToCollisionY)

        feature.append(1/f_c_car_ttcy if r_r_car_ttcy is not None else None)
        feature.append(1/r_r_car_ttcy if r_r_car_ttcy is not None else None)

        features.append(feature)
    # dfdezenbumatomeru
    import seaborn as sns
    plot_data = pd.DataFrame(features, columns=columns)
    green_to_red = sns.diverging_palette(145, 10, n=100, center="dark")  # , s=70, l=40, n=3
    ax = sns.pairplot(pd.DataFrame(features, columns=columns), hue="label", palette=green_to_red[first_of_color_palette:])
    ax._legend.remove()

sns.plt.legend([green_dot, red_dot, ], ['100frame_before', '1frame_before'], bbox_to_anchor=(2, 1))
ax.savefig('tes')

