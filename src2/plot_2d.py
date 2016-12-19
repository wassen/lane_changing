#!/usr/bin/env python
# -*- coding: utf-8 -*-

import repo_env
from os.path import join
from os import listdir as ls
import numpy as np
import ddTools as dT
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# label間違ってた問題

# named tuple

types = ('drv', 'roa', 'sur')
columns = ['label',
           'gas', 'brake', 'steer',
           'front_center_distance','front_center_relvy', 'front_center_ittc_2ndy',
           'rear_right_distance', 'rear_right_relvy', 'rear_right_ittc_2ndy']
green_to_red = sns.diverging_palette(145, 10, n=100, center="dark")  # , s=70, l=40, n=3

def load_right_divide_9000():

    def load(behavior, lc_series_num, type):
        return pd.read_pickle(join(repo_env.DATA_DIR, 'divide_9000', behavior, 'right', lc_series_num, type))

    l = []
    for behavior in sorted(ls(join(repo_env.DATA_DIR, 'divide_9000'))):
        for lc_series_num in sorted(ls(join(repo_env.DATA_DIR, 'divide_9000', behavior, 'right'))):
                types = ('drv', 'roa', 'sur')
                l.append({tYpe: load(behavior, lc_series_num, tYpe) for tYpe in types})
    return l

pd_list=[]
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
    length = len(drv_10_sec)
    # print(length)
    features = []
    for i, (drv, roa, sur) in enumerate(zip(drv_10_sec.iterrows(),roa_10_sec.iterrows(),sur_10_sec)):

        #print(i)
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

        feature.append("{}frame_before".format(length - i))
        # 逆数取る特徴を定義
        feature.append(drv['gas'])
        feature.append(drv['brake'])
        feature.append(drv['steer'])
        feature.append(feature_if_exist(f_c_car, dT.Features.Distance))
        feature.append(feature_if_exist(f_c_car, 'vy'))

        f_c_car_ttcy = feature_if_exist(f_c_car, dT.Features.TimeToCollisionY)
        if f_c_car_ttcy == 0:
            f_c_car_ttcy = None

        r_r_car_ttcy = feature_if_exist(r_r_car, dT.Features.TimeToCollisionY)
        if r_r_car_ttcy == 0:
            r_r_car_ttcy = None

        feature.append(1/f_c_car_ttcy if f_c_car_ttcy is not None else None)
        feature.append(feature_if_exist(r_r_car, dT.Features.Distance))
        feature.append(feature_if_exist(r_r_car, 'vy'))
        feature.append(1/r_r_car_ttcy if r_r_car_ttcy is not None else None)
        # print(float('Nan') in feature)
        # print(float('Inf') in feature)
        features.append(feature)

# dfdezenbumatomeru

#, dropna=True
# Noneの点が消えるようになってるが、ヒストグラムのところでバグが出る。

    plot_data = pd.DataFrame(features, columns=columns)
    del(features)
    # green_to_red = sns.diverging_palette(145, 10, n=100, center="dark")  # , s=70, l=40, n=3
    # ax = sns.pairplot(pd.DataFrame(features, columns=columns), hue="label", palette=green_to_red[first_of_color_palette:])
    # ax._legend.remove()
    pd_list.append(plot_data)

pd.to_pickle(pd_list, 'tes')

# forにしないでもできるだろうけど、colorの指定がめんどくさそう
# [color for _ in pd_list for color in green_to_red]とかで[g_t_r, g_t_r,...]って並べたらいけるとおもう
# coms = combinations(columns[1:], 2)
# repo_env.make_dirs("out","all", exist_ok=True)
# for com in coms:
#     for plot_data in pd_list:
#         f0 = columns.index(com[0])
#         f1 = columns.index(com[1])
#         plot_features = plot_data[[f0, f1]].as_matrix().T
#         plt.scatter(*plot_features, color=green_to_red[- plot_features.shape[1]:])
#
#         plt.xlabel(com[0])
#         plt.ylabel(com[1])
#     plt.savefig(repo_env.path("out","all","scatter_{}_{}.png".format(com[0], com[1])))
#     plt.close()

# repo_env.make_dirs("out", exist_ok=True)
# # きょりと相対速度を位置車線変更ごとに個別

col1 = 'front_center_distance'
col2 = 'front_center_relvy'

col1_all_data = [val for plot_data in pd_list for val in plot_data[col1] if not val is None ]
col2_all_data = [val for plot_data in pd_list for val in plot_data[col2] if not val is None]

max1 = max(col1_all_data)
max2 = max(col2_all_data)
min1 = min(col1_all_data)
min2 = min(col2_all_data)

print(max1,
max2,
min1,
min2,
)


# for i, plot_data in enumerate(pd_list):
#     plot_features = plot_data[[col1, col2]].as_matrix().T
#     print(plot_features.shape)
#     plt.scatter(*plot_features, color=green_to_red[- plot_features.shape[1]:])
#     print('a')
#     # plt.xlim(min1, max1)
#     # plt.ylim(min2, max2)
#     plt.xlabel(col1)
#     plt.ylabel(col2)
#     repo_env.make_dirs("out", "{}_{}".format(col1, col2,), exist_ok=True)
#     # nameに汎用性を
#     plt.savefig(repo_env.path("out", "{}_{}".format(col1, col2,), "scatter_{}.png".format(i, )))
#     plt.close()

# 3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
repo_env.make_dirs("out", "3d", "{}_{}".format(col1, col2,), exist_ok=True)

for i, plot_data in enumerate(pd_list):
    z = [int(label_str[:label_str.find('f')]) for label_str in plot_data['label']]
    x, y = plot_features = plot_data[[col1, col2]].as_matrix().T
    x, y, z = pd.DataFrame([x,y,z]).T.dropna().as_matrix().T
    # ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, alpha=0.3, linewidth=0.5)
    # plt.xlim(min1, max1)
    # plt.ylim(min2, max2)
    # plt.xlabel(col1)
    # plt.ylabel(col2)
    # fig.savefig(repo_env.path("out", "3d", "{}_{}".format(col1, col2,), "{}.png".format(i)))
    # fig.delaxes(ax)
# fig.savefig(repo_env.path("out", "3d", "{}_{}".format(col1, col2,)))
plt.show()
# def to_hist_data(df_list, col):
#     number = 100
#     b = [[] for _ in range(number)]
#     for df_one_lc in df_list:
#         df_one_lc=df_one_lc[col]
#         for i, item in enumerate(df_one_lc):
#             if not (item is None or np.isnan(item)):
#                 b[i + number - len(df_one_lc)].append(item)
#     return b

# for col in columns[1:]:
#     print(col)
#     pd = to_hist_data(pd_list, col)
#     # for i, plot_data in enumerate(pd_list):
#     #     print(i)
#     #     pd_non_na = plot_data[col].dropna()
#     #     if pd_non_na.as_matrix().T.shape[0] == 0:
#     #         continue
#
#     plt.hist(pd, color=green_to_red[:], stacked=True)
#     plt.xlabel(col)
#     plt.savefig(repo_env.path("out","hist_{}.png".format(col)))
#     plt.close()
#
# plt.show()

# z = [0]
# green_dot, = plt.plot(z, "go", markersize=3)
# red_dot, = plt.plot(z, "ro", markersize=3)
# plt.close()
# sns.plt.legend([green_dot, red_dot, ], ['100frame_before', '1frame_before'], bbox_to_anchor=(2, 1))
# ax.savefig('tes')
# plt.savefig('teiis')
#
# seabornでエラー出るの、全部Nanのデータがあるからか？いや、全系列まとめてるから全部がNanってことはないと思うんだけど・・・