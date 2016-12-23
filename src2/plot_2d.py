#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import combinations
from os import listdir as ls
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import progressbar
import seaborn as sns

import ddTools as dT
import repo_env

# label間違ってた問題、影響ある？ seabornのpairplotはlabelで自動色分けできたけど、それ以外は影響ないはず。seabornも結局エラー祭りだったし。

types = ('drv', 'roa', 'sur')
columns = ('label',
           'gas', 'brake', 'steer',
           'front_center_distance', 'front_center_relvy', 'front_center_ittc_2ndy',
           'rear_right_distance', 'rear_right_relvy', 'rear_right_ittc_2ndy')



def __load_data_list():
    def load_right_divide_9000():

        def load(behavior, lc_series_num, type):
            return pd.read_pickle(join(repo_env.DATA_DIR, 'divide_9000', behavior, 'right', lc_series_num, type))

        type_df_dict_list = []
        for behavior in sorted(ls(join(repo_env.DATA_DIR, 'divide_9000'))):
            for lc_series_num in sorted(ls(join(repo_env.DATA_DIR, 'divide_9000', behavior, 'right'))):
                type_df_dict_list.append({tYpe: load(behavior, lc_series_num, tYpe) for tYpe in types})
        return type_df_dict_list

    for j, type_df_dict in enumerate(load_right_divide_9000()):
        start_index = dT.start_index(type_df_dict['roa']['lc'])['right'][0]
        if start_index < 100:
            first_of_array = 0
        else:
            first_of_array = start_index - 100

        drv_10_sec = type_df_dict['drv'][first_of_array:start_index]
        roa_10_sec = type_df_dict['roa'][first_of_array:start_index]
        sur_10_sec = dT.add_accel(type_df_dict['sur'][first_of_array:start_index])
        length = len(drv_10_sec)
        features = []
        for i, (drv, roa, sur) in enumerate(zip(drv_10_sec.iterrows(), roa_10_sec.iterrows(), sur_10_sec)):

            # print(i)
            feature = []
            drv = drv[1]

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

            feature.append(1 / f_c_car_ttcy if f_c_car_ttcy is not None else None)
            feature.append(feature_if_exist(r_r_car, dT.Features.Distance))
            feature.append(feature_if_exist(r_r_car, 'vy'))
            feature.append(1 / r_r_car_ttcy if r_r_car_ttcy is not None else None)
            features.append(feature)

            # pd.Series([
            #     length - i,
            #     drv['gas'],
            #     drv['brake'],
            #     drv['steer'],
            #     feature_if_exist(f_c_car, dT.Features.Distance),
            #     feature_if_exist(f_c_car, 'vy'),
            #     1 / f_c_car_ttcy,
            #     feature_if_exist(r_r_car, dT.Features.Distance),
            #     feature_if_exist(r_r_car, 'vy'),
            #     1 / r_r_car_ttcy,
            # ], index=columns)

            # , dropna=True
            # Noneの点が消えるようになってるが、ヒストグラムのところでバグが出る。

        plot_data = pd.DataFrame(features, columns=columns)
        del (features)
        # green_to_red = sns.diverging_palette(145, 10, n=100, center="dark")  # , s=70, l=40, n=3
        # ax = sns.pairplot(pd.DataFrame(features, columns=columns), hue="label", palette=green_to_red[first_of_color_palette:])
        # ax._legend.remove()
        yield plot_data


def get_feature_combinations():
    return combinations(columns[1:], 2)


def scatter_each_behavior(x_sample_name, y_sample_name):
    def get_lims():
        x_all_samples = [val for data in data_list for val in data[x_sample_name] if val is not None]
        y_all_samples = [val for data in data_list for val in data[y_sample_name] if val is not None]

        xlims = (min(x_all_samples), max(x_all_samples))
        ylims = (min(y_all_samples), max(y_all_samples))

        return xlims, ylims

    xlims, ylims = get_lims()

    fig = plt.figure()

    repo_env.make_dirs("out", "{}_{}".format(x_sample_name, y_sample_name, ), exist_ok=True)

    bar = progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        progressbar.Counter(),
        ' (', progressbar.ETA(), ') ',
    ])

    for i, plot_data in enumerate(bar(data_list)):
        ax = fig.add_subplot(1, 1, 1)

        plot_samples = plot_data[[x_sample_name, y_sample_name]].as_matrix().T
        ax.scatter(*plot_samples, color=green_to_red[-plot_samples.shape[1]:])

        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)

        ax.set_xlabel(x_sample_name)
        ax.set_ylabel(y_sample_name)

        fig.savefig(repo_env.path("out", "{}_{}".format(x_sample_name, y_sample_name, ), "scatter_{}.png".format(i, )))
        fig.clf()
    plt.close()


def scatter_all_behavior(x_sample_name, y_sample_name):
    # forにしないでもできるだろうけど、colorの指定がめんどくさそう
    # [color for _ in pd_list for color in green_to_red]とかで[g_t_r, g_t_r,...]って並べたらいけるとおもう

    fig = plt.figure()

    repo_env.make_dirs("out", "all", exist_ok=True)

    for plot_data in data_list:
        ax = fig.add_subplot(1, 1, 1)

        plot_samples = plot_data[[x_sample_name, y_sample_name]].as_matrix().T
        ax.scatter(*plot_samples, color=green_to_red[-plot_samples.shape[1]:])

        ax.set_xlabel(x_sample_name)
        ax.set_ylabel(y_sample_name)

    plt.savefig(repo_env.path("out", "all", "scatter_{}_{}.png".format(x_sample_name, y_sample_name)))
    plt.close()


def scatter_animation(x_sample_name, y_sample_name):
    from matplotlib import animation
    fig = plt.figure()

    # repo_env.make_dirs("out", "{}_{}".format(x_sample_name, y_sample_name, ), exist_ok=True)

    time_before_lane_changing = max([plot_data.shape[0] for plot_data in data_list])
    print(time_before_lane_changing)
    import numpy as np

    l = []
    # 割りとクソコード
    for i in range(time_before_lane_changing):
        g = []
        for data in data_list:
            if time_before_lane_changing - 1 - len(data) < i:
                g.append(data.ix[i + len(data) - time_before_lane_changing, [x_sample_name, y_sample_name]])
            else:
                g.append((np.nan, np.nan))
        l.append(np.array(g).transpose())
    # print(l)

    # plot_data_list = [
    #     np.array(
    #         [data.ix[i, [x_sample_name, y_sample_name]] for data in data_list]
    #     ).transpose()
    #     for i in range(time_before_lane_changing)
    #     ]

    ims = []
    ax = fig.add_subplot(1, 1, 1)

    for i, (plot_data, color) in enumerate(zip(l, green_to_red)):
        im = ax.scatter(*plot_data, color=color)
        ims.append([im])
        # ax.set_xlim(*xlims)
        # ax.set_ylim(*ylims)

        ax.set_xlabel(x_sample_name)
        ax.set_ylabel(y_sample_name)

    # fig.savefig(repo_env.path("out", "{}_{}".format(x_sample_name, y_sample_name, ), "scatter_{}.png".format(i, )))
    interval = int(1. / len(ims) * (time_before_lane_changing / 10) * 1000)
    ani = animation.ArtistAnimation(fig, ims, interval=100, repeat=False)  # , interval=1, repeat_delay=1000

    # 表示

    ani.save('a.gif', writer="imagemagick", fps=10)
    plt.show()



# repo_env.make_dirs("out", exist_ok=True)
# # きょりと相対速度を位置車線変更ごとに個別
#
#
# # 3d
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# repo_env.make_dirs("out", "3d", "{}_{}".format(col1, col2, ), exist_ok=True)
#
# for i, plot_data in enumerate(pd_list):
#     z = [int(label_str[:label_str.find('f')]) for label_str in plot_data['label']]
#     x, y = plot_features = plot_data[[col1, col2]].as_matrix().T
#     x, y, z = pd.DataFrame([x, y, z]).T.dropna().as_matrix().T
#     # ax = fig.add_subplot(111, projection='3d')
#     ax.plot(x, y, z, alpha=0.3, linewidth=0.5)
#     # plt.xlim(min1, max1)
#     # plt.ylim(min2, max2)
#     # plt.xlabel(col1)
#     # plt.ylabel(col2)
#     # fig.savefig(repo_env.path("out", "3d", "{}_{}".format(col1, col2,), "{}.png".format(i)))
#     # fig.delaxes(ax)
# # fig.savefig(repo_env.path("out", "3d", "{}_{}".format(col1, col2,)))
# plt.show()
#
# # def to_hist_data(df_list, col):
# #     number = 100
# #     b = [[] for _ in range(number)]
# #     for df_one_lc in df_list:
# #         df_one_lc=df_one_lc[col]
# #         for i, item in enumerate(df_one_lc):
# #             if not (item is None or np.isnan(item)):
# #                 b[i + number - len(df_one_lc)].append(item)
# #     return b
#
# # for col in columns[1:]:
# #     print(col)
# #     pd = to_hist_data(pd_list, col)
# #     # for i, plot_data in enumerate(pd_list):
# #     #     print(i)
# #     #     pd_non_na = plot_data[col].dropna()
# #     #     if pd_non_na.as_matrix().T.shape[0] == 0:
# #     #         continue
# #
# #     plt.hist(pd, color=green_to_red[:], stacked=True)
# #     plt.xlabel(col)
# #     plt.savefig(repo_env.path("out","hist_{}.png".format(col)))
# #     plt.close()
# #
# # plt.show()
#
# # z = [0]
# # green_dot, = plt.plot(z, "go", markersize=3)
# # red_dot, = plt.plot(z, "ro", markersize=3)
# # plt.close()
# # sns.plt.legend([green_dot, red_dot, ], ['100frame_before', '1frame_before'], bbox_to_anchor=(2, 1))
# # ax.savefig('tes')
# # plt.savefig('teiis')
# #
# # seabornでエラー出るの、全部Nanのデータがあるからか？いや、全系列まとめてるから全部がNanってことはないと思うんだけど・・・
#

if __name__ == "__main__":
    pass
else:
    global data_list, green_to_red
    data_list = list(__load_data_list())
    green_to_red = sns.diverging_palette(145, 10, n=100, center="dark")# , s=70, l=40, n=3