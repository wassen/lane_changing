#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import combinations
from os import listdir as ls
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import progressbar
import seaborn as sns
import numpy as np
import ddTools as dT
import repo_env
from matplotlib.patches import Ellipse

# label間違ってた問題、影響ある？ seabornのpairplotはlabelで自動色分けできたけど、それ以外は影響ないはず。seabornも結局エラー祭りだったし。

types = ('drv', 'roa', 'sur')


class ExtMatrix(np.matrix):
    def __init__(self, matrix):
        np.matrix.__init__(matrix)
        self._det = None
        self._eig = None

    def __geteig(self):
        if self._eig is None:
            self._eig = np.linalg.eig(self)
        return self._eig

    def __seteig(self, value):
        raise

    def __deleig(self):
        del self._eig

    eig = property(__geteig, __seteig, __deleig)

    def __getdet(self):
        if self._det is None:
            self._det = np.linalg.det(self)
        return self._det

    def __setdet(self, value):
        raise

    def __deldet(self):
        del self._det

    det = property(__getdet, __setdet, __deldet)


# def get_lims(x_sample_name, y_sample_name):
#     x_all_samples = [val for data in data_list for val in data[x_sample_name] if val is not None]
#     y_all_samples = [val for data in data_list for val in data[y_sample_name] if val is not None]
#
#     xlims = (min(x_all_samples), max(x_all_samples))
#     ylims = (min(y_all_samples), max(y_all_samples))
#
#     return xlims, ylims


def get_lims(data_list2):
    x_all_samples = [val for data in data_list2 for val in data.ix[:, 0] if val is not None]
    y_all_samples = [val for data in data_list2 for val in data.ix[:, 1] if val is not None]

    xlims = (min(x_all_samples), max(x_all_samples))
    ylims = (min(y_all_samples), max(y_all_samples))

    return xlims, ylims



# getlim修正済み
def scatter_each_behavior(data_list2):
    xlims, ylims = get_lims(data_list2)

    fig = plt.figure()

    x_name = data_list2[0].columns[0]
    y_name = data_list2[0].columns[1]

    repo_env.make_dirs("out", "{}_{}".format(x_name, y_name), exist_ok=True)

    bar = progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        progressbar.Counter(),
        ' (', progressbar.ETA(), ') ',
    ])

    for i, plot_data in enumerate(bar(data_list2)):
        ax = fig.add_subplot(1, 1, 1)

        plot_samples = plot_data.as_matrix().T
        ax.scatter(*plot_samples, color=green_to_red_20)

        # ax.set_xlim(*xlims)
        # ax.set_ylim(*ylims)

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)

        fig.savefig(repo_env.path("out", "{}_{}".format(x_name, y_name), "scatter_{}.png".format(i, )))
        fig.clf()
    plt.close()

# getlim修正済み
def scatter_all_behavior(data_list2):
    xlims, ylims = get_lims(data_list2)
    # forにしないでもできるだろうけど、colorの指定がめんどくさそう
    # [color for _ in pd_list for color in green_to_red]とかで[g_t_r, g_t_r,...]って並べたらいけるとおもう

    x_name = data_list2[0].columns[0]
    y_name = data_list2[0].columns[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    repo_env.make_dirs("out", "all", exist_ok=True)

    data_3d_array = np.array([data.as_matrix() for data in data_list2])
    data_array_each_time = data_3d_array.transpose(1, 0, 2)
    for data, color in zip(data_array_each_time, green_to_red_20):
        # plot_samples = plot_data.as_matrix().T
        # ax.scatter(*plot_samples, color=green_to_red2)

        ax.scatter(*data.T, s=10, alpha=0.5, color=color)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 5)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    plt.savefig(repo_env.path("out", "all", "scatter_{}_{}.png".format(x_name, y_name)))
    plt.close()


def scatter_each_time(x_sample_name, y_sample_name):
    xlims, ylims = get_lims(x_sample_name, y_sample_name)
    fig = plt.figure()

    filtered_data_list = filter(
        lambda plot_data: plot_data[[x_sample_name, y_sample_name]].dropna().shape[0] == 100,
        data_list)

    data_panel = pd.Panel({i: d for i, d in enumerate(filtered_data_list)})
    data_panel = data_panel.transpose(1, 0, 2)
    for i, color in zip(data_panel, green_to_red):
        ax = fig.add_subplot(1, 1, 1)
        if i % 5 != 0:
            continue
        plot_data = data_panel[i][[x_sample_name, y_sample_name]]
        print(plot_data.shape)
        ax.scatter(*plot_data.as_matrix().T, color=color)
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        ax.set_title("{}sec_before".format((100 - i) / 10.))
        ax.set_xlabel(x_sample_name)
        ax.set_ylabel(y_sample_name)
        fig.savefig("{}.png".format(i))
        fig.clf()


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





def contours(data_list2):
    xlims, ylims = get_lims(data_list2)

    x_name = data_list2[0].columns[0]
    y_name = data_list2[0].columns[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    delta = 0.025
    # x = np.arange(start=xlims[0], stop=xlims[1], step=delta)
    # y = np.arange(start=ylims[0], stop=ylims[1], step=delta)
    x = np.arange(start=20, stop=80, step=delta)
    y = np.arange(start=-5, stop=5, step=delta)
    X, Y = np.meshgrid(x, y)
    from matplotlib import mlab
    from math import sqrt

    lc_time_feature_array = np.array([data.as_matrix() for data in data_list2])
    time_lc_feature_array = lc_time_feature_array.transpose(1, 0, 2)
    for lc_feature_array, color in zip(time_lc_feature_array, green_to_red_20):
        m = np.mean(lc_feature_array, axis=0)
        cov = np.cov(lc_feature_array, rowvar=False)

        # print("mean={},cov={}".format(m, cov))

        def sigma_from(cov):
            return sqrt(cov[0][0]), sqrt(cov[1][1]), cov[0][1]

        mux, muy = m
        sigmax, sigmay, sigmaxy = sigma_from(cov)
        Z = mlab.bivariate_normal(X, Y, mux=mux, muy=muy, sigmax=sigmax, sigmay=sigmay, sigmaxy=sigmaxy)
        # levels = [0.003]
        # CS = ax.contour(X, Y, Z, levels=levels, alpha=1, linewidth=0.5, colors=[color for _ in levels])
        # CS.clabel(fontsize=9, inline=1)

        # クソ
        # print(cov.dtype)
        # cov = np.array(cov, dtype=float)
        cov = ExtMatrix(cov)
        # 固有ベクトルの一行一列がcos\thetaになんのか・・・
        eig_val, eig_vec = cov.eig
        # array->mat->arrayでややこしいんやけど

        def angle(v):
            v = np.array(v)
            first_vec = v[:, 0]
            return np.rad2deg(np.arctan(first_vec[1] / first_vec[0]))

        ell = Ellipse(xy=m, width=sqrt(eig_val[0] * 2 * 2), height=sqrt(eig_val[1] * 2 * 2), angle=angle(eig_vec),
                      alpha=1, edgecolor=color, fill=False, linewidth=2)
        ax.add_patch(ell)

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)

    ax.autoscale()
    fig.savefig(repo_env.path("out", "contour_{}_{}.png".format(x_name, y_name, )))


# scatter aniumationとかぶってる
def contours_old(x_sample_name, y_sample_name):
    fig = plt.figure()

    # 100フレーム揃ってるやつだけ抽出。意図せず外れてしまっているやつを直したい。
    filtered_data_list = filter(lambda plot_data: plot_data[[x_sample_name, y_sample_name]].dropna().shape[0] == 100,
                                data_list)
    print(len(filtered_data_list))
    import numpy as np

    ax = fig.add_subplot(1, 1, 1)
    xlims, ylims = get_lims(x_sample_name, y_sample_name)

    delta = 0.025
    # x = np.arange(start=xlims[0], stop=xlims[1], step=delta)
    # y = np.arange(start=ylims[0], stop=ylims[1], step=delta)
    x = np.arange(start=20, stop=80, step=delta)
    y = np.arange(start=-5, stop=5, step=delta)
    X, Y = np.meshgrid(x, y)
    from matplotlib import mlab
    from math import sqrt

    data_panel = pd.Panel({i: d for i, d in enumerate(filtered_data_list)})
    data_panel = data_panel.transpose(1, 0, 2)
    for i, color in zip(data_panel, green_to_red):
        if i % 5 != 0:
            continue
        print(i)
        plot_data = data_panel[i]
        m = np.mean(plot_data[[x_sample_name, y_sample_name]], axis=0)
        cov = np.cov(plot_data[[x_sample_name, y_sample_name]], rowvar=False)

        # print("mean={},cov={}".format(m, cov))

        def sigma_from(cov):
            return sqrt(cov[0][0]), sqrt(cov[1][1]), cov[0][1]

        mux, muy = m
        sigmax, sigmay, sigmaxy = sigma_from(cov)
        Z = mlab.bivariate_normal(X, Y, mux=mux, muy=muy, sigmax=sigmax, sigmay=sigmay, sigmaxy=sigmaxy)
        print(cov)
        # levels = [0.003]
        # CS = ax.contour(X, Y, Z, levels=levels, alpha=1, linewidth=0.5, colors=[color for _ in levels])
        # CS.clabel(fontsize=9, inline=1)

        # クソ
        cov = np.array(cov, dtype=float)
        cov = ExtMatrix(cov)
        # 固有ベクトルの一行一列がcos\thetaになんのか・・・
        eig_val, eig_vec = cov.eig
        # array->mat->arrayでややこしいんやけど
        eig_vec = np.array(eig_vec)

        def angle(v):
            first_vec = v[:, 0]
            return np.rad2deg(np.arctan(first_vec[1] / first_vec[0]))

        ell = Ellipse(xy=m, width=sqrt(eig_val[0] * 2 * 2), height=sqrt(eig_val[1] * 2 * 2), angle=angle(eig_vec),
                      alpha=1, edgecolor=color, fill=False, linewidth=2)
        ax.add_patch(ell)
        # 色をもとから２０分割にしたい

        ax.set_xlabel(x_sample_name)
        ax.set_ylabel(y_sample_name)

    # fig.savefig(repo_env.path("out", "{}_{}".format(x_sample_name, y_sample_name, ), "scatter_{}.png".format(i, )))
    ax.autoscale()
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


if __name__ == "__main__":
    pass
else:
    global data_list, green_to_red, green_to_red_20
    # data_list = list(__load_data_list())
    green_to_red = sns.diverging_palette(145, 10, n=100, center="dark")  # , s=70, l=40, n=3
    green_to_red_20 = sns.diverging_palette(145, 10, n=20, center="dark")  # , s=70, l=40, n=3
