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
from math import sqrt
import repo_env
from matplotlib.patches import Ellipse
import math

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


def get_filtered_data_list(data_list, x_sample_name, y_sample_name):
    return filter(
        lambda plot_data: plot_data[[x_sample_name, y_sample_name]].dropna().shape[0] == 100,
        data_list)


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


def __ellipse_with_mean_and_covariance(ax, mean, cov, color):
    cov = ExtMatrix(cov)
    eig_val, eig_vec = cov.eig

    # array->mat->arrayでややこしいんやけど
    def angle(v):
        v = np.array(v)
        first_vec = v[:, 0]
        return np.rad2deg(np.arctan(first_vec[1] / first_vec[0]))

    return Ellipse(xy=mean, width=sqrt(eig_val[0]) * 2, height=sqrt(eig_val[1]) * 2, angle=angle(eig_vec),
                   alpha=1, edgecolor=color, fill=False, linewidth=2)


def scatter_and_ellipse_each_time(data_list2):
    xlims, ylims = get_lims(data_list2)

    x_name = data_list2[0].columns[0]
    y_name = data_list2[0].columns[1]

    # numpyをextendしてcolumnつけたほうがマシじゃないか？pandas脱却したい．
    trial_time_feature_array = np.array([data.as_matrix() for data in data_list2])
    time_trial_feature_array = trial_time_feature_array.transpose(1, 0, 2)

    fig0, axes0_2d = plt.subplots(3, 2, figsize=(6, 9), sharex=True, sharey=True)
    fig1, axes1_2d = plt.subplots(3, 2, figsize=(6, 9), sharex=True, sharey=True)
    fig2, axes2_2d = plt.subplots(3, 2, figsize=(6, 9), sharex=True, sharey=True)
    fig3, axes3_2d = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)

    flat_axes0 = np.array(axes0_2d).reshape(6)
    flat_axes1 = np.array(axes1_2d).reshape(6)
    flat_axes2 = np.array(axes2_2d).reshape(6)
    flat_axes3 = np.array(axes3_2d).reshape(2)

    axes = list(flat_axes0) + list(flat_axes1) + list(flat_axes2) + list(flat_axes3)

    repo_env.make_dirs("out", "scatter_ellipse", exist_ok=True)
    for i, (trial_feature_array, color, ax) in enumerate(zip(time_trial_feature_array, green_to_red_20, axes)):
        # index_column = math.floor(i / 2.)
        # if index_column <= 4:
        #     axes_for_plot = axes0
        # else:
        #     axes_for_plot = axes1
        #     index_column -= 5
        # index_row = i % 2
        # print(axes0.shape)
        #
        # ax = axes_for_plot[index_column][index_row]

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)

        ax.scatter(*trial_feature_array.T, color=color, s=2)

        mean = np.mean(trial_feature_array, axis=0)
        cov = np.cov(trial_feature_array, rowvar=False)
        ellipse = __ellipse_with_mean_and_covariance(ax, mean, cov, color)

        ax.add_patch(ellipse)

        # ax.autoscale()
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        # ax.xaxis.set_tick_params(labelsize=13)
        # ax.yaxis.set_tick_params(labelsize=13)
        # ax.set_title(r"$\tau={}$".format((20 - i) / 2.))#, fontsize=18)
        ax.set_title(r"{} sec".format(- (20 - i) / 2.))#, fontsize=18)
        # コピペ
        # xlim = ax.get_xlim()
        # ylim = ax.get_ylim()
        # aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
        # ax.set_aspect(aspect)

        # fig.savefig(repo_env.path("out", "scatter_ellipse", "{}_{}{}.png".format(x_name, y_name, i)))
        # fig.clf()

    figs = (fig0, fig1, fig2, fig3)

    for i, fig in enumerate(figs):
        fig.text(0.5, 0.025, x_name, ha='center', va='center')  # , fontsize=18)
        fig.text(0.04, 0.5, y_name, ha='center', va='center', rotation='vertical')  # , fontsize=18)
        # fig.subplots_adjust(left=0.1, top=0.99, bottom=0.06)
        if i == 3:
            fig.subplots_adjust(bottom=0.2)
        fig.savefig(repo_env.path("out", "scatter_ellipse_{}_{}{}.pdf".format(x_name, y_name, i)))



def scatter_animation(data_list2):
    from matplotlib import animation

    x_name = data_list2[0].columns[0]
    y_name = data_list2[0].columns[1]

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)  # プロット消去せずとも前の結果が残らない不思議

    lc_time_feature_array = np.array([data.as_matrix() for data in data_list2])
    time_lc_feature_array = lc_time_feature_array.transpose(1, 0, 2)

    def gen_ims():
        for lc_feature_array, color in zip(time_lc_feature_array, green_to_red_20):
            im = ax.scatter(*lc_feature_array.T, color=color)
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            # x, y軸をグラフ間で自分で揃える必要はなし
            yield [im]

    ims = list(gen_ims())

    interval = int(10. / 20. * 1000.)
    ani = animation.ArtistAnimation(fig, ims, interval=interval, repeat=False)  # , interval=1, repeat_delay=1000

    repo_env.make_dirs("out", exist_ok=True)
    ani.save("{}_{}.gif".format(x_name, y_name), writer="imagemagick", )  # fps=2
    plt.show()


def ellipse_all_time(data_list2):
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
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    fig.savefig(repo_env.path("out", "ellipse_{}_{}.pdf".format(x_name, y_name, )))


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
