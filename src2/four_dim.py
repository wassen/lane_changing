#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import math
from matplotlib import mlab
from math import sqrt
from matplotlib import pyplot as plt
import seaborn as sns
from math import log
from itertools import combinations
from os.path import join
import os

import matplotlib.pyplot as plt
import pandas as pd
import progressbar
import seaborn as sns
import numpy as np
import ddTools as dT
import repo_env
import plot_2d
from sklearn.decomposition import PCA

if __name__ == '__main__':

    if not os.path.exists(join(repo_env.DATA_DIR, "data_list")):
        print os.path.exists(join(repo_env.DATA_DIR, "data_list"))
        data_list_unnecessary = list(dT.load_each_lc(105))
        # 5フレームごと、前方車両の距離と相対速度だけ。
        saved_data = [data[::5][['front_center_distance', 'front_center_relvy']] for data in data_list_unnecessary]
        pd.to_pickle(saved_data, join(repo_env.DATA_DIR, "data_list"))
    else:
        saved_data = pd.read_pickle(join(repo_env.DATA_DIR, "data_list"))

    # first axis 各LCトライアル, second axis 時刻, third axis 特徴
    data_list = saved_data

    filtered_data_list = filter(
        lambda plot_data: plot_data.dropna().shape[0] == 105 / 5,
        data_list
    )
    print("105フレーム揃ってる車線変更データ数は{}個です".format(len(filtered_data_list)))


    def add_prev_diff(df):
        for i in df:
            n = [None]
            n.extend(df[:-1][i])
            df["prev_{}".format(i)] = n
            kwargs = {"diff_{}".format(i): lambda df: df[i] - df["prev_{}".format(i)]}
            df = df.assign(**kwargs)
        return df.dropna().set_index(np.arange(0, 100, 5))


    filtered_data_with_prev_diff_list = [
        add_prev_diff(data) for data in filtered_data_list
    ]
    data_list_diffs = [
        data[["diff_front_center_distance", "diff_front_center_relvy"]] for data in filtered_data_with_prev_diff_list
    ]

    # plot_2d.scatter_each_behavior2(data_list_diffs)
    # plot_2d.scatter_all_behavior2(data_list_diffs)

    data_list_add_prev = [
        data[["front_center_distance", "front_center_relvy", "prev_front_center_distance", "prev_front_center_relvy"]]
        for data in filtered_data_with_prev_diff_list
    ]

    # dataframeのリストをそのままnparrayに突っ込んでも3次元arrayにならない。（おそらくpandas3大クソ仕様のうちの一つの__iter__()をcolumnsにしているせい）
    filtered_3d_array = np.array([data.as_matrix() for data in data_list_add_prev])
    # first axis 時刻, second axis 各LCトライアル, third axis 特徴
    array_each_time = filtered_3d_array.transpose(1, 0, 2)
    # alltime_and_trial = filtered_3d_array.reshape()でもかわらずか？
    shape = array_each_time.shape
    alltime_and_trial = array_each_time.reshape(shape[0]*shape[1],shape[2])

    data = alltime_and_trial
    print(data.shape)
    # 主成分分析による次元削減
    pca = PCA(n_components=4)
    pca.fit(data)
    data_pca = pca.transform(data)
    print(pca.get_covariance())
    print(pca.explained_variance_ratio_)

    # 主成分分析後のサイズ
    print(data_pca.shape)