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
        features = ["front_center_distance", "front_center_relvy"]

        delc = dT.DataEachLC(features=features)
        features_df_list = delc.extract_columns(features + delc.prevs)
        pd.to_pickle(features_df_list, join(repo_env.DATA_DIR, "data_list"))
    else:
        features_df_list = pd.read_pickle(join(repo_env.DATA_DIR, "data_list"))

    # dataframeのリストをそのままnparrayに突っ込んでも3次元arrayにならない。（おそらくpandas3大クソ仕様のうちの一つの__iter__()をcolumnsにしているせい）
    data_3d_array = np.array([data.as_matrix() for data in features_df_list])
    data_array_each_time = data_3d_array#.transpose(1, 0, 2)
    # ここメソッド化したいが．．．ddToolsか？

    shape = data_array_each_time.shape
    alltime_and_trial = data = data_array_each_time.reshape(shape[0] * shape[1], shape[2])

    import sklearn
    scaler = sklearn.preprocessing.MinMaxScaler()
    normalized_data = scaler.fit_transform(data)

    # 主成分分析による次元削減
    pca = PCA(n_components=4)
    pca.fit(normalized_data)
    pca_data = pca.transform(normalized_data)

    print(pca.components_)
    print("1:{:.2%}, 2:{:.2%}, 3:{:.2%}, 4:{:.2%} ".format(*pca.explained_variance_ratio_))
