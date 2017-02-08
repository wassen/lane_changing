#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ddTools as dT
import plot_2d
import numpy as np
from sklearn.decomposition import PCA


if __name__ == '__main__':
    features = ["front_center_distance", "front_center_relvy"]
    delc = dT.DataEachLC(features=features)
    features_df_list = delc.extract_columns(features)

    # 2.2
    # plot_2d.scatter_and_ellipse_each_time(features_df_list)
    # 2.3
    plot_2d.ellipse_all_time(features_df_list)
    exit()
    with_prevs = features + delc.prevs
    features_df_list_with_prevs = delc.extract_columns(with_prevs)
    # データ変形 dataframeのリストをそのままnparrayに突っ込んでも3次元arrayにならない。（おそらくpandas3大クソ仕様のうちの一つの__iter__()をcolumnsにしているせい）
    data_3d_array = np.array([data.as_matrix() for data in features_df_list_with_prevs])
    dshape = data_3d_array.shape
    alltime_and_trial = data = data_3d_array.reshape(dshape[0] * dshape[1], dshape[2])
    # 正規化
    import sklearn

    scaler = sklearn.preprocessing.MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    # 主成分分析による次元削減
    pca = PCA(n_components=4)
    pca2 = PCA(n_components=2)
    pca.fit(normalized_data)
    pca2.fit(normalized_data)
    pca_data = pca.transform(normalized_data)

    # 表2.1
    # print(" & ".join(map(lambda x: "{:.2}".format(x), pca.explained_variance_ratio_)))
    # 表2.2
    # print "\\\\ \hline\n".join([' & '.join(map(lambda x: "{:.2}".format(x), compo)) for compo in pca.components_.T])
    #
    import pandas as pd

    transformed_df_list = [pd.DataFrame(pca2.transform(scaler.transform(df_with_prev)), columns=["pca_first", "pca_second"])
                           for df_with_prev in features_df_list_with_prevs
                           ]
    plot_2d.ellipse_all_time(transformed_df_list)

    # 一部Sequential_bayes_inferenceの中にあり
