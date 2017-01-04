#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == '__main__':
    # いちいち読み込むのが時間かかる。
    # import sys
    # sys.argv

    # if sys.argv == "2d":
    import plot_2d
    # plot_2d.scatter_each_behavior('front_center_distance', 'front_center_relvy')
    # print('')
    # for com in plot_2d.get_feature_combinations():
    #     plot_2d.scatter_all_behavior(*com)
    # plot_2d.contours('front_center_distance', 'front_center_relvy')
    # plot_2d.scatter_all_behavior('front_center_distance', 'front_center_relvy')
    # plot_2d.scatter_each_time('front_center_distance', 'front_center_relvy')

    import ddTools as dT

    features = ["front_center_distance", "front_center_relvy"]
    diffs = ["diff_{}".format(feature) for feature in features]
    prevs = ["prev_{}".format(feature) for feature in features]

    delc = dT.DataEachLC(features=features)

    diffs_df_list = delc.extract_columns(diffs)

    # plot_2d.scatter_each_behavior(diffs_df_list)
    plot_2d.scatter_all_behavior(diffs_df_list)

    with_prevs = features + prevs
    with_prevs_df_list = delc.extract_columns(with_prevs)

    import numpy as np

    # dataframeのリストをそのままnparrayに突っ込んでも3次元arrayにならない。（おそらくpandas3大クソ仕様のうちの一つの__iter__()をcolumnsにしているせい）
    # second axis 各LCトライアル, first axis 時刻, third axis 特徴
    lc_time_feature_array = np.array([data.as_matrix() for data in with_prevs_df_list])
    shape = lc_time_feature_array.shape
    # 全時間と車線変更 × 特徴のペア
    all_time_and_lc = lc_time_feature_array.reshape(shape[0] * shape[1], shape[2])

    from sklearn.decomposition import PCA

    # 主成分分析による次元削減
    pca = PCA(n_components=2)
    pca.fit(all_time_and_lc)
    data_pca = pca.transform(all_time_and_lc)
    print(pca.get_covariance())
    print(pca.explained_variance_ratio_)
    # 主成分分析後のサイズ
    print(data_pca.shape)
    print(type(data_pca))

    import pandas as pd

    transformed_df_list = [pd.DataFrame(pca.transform(with_prev_df), columns=["pca_first", "pca_second"])
                           for with_prev_df in with_prevs_df_list
                           ]
    plot_2d.contours2(transformed_df_list)
    plot_2d.contours2(delc.extract_columns(features))