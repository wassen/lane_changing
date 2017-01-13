#!/usr/bin/env python
# -*- coding: utf-8 -*-
import plot_2d
import ddTools as dT
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

if __name__ == '__main__':

    features = ["front_center_distance", "front_center_relvy"]

    delc = dT.DataEachLC(features=features)


    def plot_features():
        features_df_list = delc.extract_columns(features)
        # plot_2d.scatter_all_behavior(features_df_list)
        # plot_2d.scatter_each_behavior(features_df_list)
        # plot_2d.contours(features_df_list)
        plot_2d.scatter_animation(features_df_list)

    plot_features()

    def plot_diffs():
        diffs_df_list = delc.extract_columns(delc.diffs)
        # plot_2d.scatter_each_behavior(diffs_df_list)
        plot_2d.scatter_all_behavior(diffs_df_list)

    # plot_diffs()

    def plot_prevs():
        with_prevs = features + delc.prevs
        with_prevs_df_list = delc.extract_columns(with_prevs)
        # dataframeのリストをそのままnparrayに突っ込んでも3次元arrayにならない。
        # （おそらくpandas3大クソ仕様のうちの一つの__iter__()をcolumnsにしているせい）
        # second axis 各LCトライアル, first axis 時刻, third axis 特徴
        lc_time_feature_array = np.array([data.as_matrix() for data in with_prevs_df_list])
        shape = lc_time_feature_array.shape
        # （全時間＋車線変更） × 特徴のペア
        all_time_and_lc = lc_time_feature_array.reshape(shape[0] * shape[1], shape[2])

        pca = PCA(n_components=2)
        pca.fit(all_time_and_lc)
        data_pca = pca.transform(all_time_and_lc)
        print(pca.explained_variance_ratio_)
        transformed_df_list = [pd.DataFrame(pca.transform(with_prev_df), columns=["pca_first", "pca_second"])
                               for with_prev_df in with_prevs_df_list
                               ]
        plot_2d.scatter_all_behavior(transformed_df_list)
        plot_2d.scatter_each_behavior(transformed_df_list)
        plot_2d.contours(transformed_df_list)
        # plot_2d.scatter_animation(transformed_df_list)

    # plot_prevs()

    def plot_others():
        # 各メソッドの仕様変更（特徴名を引数に→DataFrameのlistを引数に）により利用不可
        for com in plot_2d.get_feature_combinations():
            plot_2d.scatter_all_behavior(*com)
        plot_2d.contours('front_center_distance', 'front_center_relvy')
        plot_2d.scatter_all_behavior('front_center_distance', 'front_center_relvy')
        plot_2d.scatter_each_time('front_center_distance', 'front_center_relvy')
