# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import pi
from math import sqrt
from numpy import log


class Bivariate_Gaussian:
    def __init__(self, mean, cov):
        self.mean = np.matrix(mean)
        self.cov = np.matrix(cov)

    def log_likelihood(self, sample):
        s_v = np.matrix(sample)

        diff = s_v - self.mean

        numer = diff * self.cov.I * diff.T / (-2)

        denom = log(2. * pi * sqrt(np.linalg.det(cov)))
        return numer[0, 0] - denom


class GaussBayesEstimation:
    def __init__(self, size):
        # 一様分布
        self.dist = prior = np.ones(size) / size
        self.time = 0
        self.size = size

    def most_likely_time(self):
        return self.size - np.argmax(self.dist)

    def normalize(self):
        if not all([prob == 0 for prob in self.dist]):
            self.dist /= sum(self.dist)

    # def update2(self, likelihoods):
    #     likelihoods = np.array(likelihoods)
    #     if self.time == 0:
    #         self.dist *= likelihoods
    #     else:
    #         cheat = [0]
    #         posterior = self.dist[:-1] * likelihoods[1:]
    #         self.dist = np.concatenate([cheat, posterior])
    #     self.normalize()
    #     self.time += 1

    def update(self, log_likelihoods):
        log_likelihoods = np.array(log_likelihoods)
        if self.time == 0:
            self.dist = np.exp(np.log(self.dist) + log_likelihoods)
        else:
            cheat = [0]
            posterior = np.exp(np.log(self.dist[:-1]) + log_likelihoods[1:])
            self.dist = np.concatenate([cheat, posterior])
        self.normalize()
        self.time += 1


if __name__ == '__main__':
    import ddTools as dT
    from ddTools import DataEachLC

    features = ["front_center_distance", "front_center_relvy"]

    delc = DataEachLC(features=features)
    with_prevs = delc.features + delc.prevs
    delc.extract_columns(with_prevs)

    train, test = delc.train_test_for_bayes()

    # これひとつのめソッドに
    data_3d = DataEachLC.dfinlist_to_nparray3d(train)
    train_2d = DataEachLC.nparray3d_to_2d(data_3d)

    # trainの全サンプルでPCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca.fit(train_2d)
    # data_pca = pca.transform(all_time_and_lc)

    train_each_time = data_3d.transpose(1, 0, 2)

    # trainの各時刻にPCA適用、mean covを出す
    import numpy as np

    train_trans_list = [pca.transform(train) for train in train_each_time]
    mean_list = [np.mean(train_trans, axis=0) for train_trans in train_trans_list]
    cov_list = [np.cov(train_trans, rowvar=False) for train_trans in train_trans_list]

    gauss_list = [Bivariate_Gaussian(mean, cov) for mean, cov in zip(mean_list, cov_list)]
    for i, tes in enumerate(test):
        print("case{}".format(i))
        tes_trans = pca.transform(tes)
        size = 20
        be = GaussBayesEstimation(size, )
        for j, tes_tra in enumerate(tes_trans):
            log_likelihoods = [gauss.log_likelihood(tes_tra) for gauss in gauss_list]
            be.update(log_likelihoods)
            print("act:{0}, pred:{1}".format(size - j, be.most_likely_time()))

# 紙に手続き書いてから実装しよう。5つ分割して、残り1つで結果出してみる。plot_interfaceのかぶりも消去




# naibu トライアルごとにshuffle
# random_divide(5)
# data = self.data
# import numpy as np
# size = len(data)
# shuffled_data = np.random.permutation(data)
# size / percentage size % percentage?????
# for i in range(num):
#     int(size/num)
#     yield data[sizetokanumtoka]
# def cv_data
# return 5bunnkatu

# 理想の形

# for comb in dT.DataEachLC.data(5):
#     train = comb[0]
#     test = comb[1]  # ここのtestをiterableにしなあかんのか
#     # train = concat_all(train)# reshape? flatten?これは内部でやっとく
#     mean, cov = mean_cov(train)
#
#     for tes in test:
#         be = GaussBayesEstimation(mean, cov, 20, )
#         for i in tes:
#             be.update(tes[i])
#             print(i, be.most_likely_time)
#         c
#             # likelihood = gaussian( ,mean, cov)naibu
