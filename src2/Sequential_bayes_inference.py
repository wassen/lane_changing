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

        denom = log(2. * pi * sqrt(np.linalg.det(self.cov)))
        return numer[0, 0] - denom


class GaussBayesEstimation:

    def normalize(self):
        if not all([prob == 0 for prob in self.dist]):
            self.dist /= sum(self.dist)

    def __init__(self, size):
        # 一様分布
        def get_prior():
            return np.arange(1.,21)[::-1]
            # return np.ones(size) / size
        self.dist = get_prior()
        self.normalize()
        self.time = 0
        self.size = size

    def most_likely_time(self, method="weight"):
        if method == "weight":
            weight = np.array(self.dist)
            time_array = np.linspace(start=self.size, stop=1, num=self.size)
            return np.dot(weight, time_array)
        elif method == "MAP":
            return self.size - np.argmax(self.dist)



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


    def one_try():

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
        cov_list = [np.cov(train_trans, rowvar=False, bias=0) for train_trans in train_trans_list]

        gauss_list = [Bivariate_Gaussian(mean, cov) for mean, cov in zip(mean_list, cov_list)]
        errors_list = []
        exps_list= []
        for i, tes in enumerate(test):
            # print("case{}".format(i))
            tes_trans = pca.transform(tes)
            size = 20
            be = GaussBayesEstimation(size, )
            errors = []
            exps=[]
            for j, tes_tra in enumerate(tes_trans):
                log_likelihoods = [gauss.log_likelihood(tes_tra) for gauss in gauss_list]
                be.update(log_likelihoods)
                act = size - j
                exp = be.most_likely_time("weight")
                # print("act:{0}, pred:{1}".format(act, exp))
                errors.append((act - exp)**2)
                exps.append(exp)
            errors_list.append(errors)
            exps_list.append(exps)
        frame_each_time = 2
        return [round(np.average(exps) / frame_each_time, 2) for exps in np.array(exps_list).T]

    result = [one_try() for _ in range(30)]
    print(np.average(result, axis=0))

# 紙に手続き書いてから実装しよう。5つ分割して、残り1つで結果出してみる。plot_interfaceのかぶりも消去

# 池田先生

# MAPと聞いてピンときていませんでしたが、以前までやっていた「確率が高いものを結果として出す」ものがそれに当たるのでしょうか。
#
# 勘違いであったなら申し訳ありませんが、その時の平均推定結果を出しましたのでお伝えします。
# また、テストとトレーニングの選び方によってばらつきが大きかったので、100回分の平均を出しました。
# [ 6.415       6.09        5.74466667  5.389       5.07233333  4.732
#   4.40033333  4.052       3.72166667  3.37933333  3.05        2.727       2.416
#   2.10833333  1.815       1.51533333  1.23466667  0.98233333  0.73933333
#   0.5       ]
# [ 6.29933333  5.97166667  5.62266667  5.27366667  4.93        4.62066667
#   4.26933333  3.92866667  3.59866667  3.25233333  2.92233333  2.608
#   2.31533333  2.019       1.73466667  1.473       1.219       0.98433333
#   0.74        0.5       ]
# weight
#[ 5.58566667  5.45933333  5.24366667  4.98333333  4.69933333  4.40666667
  # 4.11366667  3.81633333  3.515       3.214       2.91666667  2.623
  # 2.33333333  2.051       1.776       1.50966667  1.24966667  0.99666667
  # 0.74633333  0.5       ]
  #   [5.55966667  5.422       5.2         4.93966667  4.65733333  4.36833333
  #    4.07633333  3.78466667  3.492       3.19833333  2.90766667  2.62033333
  #    2.334       2.055       1.78033333  1.51333333  1.25266667  0.99866667
  #    0.748       0.5]




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
