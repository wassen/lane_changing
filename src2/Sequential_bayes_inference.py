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
            # return np.arange(1.,21)[::-1]
            return np.ones(size) / size

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
        # print('a')

        train, test = delc.train_test_for_bayes()

        # これひとつのめソッドに
        data_3d = DataEachLC.dfinlist_to_nparray3d(train)
        train_2d = DataEachLC.nparray3d_to_2d(data_3d)

        import sklearn
        from sklearn.decomposition import PCA
        scaler = sklearn.preprocessing.MinMaxScaler()
        normalized_train_2d = scaler.fit_transform(train_2d)
        # trainの全サンプルでPCA
        pca = PCA(n_components=2)
        pca.fit(normalized_train_2d)
        # data_pca = pca.transform(all_time_and_lc)

        train_each_time = data_3d.transpose(1, 0, 2)

        # trainの各時刻にPCA適用、mean covを出す
        import numpy as np

        train_trans_list = [pca.transform(scaler.transform(train)) for train in train_each_time]
        mean_list = [np.mean(train_trans, axis=0) for train_trans in train_trans_list]
        cov_list = [np.cov(train_trans, rowvar=False, bias=0) for train_trans in train_trans_list]

        def process_for_figure():
            print(np.array(train_trans_list)[[0, 1, 10, 18, 19], 0, :])
            print(np.array(train_trans_list)[[0, 1, 10, 18, 19], 1, :])
            print(np.array(train_trans_list)[[0, 1, 10, 18, 19], 128, :])
            m = np.array(mean_list)
            c = np.array(cov_list)
            print(m[[0, 1, 10, 18, 19], :])
            print(c[[0, 1, 10, 18, 19], :])

            exit()

        # process_for_figure()

        gauss_list = [Bivariate_Gaussian(mean, cov) for mean, cov in zip(mean_list, cov_list)]
        # errors_list = []
        exps_weight_list = []
        exps_MAP_list = []
        exps_mle_list = []
        for i, tes in enumerate(test):
            tes_trans = pca.transform(scaler.transform(tes))
            size = 20
            be = GaussBayesEstimation(size, )
            exps_weight = []
            exps_MAP = []
            exps_mle = []
            for j, tes_tra in enumerate(tes_trans):
                # print(tes_tra)
                log_likelihoods = [gauss.log_likelihood(tes_tra) for gauss in gauss_list]
                likelihood = np.exp(log_likelihoods)  # 一応
                # print(likelihood)図用#
                be.update(log_likelihoods)
                # print(be.dist) #図用

                exp_weight = be.most_likely_time("weight")
                exp_MAP = be.most_likely_time("MAP")
                exp_mle = size - np.argmax(likelihood)

                exps_weight.append(exp_weight)
                exps_MAP.append(exp_MAP)
                exps_mle.append(exp_mle)

            exps_weight_list.append(exps_weight)
            exps_MAP_list.append(exps_MAP)
            exps_mle_list.append(exps_mle)
        frame_each_time = 2
        # HDDアクセスで結果を残して，他で処理って感じにしたい．30回繰り返すやつとグラフを書くやつをどうかき分けるか
        # np.save()
        return exps_weight_list, exps_MAP_list, exps_mle_list


    result_seq_lists = one_try()
    names = ("weight", "MAP", "mle")
    colors = ("#5EBABA", "#DBA946", "#F58E7E")
    import matplotlib.pyplot as plt

    # やっつけなのですごいみにくい
    fig0, axes0_2d = plt.subplots(2, 3, figsize=(9, 6), sharex=True, sharey=True)
    fig1, axes1_2d = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    hists = []
    for i, (result_seq_list, name, color) in enumerate(zip(result_seq_lists, names, colors)):
        # line plot
        ax = axes0_2d[0][i]
        ax.plot(range(-20, 0), range(-20, 0), color="#737373", linewidth=5)
        for result_seq in result_seq_list:
            x = range(-20, 0)
            y = - (np.array(result_seq))
            ax.plot(x, y, color=color)
        ax.set_xlim(-20, 0)
        ax.set_ylim(-20, 0)
        ax.set_title(name)

        # errorbar
        ax = axes0_2d[1][i]
        mean = - np.mean(result_seq_list, axis=0)
        std = np.std(result_seq_list, axis=0)
        ax.plot(range(-20, 0), range(-20, 0), color="#737373", linewidth=5)
        ax.set_xlim(-20, 0)
        ax.set_ylim(-20, 0)
        ax.errorbar(range(-20, 0), mean, yerr=std, color=color)

        # hist
        from itertools import chain

        s = list(sorted(set([0, 1, 2]) - set([i])))

        ax0 = axes1_2d[s[0]]
        ax1 = axes1_2d[s[1]]
        error_all = list(
            chain.from_iterable(
                [- np.array(result_seq) - np.arange(-20, 0) for result_seq in result_seq_list]
            )
        )

        hist = ax0.hist(error_all, color=color, alpha=0.5, bins=range(-20, 20), label=name)
        ax1.hist(error_all, color=color, alpha=0.5, bins=range(-20, 20), label=name)
        hists.append(hist)

    import repo_env

    # エラー イミフ maptlotlib ax share legendとかで調べたが
    # fig1.legend(hists, names)

    fig0.text(0.5, 0.025, r"Time_frame$\tau$", ha='center', va='center')  # , fontsize=18)
    fig0.text(0.04, 0.5, r"Predicted_time_frame$\hat{\tau}$", ha='center', va='center',
              rotation='vertical')  # , fontsize=18)  # 日本語？？？フォント設定
    fig0.savefig(repo_env.path("out", "result_line_error.pdf"))
    fig1.savefig(repo_env.path("out", "result_hist.pdf"))


    # 紙に手続き書いてから実装しよう。←これ子供に伝えたいことのあれに．5つ分割して、残り1つで結果出してみる。plot_interfaceのかぶりも消去

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
    # [ 5.58566667  5.45933333  5.24366667  4.98333333  4.69933333  4.40666667
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
