#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import math
from matplotlib import mlab
from math import sqrt
from matplotlib import pyplot as plt
import seaborn as sns
from math import log


class BayesEstimation():
    def __init__(self, size):
        # 一様分布
        self.dist = prior = np.ones(size) / size
        self.time = 0
        self.end_time = size

    def normalize(self):
        if not all([prob == 0 for prob in self.dist]):
            self.dist /= sum(self.dist)


    def update2(self, likelihoods):
        likelihoods = np.array(likelihoods)
        if self.time == 0:
            self.dist *= likelihoods
        else:
            cheat = [0]
            posterior = self.dist[:-1] * likelihoods[1:]
            self.dist = np.concatenate([cheat, posterior])
        self.normalize()
        self.time += 1

    def update(self, likelihoods):
        logli = np.log(likelihoods)
        if self.time == 0:
            self.dist = np.exp(np.log(self.dist) + logli)
        else:
            cheat = [0]
            posterior = np.exp(np.log(self.dist[:-1]) + logli[1:])
            self.dist = np.concatenate([cheat, posterior])
        self.normalize()
        self.time += 1


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


if __name__ == "__main__":

    from scipy.optimize import minimize
    from scipy.stats import multivariate_normal


    def opt_sample():

        def f(x):
            return (x[0] - 1) * (x[0] + 1) + (x[1] - 1) * (x[1] + 1)

        def f2(x):
            return f(x) * f(x)

        result = minimize(f2, x0=[-100, 0], method="SLSQP")

        # print(result)


    # x = np.linspace(0, 5, 10, endpoint=False)
    # l = multivariate_normal(mean=[1, -3], cov=[[1, 0.5], [0.5, 1]])
    #
    # print(l.pdf([[1, 2]]))
    #
    #
    def get_cov_matrix(x):
        cov = [[x[2], x[3]], [x[3], x[4]]]
        return ExtMatrix(cov)


    def log_self_bibariate(sample, x):
        mean = x[0:2]
        s_m = np.matrix(sample)
        mean_m = np.matrix(mean)
        cov_m = get_cov_matrix(x)
        diff = s_m - mean_m
        if any(cov_m.eig[0] <= 0):
            # どうせ制約式で弾かれるはずだけど、一応できるだけ小さい数を
            return float("-inf")

        numer = diff * cov_m.I * diff.T / (-2)
        denom = log(2. * math.pi * sqrt(cov_m.det))
        return numer - denom


    def cons(x):
        cov_m = get_cov_matrix(x)
        return min(cov_m.eig[0])


    # a = mlab.bivariate_normal(1, 2, mux=1, muy=-3, sigmax=1, sigmay=1, sigmaxy=0.5)
    # print(a)
    # a = self_bibariate([1, 2], [1, -3], cov=[[1, 0.5], [0.5, 1]])
    # print(a)

    def loglik(x):
        return sum([log_self_bibariate(s, x) for s in [[0, 1], [1, 0], [-1, 0], [0, -1]]])


    def mini(x):
        return - loglik(x)


    # cons = (
    #     {'type': 'ineq', 'fun': cons}
    # )
    # result = minimize(mini, x0=[4, 3, 1, 0, 1], constraints=cons, method="SLSQP")
    # print(result)

    # 逐次推定をlogで
    def bayes_update_sample():
        bs = BayesEstimation(10)
        print(bs.dist)
        bs.update([1, 2, 2, 4, 5, 6, 7, 8, 9, 10])
        print(bs.dist)
        bs.update([1, 2, 2, 4, 5, 6, 7, 8, 9, 10])
        print(bs.dist)
        bs.update([1, 2, 2, 4, 5, 6, 7, 8, 9, 10])
        print(bs.dist)
        bs.update([1, 2, 2, 4, 5, 6, 7, 8, 9, 10])
        print(bs.dist)
        bs.update([1, 2, 2, 4, 5, 6, 7, 8, 9, 10])
        print(bs.dist)
        bs.update([1, 2, 2, 4, 5, 6, 7, 8, 9, 10])
        print(bs.dist)
        bs.update([1, 2, 2, 4, 5, 6, 7, 8, 9, 10])
        print(bs.dist)
        bs.update([1, 2, 2, 4, 5, 6, 7, 8, 9, 10])
        print(bs.dist)
        bs.update([1, 2, 2, 4, 5, 6, 7, 8, 9, 10])
        print(bs.dist)
        bs.update([1, 2, 2, 4, 5, 6, 7, 8, 9, 10])
        print(bs.dist)
        bs.update([1, 2, 2, 4, 5, 6, 7, 8, 9, 10])
        print(bs.dist)
    bayes_update_sample()

    from matplotlib.patches import Ellipse



    def contour_sample():
        samples = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=(100))
        # lll = [[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [-1, -1]]

        m = np.mean(samples, axis=0)
        cov = np.cov(samples, rowvar=False)
        print("mean={},cov={}".format(m, cov))

        mux, muy = m

        def sigma_from(cov):
            return sqrt(cov[0][0]), sqrt(cov[1][1]), cov[0][1]

        sigmax, sigmay, sigmaxy = sigma_from(cov)
        # x = 0
        # y = 0
        # print(sigma_from(cov))
        # l = mlab.bivariate_normal(x, y, mux=mux, muy=muy, sigmax=sigmax, sigmay=sigmay, sigmaxy=sigmaxy)
        # print(lll)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        delta = 0.025
        x = np.arange(start=-3.0, stop=3.0, step=delta)
        y = np.arange(start=-3.0, stop=3.0, step=delta)
        X, Y = np.meshgrid(x, y)
        Z = mlab.bivariate_normal(X, Y, mux=mux, muy=muy, sigmax=sigmax, sigmay=sigmay, sigmaxy=sigmaxy)

        green_to_red = sns.diverging_palette(145, 10, n=5, center="dark")
        CS = ax.contour(X, Y, Z, levels=[0.005], alpha=1, linewidth=0.5, colors=[green_to_red[0]] * 1)

        # def get_angle(cov):
        #     val, vec = np.linalg.eig(cov)
        #     vec[0]

        # angle = np.rad2deg(np.arccos(vec[0, 0]))
        val, vec = np.linalg.eig(cov)
        print(cov)
        print(val)
        print(vec)
        def angle(v):
            first_vec = v[:, 0]
            # np.rad2deg(np.arccos(vec[0, 0]))90度ズレる
            # eig vecの返り値を勘違いしていた。
            return np.rad2deg(np.arctan(first_vec[1]/first_vec[0]))
        ell = Ellipse(xy=m, width=math.sqrt(val[0] * 2 * 2), height=math.sqrt(val[1] * 2 * 2), angle=angle(vec),
                      alpha=1, edgecolor=green_to_red[0], fill=False, linewidth=2)
        ax.add_patch(ell)
        ax.set_aspect('equal')
        ax.autoscale()

        # ax.scatter(*samples.T, alpha=0.5)
        ax.set_aspect('equal', adjustable='box')
        CS.clabel(fontsize=9, inline=1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plt.show()


    # contour_sample()
