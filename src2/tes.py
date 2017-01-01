#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA

if __name__ == "__main__":

    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # 主成分分析前のサイズ
    print(X.shape)

    # 主成分分析による次元削減
    pca = PCA(n_components = 2)
    pca.fit(X)
    X_pca= pca.transform(X)

    # 主成分分析後のサイズ
    print(X_pca.shape)
