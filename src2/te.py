#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def pairplot(df, **kargs):

    if kargs.has_key("hue"):
        label = df[kargs["hue"]]
        # axisいんのか？
        feature = df.drop(kargs["hue"], axis=1)
    else:
        # 面倒なので今のところエラーで
        raise
    feature_number = feature.shape[1]
    feature_names = feature.columns

    comb = [(x,y) for x in feature_names for y in feature_names]

    def hist(feature):

        lims = {fname:(min(feature[fname]), max(feature[fname])) for fname in feature}

    for i, two_feature_names in enumerate(comb):
        plt.subplot(feature_number, feature_number, i + 1)
        if two_feature_names[0] == two_feature_names[1]:
            plt.hist(feature[two_feature_names[0]])
            plt.title('A tale of 2 subplots')
            plt.ylabel('Damped oscillation')
        else:
            plt.scatter(*[feature[fname] for fname in two_feature_names])
            plt.xlabel('time (s)')
            plt.ylabel('Undamped')


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sns

    plt.hist([1,2,3])


    plt.pairplot = pairplot

    df = pd.DataFrame([[1, 2, 3, 'a'], [4, 5, 6, 'b'], [7, 8, 9, 'c']], columns=['a', 'b', 'c', 'label'])
    plt.pairplot(df, hue='label')
    plt.show()

    df = sns.load_dataset("iris")
    df = pd.DataFrame([[1, None, None, 'a'], [None, 3, 4, 'b'], [None, None, None, 'c']], columns=['a', 'b', 'c', 'label'])
    df = pd.DataFrame([[1, 2, 3, 'a'], [4, 5, 6, 'b'], [7, 8, 9, 'c']], columns=['a', 'b', 'c', 'label'])
    sns.pairplot(df, hue='label', dropna=True)
    sns.plt.show()
