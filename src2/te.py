#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def pairplot(df, **kargs):

    fig = plt.figure()

    if kargs.has_key("hue"):
        label = df[kargs["hue"]]
        # axisいんのか？
        feature = df.drop(kargs["hue"], axis=1)
    else:
        # 面倒なので今のところエラーで
        raise
    feature_number = feature.shape[1]
    feature_names = feature.columns

    comb = [(x,y) for y in feature for x in feature]

    lims = {fname: (min(feature[fname]), max(feature[fname])) for fname in feature}

    # fig, subplots_2d = plt.subplots(nrows=feature_number, ncols=feature_number, sharex=True, sharey=True)
    #
    # from itertools import chain
    # subplots = list(chain.from_iterable(subplots_2d))

    sp_list=[]
    scatter_flag=[False for _ in feature]
    for i, fnames in enumerate(comb):
        # sp_number = feature_number
        # int(i/sp_number)
        # i%sp_number

        sp_row = list(feature_names).index(fnames[1])
        sp_col = list(feature_names).index(fnames[0])

        kwargs_sp = {}
        if sp_row > 0:
            kwargs_sp['sharex'] = sp_list[sp_col]
        if sp_row != sp_col:
            if scatter_flag[sp_row]:
                kwargs_sp['sharey'] = sp_list[feature_number * sp_row + 0 if sp_row > 0 else 1]
            else:
                scatter_flag[sp_row] = True

        sp = fig.add_subplot(feature_number, feature_number, i + 1, **kwargs_sp)
        sp_list.append(sp)
        # sp.set_xlim(lims[fnames[0]])
        # sp.set_ylim(lims[fnames[1]])
        if sp_row == sp_col:
            sp.hist(feature[fnames[0]])
            # sp.title = 'A tale of 2 subplots'
            # sp.ylabel('Damped oscillation')
        else:
            sp.scatter(*[feature[fname] for fname in fnames])
            # sp.xlabel('time (s)')
            # sp.ylabel('Undamped')
    return fig

if __name__ == "__main__":

    from matplotlib import pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sns

    # plt.hist([1,2,3])

    df = pd.DataFrame([[1, 2, 3, 'f'], [6, 5, 4, 'g'], [7, 8, 9, 'h']], columns=['a', 'b', 'c', 'label'])
    pairplot(df, hue='label').show()

    df = sns.load_dataset("iris")
    df = pd.DataFrame([[1, None, None, 'a'], [None, 3, 4, 'b'], [None, None, None, 'c']], columns=['a', 'b', 'c', 'label'])
    df = pd.DataFrame([[1, 2, 3, 'f'], [6, 5, 4, 'g'], [7, 8, 9, 'h']], columns=['a', 'b', 'c', 'label'])
    sns.pairplot(df, hue='label', dropna=True)
    sns.plt.show()
