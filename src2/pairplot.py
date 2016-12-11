#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def rm_duplicates(l):
    s = []
    for item in l:
        if not all(np.in1d(item, s)):
            s.append(item)
    return s

def pairplot(df, **kwargs):
    """

    :param df:
    :param kwargs:
    :return: matplotlib.pyplot.figure
    """

    fig = plt.figure()

    # kwargs
    if kwargs.has_key("hue"):
        labels = df[kwargs["hue"]]
    else:
        # 面倒なので今のところエラーで
        raise
    if kwargs.has_key("palette"):
        palette=kwargs["palette"]
    else:
        # 面倒なので今のところエラーで
        raise

    def prepro(df):
        def assign_color_to_label(df):
            corresp = {label: color for label, color in zip(rm_duplicates(df[labels]), palette)}
            return [corresp[label] for label in df[labels]]

        return df.assign(**{"color": assign_color_to_label})

    df = prepro(df)

    feature = df.drop("color", axis=1).drop(kwargs["hue"], axis=1)

    feature_number = feature.shape[1]

    comb = [(x,y) for y in feature for x in feature]

    # lims = {fname: (min(feature[fname]), max(feature[fname])) for fname in feature}

    sp_list=[]
    scatter_flag=[False for _ in feature]
    for i, fnames in enumerate(comb):

        sp_row = feature.columns.get_loc(fnames[1])
        sp_col = feature.columns.get_loc(fnames[0])

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

        f = df[[fnames[0],fnames[1],kwargs["hue"],"color"]].dropna()

        def feature_each_label(feature, labels):
            return [feature.as_matrix()[np.where(np.array(labels) == label)[0]] for label in rm_duplicates(labels)]
        if sp_row == sp_col:
            # クソコード
            sp.hist(feature_each_label(f[fnames[0]], f[kwargs["hue"]]), align='left', color=rm_duplicates(f["color"]), stacked=True)
        else:
            # d=[feature_each_label(feature[fname], labels) for fname in fnames]
            # data = np.array([feature_each_label(feature[fname], labels) for fname in fnames])
            # # nanの行をpaletetと一緒に削除
            sp.scatter(*[feature_each_label(f[fname], f[kwargs["hue"]]) for fname in fnames], color=rm_duplicates(f["color"]))

        def hide_unnecessary_label():

            if sp_row != feature_number - 1:
                sp.get_xaxis().set_visible(False)
            if sp_col != 0:
                sp.get_yaxis().set_visible(False)
        hide_unnecessary_label()

    def top_right_ylabel():
        """
            左上のヒストグラムのスケールを変化させずに、軸の表記だけを変更させる。
        """
        y_label = [int(f) for f in sp_list[1].get_yticks()]
        sp_list[0].set_yticks(np.linspace(0, 1, len(y_label)))
        sp_list[0].set_yticklabels(y_label)

    top_right_ylabel()

    return fig



if __name__ == "__main__":

    from matplotlib import pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sns

    # plt.hist([1,2,3])

    df = pd.DataFrame([[1, 2, 3, 'f'], [6, 5, 4, 'g'], [7, 8, 9, 'h']], columns=['a', 'b', 'c', 'label'])
    df = pd.DataFrame([[1, None, 3, 'a'], [6, 3, 4, 'b'], [5, None, None, 'c']], columns=['a', 'b', 'c', 'label'])
    green_to_red = sns.diverging_palette(145, 10, n=3, center="dark")
    pairplot(df, hue='label', palette=green_to_red).show()

    df = sns.load_dataset("iris")
    df = pd.DataFrame([[1, 2, 3, 'f'], [6, 5, 4, 'g'], [7, 8, 9, 'h']], columns=['a', 'b', 'c', 'label'])
    df = pd.DataFrame([[1, None, 3, 'a'], [6, 3, 4, 'b'], [None, 5, None, 'c']], columns=['a', 'b', 'c', 'label'])
    green_to_red = sns.diverging_palette(145, 10, n=3, center="dark")
    sns.pairplot(df, hue='label', palette=green_to_red, dropna=True)
    sns.plt.show()
