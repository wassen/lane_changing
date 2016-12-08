#!/usr/bin/env python
# -*- coding:utf-8 -*-

if __name__ == "__main__":

    import pandas as pd
    import numpy as np


    a = pd.read_pickle('chinpo')
    am = a.as_matrix()
    # for i in am.T:
    #     print(a[[1]])

    print(a[[3]])

    import seaborn as sns
    print(a.index)
    # a = a.dropna()
    green_to_red = sns.diverging_palette(145, 10, n=100, center="dark")  # , s=70, l=40, n=3
    ax = sns.pairplot(a.fillna(float(0)), hue="label",palette=green_to_red[:], dropna=True)
    sns.plt.show()



    

