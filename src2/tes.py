#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA

if __name__ == "__main__":

    import ddTools as dT

    data_each_lc = dT.DataEachLC(features=['front_center_distance', 'front_center_relvy'])
    print(list(data_each_lc.prevs())[0].shape)
