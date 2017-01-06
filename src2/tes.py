# !/usr/bin/env python
# -*- coding: utf-8 -*-

if __name__ == '__main__':
    def divided_data(l, num=5):
        center = len(l) / 5
        return l[:center], l[center:]


    import numpy as np
    import pandas as pd

    l = [pd.DataFrame([np.arange(2) * i, np.arange(2, 4) * i]) for i in range(104)]

    new_l = np.array(np.array(df) for df in l)

    former, latter = divided_data(l)
    print(latter)
    new_l = np.array([np.array(df) for df in latter])
    shape = new_l.shape
    training = new_l.reshape(shape[0]*shape[1], shape[2])

    test = former
