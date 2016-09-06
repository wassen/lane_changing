#!/usr/bin/env python
# -*- coding: utf-8 -*-
# testやりたいためだけに残してる。
import numpy as np
import math
if __name__ == '__main__':
    car_tests = []
    expects = []

    car_tests.append([4,0,-1,0,0,0])
    expects.append((2. * 1000 / 3600, np.float('nan')))
    car_tests.append([4,0,0,0,-1,0])
    expects.append((2. * 1000 / 3600, np.float('nan')))
    car_tests.append([0,6,0,-1,0,0])
    expects.append((np.float('nan'), 2. * 1000 / 3600))
    car_tests.append([0,6,0,0,0,-1])
    expects.append((np.float('nan'), 2. * 1000 / 3600))
    # 適当expected
    car_tests.append([0, 6, 0, 100, 0, -1])
    expects.append((np.float('nan'), 2. * 1000 / 3600))
    car_tests.append([0,6,0,-3,0,1])
    expects.append((np.float('nan'), 2. * 1000 / 3600))
    car_tests.append([0,4 ,0, 3,0,-1])
    expects.append((np.float('nan'), 2. * 1000 / 3600))

    for car, expect in zip(car_tests, expects):
        actual = calc_feature_from_car(car)
        print(expect, actual)
