#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

WIDTH_OF_CARS=2
LENGTH_OF_CARS=4

def calc_feature_from_car(car):
    x = float(car[0])
    y = float(car[1])
    vx = float(car[2])
    vy = float(car[3])
    ax = float(car[4])
    ay = float(car[5])

    def solve_quad(a, b, c):
        if a == 0:
            return - c / b , - c / b
        else :
            return (-b + math.sqrt(pow(b, 2) - 4 * a * c)) / 2 / a, (-b - math.sqrt(pow(b, 2) - 4 * a * c)) / 2 / a

    
    try:
        sols = solve_quad(1./2.*ax, vx, (x - WIDTH_OF_CARS * np.sign(x)))
        if sols[0] > 0 and sols[1] > 0:
            resultx =  min(sols)* 1000. / 3600.
        else:
            resultx =  max(sols)* 1000. / 3600.
    except:
        resultx =  float('inf')
    try:
        sols = solve_quad(1./2.*ay, vy, (y - LENGTH_OF_CARS * np.sign(y)))
        if sols[0] > 0 and sols[1] > 0:
            resulty =  min(sols) * 1000. / 3600.
        else:
            resulty =  max(sols) * 1000. / 3600.
    except:
        resulty =  float('inf')
    return resultx, resulty

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
