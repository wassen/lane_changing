#!/usr/bin/env python
# import os,sys

import numpy as np
import os, sys
import lane_changing as lc
from lane_changing import Container, DataInput, Features, Label
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ctn = Container(DataInput.loadOriginalData)

load = True
xlist_2dim = ctn.feature_with_frames(Features.TimeToCollisionX, load=load)
ylist_2dim = ctn.feature_with_frames(Features.TimeToCollisionY, load=load)
start_labels = ctn.feature_with_frames(Features.Label, load=load)

xlist, delete_index = ctn.extract_nearest_car(xlist_2dim)
ylist, delete_index = ctn.extract_nearest_car(ylist_2dim)
llist = np.delete(start_labels, delete_index)


llist = np.array(llist)
left = ctn.two_features_of_specific_label(xlist, ylist, llist, Label.begin_left_lanechange)
straight = ctn.two_features_of_specific_label(xlist, ylist, llist, Label.go_straight)
brake = ctn.two_features_of_specific_label(xlist, ylist, llist, Label.braking_and_go_straight)
right = ctn.two_features_of_specific_label(xlist, ylist, llist, Label.begin_right_lanechange)

print(len(left[0]))
print(len(straight[0]))
print(len(brake[0]))
print(len(right[0]))

fig = plt.figure()
# 2*2に区切った1番めのプロット。221とも表記できる。
sp1 = fig.add_subplot(2, 2, 1)
sp2 = fig.add_subplot(2, 2, 2)
sp3 = fig.add_subplot(2, 2, 3)
sp4 = fig.add_subplot(2, 2, 4)

left = np.array(left)


def delete_inf_and_nan(array):
    return np.delete(array, np.where((array == np.float('inf')) | (np.isnan(array)))[1], 1)


left = delete_inf_and_nan(left)
straight = delete_inf_and_nan(straight)
right = delete_inf_and_nan(right)
brake = delete_inf_and_nan(brake)

xlim = (-5, 5)
ylim = (-5, 5)

normed = False

H1 = sp1.hist2d(*left, bins=[np.linspace(*xlim, 61), np.linspace(*ylim, 61)], normed=normed)
H2 = sp2.hist2d(*right, bins=[np.linspace(*xlim, 61), np.linspace(*ylim, 61)], normed=normed)
H3 = sp3.hist2d(*straight, bins=[np.linspace(*xlim, 61), np.linspace(*ylim, 61)], normed=normed)
H4 = sp4.hist2d(*brake, bins=[np.linspace(*xlim, 61), np.linspace(*ylim, 61)], normed=normed)

sp1.set_title('1st graph')
sp2.set_title('2st graph')
sp3.set_title('3st graph')
sp4.set_title('4st graph')
sp1.set_xlabel('x')
sp2.set_xlabel('x')
sp3.set_xlabel('x')
sp4.set_xlabel('x')
sp1.set_ylabel('y')
sp2.set_ylabel('y')
sp3.set_ylabel('y')
sp4.set_ylabel('y')
fig.colorbar(H1[3], ax=sp1)
fig.colorbar(H2[3], ax=sp2)
fig.colorbar(H3[3], ax=sp3)
fig.colorbar(H4[3], ax=sp4)

plt.savefig('fig.pdf')

plt.clf()
