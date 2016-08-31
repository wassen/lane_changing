#!/usr/bin/env python
# import os,sys

import numpy as np
import os, sys
import lane_changing as lc
from lane_changing import Container, ContainerInitializer, Features, Label
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(purpose, line, overwrite=False):
    report_file = open("{}_report.txt".format(purpose),'w' if overwrite else 'a')
    report_file.write(line)
    report_file.close()


feature1 = Features.TimeToCollisionX
feature2 = Features.TimeToCollisionY

# ctn = Container(DataInput.readFromCSVData)
ctn = Container(ContainerInitializer.loadOriginalData)

load = True
xlist_2dim = ctn.feature_with_frames(feature1, load=load)
ylist_2dim = ctn.feature_with_frames(feature2, load=load)
start_labels = ctn.feature_with_frames(Features.Label, load=False)

xlist, delete_index = ctn.extract_nearest_car(xlist_2dim)
ylist, delete_index = ctn.extract_nearest_car(ylist_2dim)
llist = np.delete(start_labels, delete_index)

llist = np.array(llist)
left = ctn.two_features_of_specific_label(xlist, ylist, llist, Label.begin_left_lanechange)
straight = ctn.two_features_of_specific_label(xlist, ylist, llist, Label.go_straight)
brake = ctn.two_features_of_specific_label(xlist, ylist, llist, Label.braking_and_go_straight)
right = ctn.two_features_of_specific_label(xlist, ylist, llist, Label.begin_right_lanechange)

report("showhist",
       "左車線変更{0}回, 右車線変更{1}回, 直進{2}フレーム, ブレーキ{3}フレーム".format(len(left[0]),
                                                             len(right[0]),
                                                             len(straight[0]),
                                                             len(brake[0])),
       True)

# left = np.array(left)


def delete_inf_and_nan(array):
    return np.delete(array, np.where((array == np.float('inf')) | (np.isnan(array)))[1], 1)


left = delete_inf_and_nan(left)
straight = delete_inf_and_nan(straight)
right = delete_inf_and_nan(right)
brake = delete_inf_and_nan(brake)

xlim = (-5, 5)
ylim = (-5, 5)

labels = [left, right, straight, brake]
graph_title = ['start_of_left_lane_changing', 'start_of_right_lane_changing', 'go_straight', 'braking']

fig = plt.figure()

# label数は4固定
for i, (label, graph_title) in enumerate(zip(labels, graph_title)):
    # 2*2に区切ったi番めのプロット。変数がないなら221とかでも表記できる。

    sp_i = fig.add_subplot(2, 2, i + 1)

    normed = False
    H = sp_i.hist2d(*label, bins=[np.linspace(*xlim, 61), np.linspace(*ylim, 61)], normed=normed)

    sp_i.set_title(graph_title)
    sp_i.set_xlabel(feature1.name)
    sp_i.set_ylabel(feature2.name)
    fig.colorbar(H[3], ax=sp_i)

fig.subplots_adjust(wspace=0.3, hspace=0.6)
plt.savefig('fig.png')

plt.clf()
