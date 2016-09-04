#!/usr/bin/env python
# import os,sys

import numpy as np
import os, sys
import lane_changing as lc
from lane_changing import Container, ContainerInitializer, Features, Label
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TODO load.close()？
def report(purpose, line, overwrite=False):
    report_file = open("{}_report.txt".format(purpose), 'w' if overwrite else 'a')
    report_file.write(line)
    report_file.close()


feature1 = Features.TimeToClosestPoint
feature2 = Features.DistanceToClosestPoint

# ctn = Container(DataInput.readFromCSVData)
ctn = Container(ContainerInitializer.loadOriginalData)

load = True
xlist_2dim = ctn.feature_with_frames(feature1, load=load)
ylist_2dim = ctn.feature_with_frames(feature2, load=load)
start_labels = ctn.feature_with_frames(Features.Label, load=load)

xlist, delete_index = ctn.extract_nearest_car(xlist_2dim)
ylist, delete_index = ctn.extract_nearest_car(ylist_2dim)
llist = np.delete(start_labels, delete_index)

llist = np.array(llist)
left = ctn.two_features_of_specific_label(xlist, ylist, llist, Label.begin_left_lanechange)
right = ctn.two_features_of_specific_label(xlist, ylist, llist, Label.begin_right_lanechange)
brake = ctn.two_features_of_specific_label(xlist, ylist, llist, Label.braking_and_go_straight)
gas = ctn.two_features_of_specific_label(xlist, ylist, llist, Label.gaspedal_and_go_straight)
straight = ctn.two_features_of_specific_label(xlist, ylist, llist, Label.go_straight)

report("showhist",
       "左車線変更{0}回, 右車線変更{1}回, ブレーキ{2}フレーム, アクセル{3}フレーム, 直進{4}フレーム,".format(len(left[0]),
                                                                           len(right[0]),
                                                                           len(brake[0]),
                                                                           len(gas[0]),
                                                                           len(straight[0])),
       True)


# left = np.array(left)


def delete_inf_and_nan(array):
    return np.delete(array, np.where((array == np.float('inf')) | (np.isnan(array)))[1], 1)


left = delete_inf_and_nan(left)
right = delete_inf_and_nan(right)
brake = delete_inf_and_nan(brake)
gas = delete_inf_and_nan(gas)
straight = delete_inf_and_nan(straight)

xlim = (-2, 2)
ylim = (0, 10)

labels = [left, right, brake, gas, straight]
graph_titles = ['start_of_\nleft_lane_changing', 'start_of_\nright_lane_changing', 'braking', 'gas_pedal', "go_straight"]

fig = plt.figure(figsize=(12, 7))

# label数は4固定
for i, (label, graph_title) in enumerate(zip(labels, graph_titles)):
    # 2*2に区切ったi番めのプロット。変数がないなら221とかでも表記できる。

    sp_i = fig.add_subplot(2, 3, i + 1)

    normed = False
    H = sp_i.hist2d(*label, bins=[np.linspace(*xlim, 61), np.linspace(*ylim, 61)], normed=normed)

    sp_i.set_title(graph_title)
    sp_i.set_xlabel(feature1.name)
    sp_i.set_ylabel(feature2.name)
    fig.colorbar(H[3], ax=sp_i)

fig.subplots_adjust(wspace=0.3, hspace=0.6)
plt.savefig('fig.png')

plt.clf()
