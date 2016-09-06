#!/usr/bin/env python
import os
import sys
# ./makeGraph ttcx ttcy
from lane_changing import Container, ContainerInitializer, Features

args = sys.argv
# ここのやつ
if len(args) != 3 or not all(arg in [item.value for item in list(Features)] for arg in args[1:3]):
    raise Exception("引数２つにして、\n{}を指定してください。".format(','.join([item.value for item in list(Features)])))
feature1 = args[1]
feature2 = args[2]

#わかりやすく整理したい
# loadの自動化
ctn = Container(ContainerInitializer.loadOriginalData)
ctn.count_feature_in_circle(Features(feature1), Features(feature2), load=True)