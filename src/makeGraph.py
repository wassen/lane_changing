#!/usr/bin/env python
import os
import sys
# ./makeGraph ttcx ttcy 使える引数を自動でエラーに出力したい。
path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(path)
from lane_changing import Container, DataInput

args = sys.argv
# ここのやつ
if len(args) != 3:
    raise Exception("引数２つにしてね。\n{}が使えます。".format(','.join([item.value for item in list(Container.Features)])))
feature1 = args[1]
feature2 = args[2]

#わかりやすく整理したい
# loadの自動化
ctn = Container(DataInput.loadOriginalData)
ctn.show_plot(Container.Features(feature1), Container.Features(feature2), load=True)

