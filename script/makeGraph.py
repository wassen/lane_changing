#!/usr/bin/env python
import os
import sys

path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(path)
from lane_changing import Container, DataInput

args = sys.argv
# ここのやつ
if len(args) != 3:
    raise Exception("引数２つにしてね")
feature1=args[1]
feature2=args[2]

# loadの自動化
ctn = Container(DataInput.loadOriginalData)
print(ctn.behavior_names)
ctn.show_plot(Container.Features(feature1), Container.Features(feature2), load=True)

