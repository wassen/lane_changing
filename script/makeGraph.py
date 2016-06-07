#!/usr/bin/env python
import os
import sys

path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(path)
from lane_changing import Container, DataInput

ctn = Container(DataInput.loadOriginalData)
ctn.show_plot(ctn.Features.TimeToClosestPoint, ctn.Features.DistanceToClosestPoint, load=False)
 #ctn.show_plot(ctn.Features.TimeToCollisionX, ctn.Features.Distance, load=True)
#ctn.show_plot(ctn.Features.TimeToCollisionY, ctn.Features.Distance, load=True)
# ctn.show_plot(ctn.Features.TimeToCollisionX, ctn.Features.TimeToCollisionY, load=False)

# もっとわかりやすくスクリプト整理を行いたい

