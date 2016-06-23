#!/usr/bin/env python
#import os,sys
#path = os.path.join(os.path.dirname(__file__), '../')
#sys.path.append(path)
#from lane_changing import Container, DataInput
#
#print(Container.start_index([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, -1, -1, 0, 0, 0, -1, 0]))
#import numpy as np
#print(np.where(np.array([1,2,1]) == 2))


#ctn = Container(DataInput.loadOriginalData)
import numpy as np
from enum import IntEnum
class Label(IntEnum):
    left_lanechanging = -2
    begin_left_lanechange = -1
    go_straight = 0
    begin_right_lanechange = 1
    right_lanechanging = 2
    braking_and_go_straight = 3

def dividebrakelabel(label, brake, threshold=0.4):
    '''
    直進ラベル(0)をブレーキ踏力のしきい値から0と3に分ける
    '''
    return [Label.braking_and_go_straight.value if b >= threshold and l == Label.go_straight else l for l,b in zip(label, brake)]

label = list(map(lambda item : int(item),np.ones(5))) + list(np.zeros(5))
brake = [0.1, 0.3, 0.4, 0.5, 0.3]
brake = brake * 2
print(dividebrakelabel(label, brake, 0.5))
