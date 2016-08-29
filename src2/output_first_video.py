#!/usr/bin/env python
# -*- coding: utf-8 -*-

import driving_video as dv
import driving_data as dd

# behavior_nameで[0]を指定できるようにしたい
# d6000_0 = dd.d6000[0]

lane_numbers = dd.d6000[0]['roa']['LN']
self_lanes = dd.d6000[0]['roa']['host_#L']
sur = dd.d6000[0]['sur']
vels = dd.d6000[0]['roa']['host_v(km/h)']
# print(dd.behavior_names6000)

# TODO selflaneのNULLを何とかする。dropじゃあ動画っぽくなくなるし、前後の補完でいいだろう。

# この処理をうまくDrivingdataに持っていくにはどうすればいいだろう。
def get_xes_list(sur):
    return [[car[0] for car in cars] for cars in sur]
def get_ys_list(sur):
    return [[car[1] for car in cars] for cars in sur]
def get_vys_list(sur):
    return [[int(car[3]) for car in cars] for cars in sur]
def completion_Null(null_list):
    l = []
    last = 0
    for item in null_list:
        if item == 'Null':
            l.append(last)
        else:
            item = int(item)
            l.append(item)
            last = item
    return l

self_lanes = completion_Null(self_lanes)
lane_numbers = completion_Null(lane_numbers)
rel_xes_list = get_xes_list(sur)
rel_ys_list = get_ys_list(sur)
rel_vys_list = get_vys_list(sur)

# こういうアクセスってどうやって実装するんだ？メソッド内にメソッド積みまくる？
# dd.subject('f6372').task('nc5').type('roa').name('LN')

dv.output_video(vels, self_lanes, lane_numbers, rel_xes_list, rel_ys_list, rel_vys_list)