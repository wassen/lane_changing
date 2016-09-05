#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 左に増えてる
lanes_self = [1,1,2,3]
lane_numbers = [1,1,6,3]

def change_of_ln(lanes_self, lane_numbers):
    prev_num = 0
    prev_self = 0

    branch_lane = []
    for i, (lane_self, lane_num) in enumerate(zip(lanes_self, lane_numbers)):
        if prev_num != lane_num:
            branch_lane.append([lane_self - prev_self, prev_self - lane_self + lane_num - prev_num])
        prev_self = lane_self
        prev_num = lane_num
    return branch_lane

branch_lane = change_of_ln(lanes_self, lane_numbers)

max = 0
all_left_addition = 0
for left_lane_addition in branch_lane:
    all_left_addition += left_lane_addition[0]
    if all_left_addition > max:
        max = all_left_addition
max -= 1
print(branch_lane)
print(max)



