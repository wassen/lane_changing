#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import time
import os
import repo_env


def meter2pix(m):
    return int(m * 20)


LANE_WIDTH = meter2pix(3.5)
CAR_WIDTH = meter2pix(2.)
CAR_LENGTH = meter2pix(4.)
CAR_SIZE = np.array([CAR_WIDTH, CAR_LENGTH])

# 何故かRGBではなく、BGRになってるが・・・
BG_COLOR = (54, 61, 59)
SELF_COLOR = (38, 166, 91)
SELF_NEARMISS_COLOR = ()
SURROUND_COLOR = (239, 144, 96)
SURROUND_NEARMISS_COLOR = ()
WHITELINE_COLOR = (220, 220, 220)




# TODO 曲率
# TODO なめらかに車線変更
# TODO 車線増加によるワープ
# TODO 自車両の中心固定
# TODO TTC1以内表示

def mod_lane_number(self_lane, lane_number):
    return lane_number + 1 - self_lane

def left_max(lanes_self, lane_numbers):
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
    m = 0
    all_left_addition = 0
    for left_lane_addition in branch_lane:
        all_left_addition += left_lane_addition[0]
        if all_left_addition > m:
            m = all_left_addition
    m -= 1
    return m

def output_video(vels, self_lanes, lcs , lane_numbers, rel_xes_list, rel_ys_list, rel_vys_list, name):

    # print(left_max(self_lanes, lane_numbers), max(lane_numbers))

    # 上下反転
    rel_ys_list = [np.array(rel_ys) * -1 for rel_ys in rel_ys_list]

    IMG_HEIGHT = 1024
    IMG_WIDTH = LANE_WIDTH * max(lane_numbers)

    BACK_GROUND = np.array([[BG_COLOR for _ in range(IMG_WIDTH)] for _ in range(IMG_HEIGHT)], np.uint8)

    fourcc = cv2.cv.CV_FOURCC(*'mp4v')
    FPS = 10
    path = os.path.join(repo_env.OUTPUT_DIR, '{}.avi'.format(name))
    vout = cv2.VideoWriter(path, fourcc, FPS, (IMG_WIDTH, IMG_HEIGHT,))

    for i, (vel, self_lane, lc, lane_number, rel_xes, rel_ys, rel_vys) in enumerate(
            zip(vels, self_lanes, lcs, lane_numbers, rel_xes_list, rel_ys_list, rel_vys_list)
    ):
        if lane_number == 0:
            lane_number = 1
        if self_lane == 0:
            self_lane = 1
        #左から順に1,2,3となるように修正
        #self_lane = mod_lane_number(self_lane, lane_number)
        img = np.array(BACK_GROUND)
        print(i)

        def center_of_self_car():
            # 左側に車線が増えることがあればややこしい。というか普通は左側に増えるのか・・・？
            return np.array([LANE_WIDTH * lane_number * (2. * self_lane - 1.) / (2. * lane_number), IMG_HEIGHT / 2.])

        def draw_speed(cosec, v):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(v) + 'km', tuple(np.array(cosec, np.uint16) - (40, 0)), font, 1, (255, 255, 255))

        def center_of_surrounding_car(cosec, rel_x, rel_y):
            return cosec + np.array([meter2pix(rel_x), meter2pix(rel_y)])

        def draw_car(cosuc, color):
            cv2.rectangle(img, tuple(np.array(cosuc + CAR_SIZE / 2., np.uint16)),
                          tuple(np.array(cosuc - CAR_SIZE / 2., np.uint16)), color, 5)

        def self_lc(cosec, lc):
            if lc == 1:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, 'right lc', tuple(np.array(cosec, np.uint16) - (80, 20)), font, 1, (255, 255, 255))
            elif lc == -1:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, 'left lc', tuple(np.array(cosec, np.uint16) - (80, 20)), font, 1, (255, 255, 255))
            elif lc == 3:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, 'braking', tuple(np.array(cosec, np.uint16) - (80, 20)), font, 1, (255, 255, 255))
            elif lc == 4:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, 'accel', tuple(np.array(cosec, np.uint16) - (80, 20)), font, 1, (255, 255, 255))


        cosec = center_of_self_car()
        draw_car(cosec, SELF_COLOR)
        draw_speed(cosec, vel)
        self_lc(cosec, lc)

        for rel_x, rel_y, rel_vy in zip(rel_xes, rel_ys, rel_vys):
            cosuc = center_of_surrounding_car(cosec, rel_x, rel_y)
            draw_car(cosuc, SURROUND_COLOR)
            print(vel, rel_vy)
            draw_speed(cosuc, vel + rel_vy)

        # 白線
        EDGE_WHITELINE_WIDTH = 10
        CENTER_WHITELINE_WIDTH = 2


        for j in range(int(lane_number)):
            if j == 0:
                cv2.line(img, (0, 0), (0, IMG_HEIGHT - 1), WHITELINE_COLOR, EDGE_WHITELINE_WIDTH)

            x_pos = IMG_WIDTH / max(lane_numbers) * (j + 1)
            if j == lane_number - 1:
                cv2.line(img, (x_pos, 0), (x_pos, IMG_HEIGHT - 1), WHITELINE_COLOR, EDGE_WHITELINE_WIDTH)
            else:
                cv2.line(img, (x_pos, 0), (x_pos, IMG_HEIGHT - 1), WHITELINE_COLOR, CENTER_WHITELINE_WIDTH)

        start = time.time()
        vout.write(img)
        print(time.time() - start)
