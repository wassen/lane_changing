#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import time


def meter2pix(m):
    return int(m * 60)


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

def mod_lane_number(self_lane, lane_number):
    return lane_number + 1 - self_lane

def output_video(vels, self_lanes, lane_numbers, rel_xes_list, rel_ys_list, rel_vys_list):

    # 上下反転
    rel_ys_list = [np.array(rel_ys) * -1 for rel_ys in rel_ys_list]

    IMG_HEIGHT = 1024
    IMG_WIDTH = LANE_WIDTH * max(lane_numbers)

    BACK_GROUND = np.array([[BG_COLOR for _ in range(IMG_WIDTH)] for _ in range(IMG_HEIGHT)], np.uint8)

    fourcc = cv2.cv.CV_FOURCC(*'mp4v')
    FPS = 10
    vout = cv2.VideoWriter('output.avi', fourcc, FPS, (IMG_WIDTH, IMG_HEIGHT,))

    for i, (vel, self_lane, lane_number, rel_xes, rel_ys, rel_vys) in enumerate(
            zip(vels, self_lanes, lane_numbers, rel_xes_list, rel_ys_list, rel_vys_list)
    ):
        #左から順に1,2,3となるように修正
        self_lane = mod_lane_number(self_lane, lane_number)
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

        cosec = center_of_self_car()
        draw_car(cosec, SELF_COLOR)
        draw_speed(cosec, vel)

        for rel_x, rel_y, rel_vy in zip(rel_xes, rel_ys, rel_vys):
            cosuc = center_of_surrounding_car(cosec, rel_x, rel_y)
            draw_car(cosuc, SURROUND_COLOR)
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
