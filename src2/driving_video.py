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


# 曲率
# なめらかに車線変更
# 自車速
# 車線増加によるワープ
# 自車両の中心固定

def mod_lane_number(self_lane, lane_number):
    return lane_number + 1 - self_lane

def output_video(self_lanes, lane_numbers, rel_xes_list, rel_ys_list):
    IMG_HEIGHT = 1024
    IMG_WIDTH = LANE_WIDTH * max(lane_numbers)

    BACK_GROUND = np.array([[BG_COLOR for _ in range(IMG_WIDTH)] for _ in range(IMG_HEIGHT)], np.uint8)

    fourcc = cv2.cv.CV_FOURCC(*'mp4v')
    FPS = 10
    vout = cv2.VideoWriter('output.avi', fourcc, FPS, (IMG_WIDTH, IMG_HEIGHT,))

    for i, (self_lane, lane_number, rel_xes, rel_ys) in enumerate(
            zip(self_lanes, lane_numbers, rel_xes_list, rel_ys_list)
    ):
        self_lane = mod_lane_number(self_lane, lane_number)
        img = np.array(BACK_GROUND)
        print(i)

        def center_of_self_car():
            # 左側に車線が増えることがあればややこしい。というか普通は左側に増えるのか・・・？
            return np.array([LANE_WIDTH * lane_number * (2. * self_lane - 1.) / (2. * lane_number), IMG_HEIGHT / 2.])

        def draw_self_car(cosec):
            cv2.rectangle(img, tuple(np.array(cosec + CAR_SIZE / 2., np.uint16)),
                          tuple(np.array(cosec - CAR_SIZE / 2., np.uint16)), SELF_COLOR, 5)

        def center_of_surrounding_car(cosec, rel_x, rel_y):
            return cosec + np.array([meter2pix(rel_x), meter2pix(rel_y)])

        def draw_surrounding_car(cosuc):
            cv2.rectangle(img, tuple(np.array(cosuc + CAR_SIZE / 2., np.uint16)),
                          tuple(np.array(cosuc - CAR_SIZE / 2., np.uint16)), SURROUND_COLOR, 5)

        cosec = center_of_self_car()
        draw_self_car(cosec)

        for rel_x, rel_y in zip(rel_xes, rel_ys):
            draw_surrounding_car(center_of_surrounding_car(cosec, rel_x, rel_y))

        # 白線
        for j in range(int(lane_number - 1)):
            j = j + 1
            x_pos = IMG_WIDTH / max(lane_numbers) * j
            cv2.line(img, (x_pos, 0), (x_pos, IMG_HEIGHT - 1), WHITELINE_COLOR, 5)

        def reverse_y_direction(img):
            return img[::-1]
        start = time.time()
        vout.write(reverse_y_direction(img))
        print(time.time() - start)
