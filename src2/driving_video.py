#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

LANE_WIDTH = 3.5 * 60
CAR_WIDTH = 2 * 60
CAR_LENGTH = 4 * 60

CAR_SIZE = np.array([CAR_WIDTH, CAR_LENGTH])
self_lanes = [1, 1, 2, 2]
lane_numbers = [2, 2, 2, 2]
BACK_GROUND = np.zeros((512, LANE_WIDTH * max(lane_numbers), 3), np.uint8)
rel_x = [3, 3, 4, 4]
rel_y = [0, 0, 1, 1]
# 曲率

max(lane_numbers)

for i, (lane_number, self_lane) in enumerate(zip(lane_numbers, self_lanes)):

    max(lane_numbers)

    img = BACK_GROUND
    img_height = img.shape[0]
    img_width = img.shape[1]
    center = np.array([img_width, img_height])

    def center_of_car():
        # return np.array([img_width * (2 * self_lane - 1)/2 * lane_number, img_width / 2])
        return np.array([img_height / 2, img_width / 2])

    def draw_self_car(img):
        cv2.rectangle(img, tuple(center_of_car() / 2 - CAR_SIZE / 2), tuple(center_of_car() / 2 + CAR_SIZE / 2), (255, 0, 0), 5)

    draw_self_car(img)

    # 左上から右下まで、太さ10の赤い線を引く
    # cv2.line(img, (width/ln, 0), (width/ln, height - 1), (0, 0, 255), 10)

    # 右上から左下まで、太さ50の青い線を引く
    for j in range(lane_number - 1):
        j = j + 1
        x_pos = img_width / max(lane_numbers) * j
        cv2.line(img, (x_pos, 0), (x_pos, img_height - 1), (255, 0, 0), 5)

    cv2.imwrite("{}.png".format(i), img)
