#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
import repo_env
import numpy as np
import os

check1 = os.path.join(repo_env.OUTPUT_DIR, 'check1.png')
img = np.zeros((720,1028, 3), np.uint8)
cv2.line(img, (20, 20), (100, 20), (255, 0, 0), 3)
print(check1)
cv2.imwrite(check1, img)

print(cv2.__version__)
check2 = os.path.join(repo_env.OUTPUT_DIR, 'check2.avi')#mov, avi, mp4
#fourcc = cv2.VideoWriter_fourcc(*'H264')
fourcc = cv2.cv.CV_FOURCC(*'mp4v')#XVID, DIVX, mp4v
vout = cv2.VideoWriter(check2, fourcc, 20, (1028, 720))
print(check2)
for _ in range(100):
    vout.write(img)
vout.release()
