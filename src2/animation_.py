#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# 360度分のデータを作成

ims = []
for i in range(100):
    rad = math.radians(i)
    im = ax.scatter(math.cos(rad), math.sin(rad))
    ims.append([im])
print(len(ims))
# アニメーション作成
print(int(1./100*1000*10))
ani = animation.ArtistAnimation(fig, ims, interval=int(1./100*1000*10), repeat_delay=1000)

# 表示
ani.save('a.gif', writer="imagemagick", fps=10)
plt.show()
