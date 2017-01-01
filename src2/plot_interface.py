#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == '__main__':
    # いちいち読み込むのが時間かかる。
    # import sys
    # sys.argv

    # if sys.argv == "2d":
    import plot_2d
    # plot_2d.scatter_each_behavior('front_center_distance', 'front_center_relvy')
    # print('')
    # for com in plot_2d.get_feature_combinations():
    #     plot_2d.scatter_all_behavior(*com)
    # plot_2d.contours('front_center_distance', 'front_center_relvy')
    # plot_2d.scatter_all_behavior('front_center_distance', 'front_center_relvy')
    plot_2d.scatter_each_time('front_center_distance', 'front_center_relvy')