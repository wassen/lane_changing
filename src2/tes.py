# !/usr/bin/env python
# -*- coding: utf-8 -*-

if __name__ == '__main__':
    import ddTools as dT

    features = ["front_center_distance", "front_center_relvy"]
    diffs = ["diff_{}".format(feature) for feature in features]
    prevs = ["prev_{}".format(feature) for feature in features]

    delc = dT.DataEachLC(features=features)
