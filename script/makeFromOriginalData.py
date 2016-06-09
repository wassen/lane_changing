#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(path)
from lane_changing import Container, DataInput

ctn = Container(DataInput.readFromCSVData)
ctn.adjust_size()

ctn.save_dataDicts()
print("すべてのデータのサンプル数が揃っているか否か：" +
      str(ctn.is_all_same_size()))

# mfv = MakeFeatureVector(DataInput.loadOriginalData)
ctn.assign_lc_to_oneDimVectors()
ctn.add_drv_to_twoDimVectors()
ctn.add_ttc_to_twoDimVectors()
ctn.add_ttn_to_twoDimVectors()
ctn.add_distAndVel_to_twoDimVectors()
ctn.add_ttcpanddtcp_to_twoDimVectors()

ctn.save_vectors()
# mfv = MakeFeatureVector(DataInput.loadVectorData)

ctn.retain_sequence(30)
ctn.map_all_m1_to_1()
ctn.delete_3frames_after()
ctn.vectors_to_label_and_feature()
ctn.connect_label_and_feature(7)
print('labelとfeatureのサイズ確認：' + str(ctn.is_label_and_feature_same_size()))
ctn.save_label_and_feature()
