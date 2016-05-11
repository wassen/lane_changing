#!/usr/bin/env python
import os
import sys

path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(path)
from lane_changing import Container, DataInput

ctn = Container(DataInput.loadOriginalData)
ctn.show_dtcp_ttcp(load=True)
