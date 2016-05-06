#!/usr/bin/env python
import os
import sys

path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(path)
from data_container import Container, DataInput

ctn = Container(DataInput.loadOriginalData)
ctn.show_dtcp_ttcp()
