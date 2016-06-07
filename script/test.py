import os,sys
path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(path)
from lane_changing import Container, DataInput

print(Container.start_index([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, -1, -1, 0, 0, 0, -1, 0]))
import numpy as np
print(np.where(np.array([1,2,1]) == 2))


#ctn = Container(DataInput.loadOriginalData)
