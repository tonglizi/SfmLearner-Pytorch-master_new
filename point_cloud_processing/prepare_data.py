import os

import pcl
import numpy as np
import path


def loadData(rootdir):
    path=rootdir
    files=os.listdir(path)
    pointclouds=[]
    for file in files:
        if not os.path.isdir(file):
            p=pcl.load(file)
            pointclouds.append(p)
    return pointclouds

def
p = pcl.load("C/table_scene_lms400.pcd")

a = np.asarray(p)       # NumPy view on the cloud
a[:] = 0                # fill with zeros
print(p[3])             # prints (0.0, 0.0, 0.0)
a[:, 0] = 1             # set x coordinates to 1
print(p[3])             # prints (1.0, 0.0, 0.0)