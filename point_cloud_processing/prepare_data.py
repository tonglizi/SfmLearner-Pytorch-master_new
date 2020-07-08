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
