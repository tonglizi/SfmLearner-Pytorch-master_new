# -*- coding: utf-8 -*-
'''
***********点云的处理库*******
pcl:c++的开源库，python的接口提供的支持较少（windows环境下超级难装！）
open3D: intel开源的python点云库，作为python环境的主力库
'''
import os
import pcl
# import numpy as np
# from pypcd import pypcd


def loadPointCloud(rootdir):
    files=os.listdir(rootdir)
    pointclouds=[]
    for file in files:
        if not os.path.isdir(file):
            p=pcl.load(rootdir+"\\"+file)
            pointclouds.append(p)
    return pointclouds

def loadImage(rootdir):
    files=os.listdir(rootdir)
    images=[]
    for file in files:
        if not os.path.isdir(file):
            img=pcl.load(rootdir+"\\"+file)
            images.append(img)
    return images

# path=r'C:\Users\93121\Desktop\velodyne_pcd\0000000000.pcd'
# p=pcl.load(path)
# print(p)
# points=load_pcd_to_ndarray(path)