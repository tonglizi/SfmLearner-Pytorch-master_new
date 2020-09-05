# -*- coding: utf-8 -*-
'''
***********点云的处理库*******
pcl:c++的开源库，python的接口提供的支持较少（windows环境下超级难装！）
open3D: intel开源的python点云库，作为python环境的主力库
在读取数据方面，两者都能达到同样的效果
'''
import os
# import pcl
# import open3d as o3d
import numpy as np


# from pypcd import pypcd


# def loadPointCloud(rootdir):
#     files=os.listdir(rootdir)
#     pointclouds=[]
#     for file in files:
#         if not os.path.isdir(file):
#             p=pcl.load(rootdir+"\\"+file)
#             pointclouds.append(p)
#     return pointclouds

# def loadImage(rootdir):
#     files=os.listdir(rootdir)
#     images=[]
#     for file in files:
#         if not os.path.isdir(file):
#             img=pcl.load(rootdir+"\\"+file)
#             images.append(img)
#     return images

# def read_pcd(path_file):
#     pcd = o3d.io.read_point_cloud(path_file)
#     return pcd


# def loadPointCloudOpen3D(rootdir):
#     files = os.listdir(rootdir)
#     pointclouds = []
#     for file in files:
#         if not os.path.isdir(file):
#             p = read_pcd(rootdir + "/" + file)
#             pointclouds.append(p)
#     return pointclouds


def loadPointCloud(rootdir):
    files = os.listdir(rootdir)
    files.sort()
    pointclouds = []
    for file in files:
        if not os.path.isdir(file):
            p = np.fromfile(rootdir + "/" + file, dtype=np.float32)
            pointclouds.append(p)
    return pointclouds

# *********test code***************
# path=r'C:\Users\93121\Desktop\dataset\velodyne_pcd\0000000000.pcd'

# test open3D read and convert to ndarray
# p=read_pcd(path)
# print(p)
# import numpy as np
# xyz_load = np.asarray(p.points)
# print(xyz_load)


# test PCL read and convert to ndarray
# p=pcl.load(path)
# print(p)
# import numpy as np
# xyz_load=np.asarray(p)
# print(xyz_load)
