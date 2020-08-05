# -*- coding: utf-8 -*- 
# @Time : 2020/7/23 8:23 
# @Author : CaiXin
# @File : trace_visualization_utils.py

'''
based on the github project:https://github.com/utiasSTARS/pykitti
'''
"""Example of pykitti.odometry usage."""
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

import pykitti

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"

# Change this to the directory where you store KITTI data
basedir = 'E:\data_odometry\dataset'

# Specify the dataset to load
sequence = '09'

# Load the data. Optionally, specify the frame range to load.
# dataset = pykitti.odometry(basedir, sequence)
#dataset = pykitti.odometry(basedir, sequence, frames=range(0, 20, 5))
dataset = pykitti.odometry(basedir, sequence)

# dataset.calib:      Calibration data are accessible as a named tuple
# dataset.timestamps: Timestamps are parsed into a list of timedelta objects
# dataset.poses:      List of ground truth poses T_w_cam0
# dataset.camN:       Generator to load individual images from camera N
# dataset.gray:       Generator to load monochrome stereo pairs (cam0, cam1)
# dataset.rgb:        Generator to load RGB stereo pairs (cam2, cam3)
# dataset.velo:       Generator to load velodyne scans as [x,y,z,reflectance]
print("dataset initialized")

# Grab some data
# the 0 is initial state. Not the first state.
second_pose = dataset.poses[2]

# Display some of the data
np.set_printoptions(precision=4, suppress=True)

print('\nSequence: ' + str(dataset.sequence))

print('\nSecond ground truth pose:\n' + str(second_pose))

print('\ntest extract pose')

print(str(second_pose[2][3]))

print(str(dataset.poses[2][2][3]))
if second_pose[2][3] == dataset.poses[2][2][3]:
	print('second_pose[0][3] == dataset.poses[1][0][3] is equal\n')

pose_x_list = []
pose_y_list = []
pose_z_list = []

for idx in range(len(dataset.poses)):
	pose_x_list.append(dataset.poses[idx][0][3])
	pose_y_list.append(dataset.poses[idx][1][3])
	pose_z_list.append(dataset.poses[idx][2][3])

ax = plt.axes(projection='3d')
# z axis is same with vehicle movement. x is left right. y is heading to ground.
ax.plot3D(pose_z_list, pose_x_list, pose_y_list, 'gray')
ax.set_xlabel('vehicle x direction')
ax.set_ylabel('vehicle y direction')
ax.set_zlabel('vechile z direction')
plt.show()


fig = plt.figure()
plt.plot(pose_x_list, pose_z_list)
plt.show()

print('\n script version2 finished\n')