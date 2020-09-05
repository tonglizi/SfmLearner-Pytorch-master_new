# -*- coding: utf-8 -*- 
# @Time : 2020/9/4 15:33 
# @Author : CaiXin
# @File : SCMTest.py
import argparse
import os
from tqdm import tqdm

from slam_utils.ScanContextManager import *
import slam_utils.UtilsPointcloud as Ptutils

#
# parser = argparse.ArgumentParser()
# parser.add_argument("--data_dir", type=str, default='/home/sda/dataset/sequences')
# parser.add_argument("--sequence_idx", type=str, default='09')

# SCM parms
num_rings = 20
num_sectors = 60
num_candidates = 10
loop_threshold = 1

# data
sequence_dir='E:\\data_odometry\\dataset\\sequences\\00\\velodyne'
sequence_manager = Ptutils.KittiScanDirManager(sequence_dir)
scan_paths = sequence_manager.scan_fullpaths


def SCM_test():
    SCM = ScanContextManager([num_rings, num_sectors], num_candidates, loop_threshold)
    num_downsample_points = 10000
    for i, scan_path in tqdm(enumerate(scan_paths),total=len(scan_paths)):
        print('Running:{}...'.format(i))
        curr_scan_pts = Ptutils.readScan(scan_path)
        curr_scan_down_pts = Ptutils.random_sampling(curr_scan_pts, num_downsample_points)

        SCM.addNode(i, curr_scan_down_pts)

        if i > 1:
            loop_idx, loop_dist, yaw_diff_deg = SCM.detectLoop()
            if loop_idx == None:
                pass
            else:
                print("Loop detected at frame : ", loop_idx)

if __name__=='__main__':
    SCM_test()