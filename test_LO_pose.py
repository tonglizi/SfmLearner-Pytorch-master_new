# -*- coding: utf-8 -*- 
# @Time : 2020/7/25 17:34 
# @Author : CaiXin
# @File : test_LO_pose.py

import os
import random
import time

import torch
from torch.autograd import Variable

from scipy.misc import imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from models import PoseExpNet
from inverse_warp import pose_vec2mat

parser = argparse.ArgumentParser(
    description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("pretrained_posenet", type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--sequences", default=['09'], type=str, nargs='*', help="sequences to test")
parser.add_argument("--output-dir", default=None, type=str,
                    help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)
parser.add_argument("--max-loops", default=100, type=int)

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    global tgt_pc, tgt_img
    args = parser.parse_args()

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()

        '''
        LO部分
        以VO的结果作为预估，并计算和真值的差值
        '''
    from VOLO import LO
    # from point_cloud_processing.prepare_data import loadPointCloudOpen3D
    # .pcd格式读取
    pointClouds = loadPointCloud(args.dataset_dir)

    total_time = 0
    max_loops = args.max_loops
    if max_loops > len(pointClouds):
        max_loops = len(pointClouds)
    abs_LO_poses = np.zeros((max_loops, 12))
    abs_LO_pose = np.identity(4)
    abs_LO_poses[0]=abs_LO_pose[:3,:].reshape(-1, 12)[0]

    # L和C的转换矩阵,对齐输出位姿
    Transform_matrix_L2C = np.identity(4)
    Transform_matrix_L2C[:3, :3] = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04],
                                             [1.480249e-02, 7.280733e-04, -9.998902e-01],
                                             [9.998621e-01, 7.523790e-03, 1.480755e-02]])
    Transform_matrix_L2C[:3, -1:] = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)

    # for j in (range(len(pointClouds)-1)):
    for j in (range(max_loops - 1)):
        tgt_pc = random_sampling(pointClouds[j], 10000)
        pc = random_sampling(pointClouds[j + 1], 10000)

        from point_cloud_processing.icpImpl import icp
        start = time.time()
        T, distacnces, iterations = icp(pc, tgt_pc, init_pose=None, tolerance=0.001, max_iterations=50)
        total_time += time.time() - start
        print('*********** LO pose {}/{}: {:.3} s/{:.3} mins *******************'.format(j, max_loops - 1,
                                                                                         time.time() - start,
                                                                                         total_time / 60))
        print(T)

        cur_LO_pose = T
        # 对齐到相机坐标系
        cur_LO_pose = Transform_matrix_L2C @ cur_LO_pose @ np.linalg.inv(Transform_matrix_L2C)

        abs_LO_pose= abs_LO_pose@cur_LO_pose
        abs_LO_poses[j+1] = abs_LO_pose[:3,:].reshape(-1, 12)[0]

        # mean_errors = errors.mean(0)
        # std_errors = errors.std(0)
        # error_names = ['ATE','RE']
        # print('')
        # print("Results")
        # print("\t {:>10}, {:>10}".format(*error_names))
        # print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
        # print("std \t {:10.4f}, {:10.4f}".format(*std_errors))
        #
        # optimized_mean_errors = optimized_errors.mean(0)
        # optimized_std_errors = optimized_errors.std(0)
        # optimized_error_names = ['optimized_ATE','optimized_RE']
        # print('')
        # print("optimized_Results")
        # print("\t {:>10}, {:>10}".format(*optimized_error_names))
        # print("mean \t {:10.4f}, {:10.4f}".format(*optimized_mean_errors))
        # print("std \t {:10.4f}, {:10.4f}".format(*optimized_std_errors))

    if args.output_dir is not None:
        np.savetxt(output_dir / 'abs_LO_poses.txt', abs_LO_poses)


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:, :, -1] * pred[:, :, -1]) / np.sum(pred[:, :, -1] ** 2)
    ATE = np.linalg.norm((gt[:, :, -1] - scale_factor * pred[:, :, -1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:, :3] @ np.linalg.inv(pred_pose[:, :3])
        s = np.linalg.norm([R[0, 1] - R[1, 0],
                            R[1, 2] - R[2, 1],
                            R[0, 2] - R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return ATE / snippet_length, RE / snippet_length


def random_sampling(orig_points, num_points):
    assert orig_points.shape[0] > num_points
    points_down_idx = random.sample(range(orig_points.shape[0]), num_points)
    down_points = orig_points[points_down_idx, :]
    return down_points

def loadPointCloud(rootdir):
    files = os.listdir(rootdir)
    pointclouds = []
    for file in files:
        if not os.path.isdir(file):
            scan = np.fromfile(rootdir + "/" + file, dtype=np.float32)
            scan = scan.reshape((-1, 4))
            ptcloud_xyz = scan[:, :-1]
            print(ptcloud_xyz.shape)
            pointclouds.append(ptcloud_xyz)
    return pointclouds


if __name__ == '__main__':
    main()
