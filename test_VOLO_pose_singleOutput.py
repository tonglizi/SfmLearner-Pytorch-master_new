# -*- coding: utf-8 -*- 
# @Time : 2020/7/24 0:01 
# @Author : CaiXin
# @File : test_VOLO_pose_singleOutput.py

import os
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


parser = argparse.ArgumentParser(description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pretrained_posenet", type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--sequences", default=['09'], type=str, nargs='*', help="sequences to test")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    global tgt_pc, tgt_img
    args = parser.parse_args()
    # from kitti_eval.VOLO_data_utils import test_framework_KITTI as test_framework
    from kitti_eval.pose_evaluation_utils import test_framework_KITTI as test_framework


    weights = torch.load(args.pretrained_posenet)
    seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
    pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
    pose_net.load_state_dict(weights['state_dict'], strict=False)

    dataset_dir = Path(args.dataset_dir)
    framework = test_framework(dataset_dir, args.sequences, seq_length)

    print('{} snippets to test'.format(len(framework)))
    errors = np.zeros((len(framework), 2), np.float32)
    optimized_errors = np.zeros((len(framework), 2), np.float32)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()
        predictions_array = np.zeros((len(framework), seq_length, 3, 4))
        abs_VO_poses=np.zeros((len(framework),12))


    abs_VO_pose=np.identity(4)[:3,:]
    for j, sample in enumerate(tqdm(framework)):

        '''
        VO部分
        并计算和真值的差值
        '''
        imgs = sample['imgs']

        h,w,_ = imgs[0].shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            imgs = [imresize(img, (args.img_height, args.img_width)).astype(np.float32) for img in imgs]

        imgs = [np.transpose(img, (2,0,1)) for img in imgs]

        ref_imgs = []
        for i, img in enumerate(imgs):
            img = torch.from_numpy(img).unsqueeze(0)
            img = ((img/255 - 0.5)/0.5).to(device)
            if i == len(imgs)//2:
                tgt_img = img
            else:
                ref_imgs.append(img)

        timeCostVO=0
        startTimeVO=time.time()
        _, poses = pose_net(tgt_img, ref_imgs)
        timeCostVO=time.time()-startTimeVO

        poses = poses.cpu()[0]
        poses = torch.cat([poses[:len(imgs)//2], torch.zeros(1,6).float(), poses[len(imgs)//2:]])

        inv_transform_matrices = pose_vec2mat(poses, rotation_mode=args.rotation_mode).numpy().astype(np.float64)

        rot_matrices = np.linalg.inv(inv_transform_matrices[:,:,:3])
        tr_vectors = -rot_matrices @ inv_transform_matrices[:,:,-1:]

        transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)
        print('**********DeepVO result: time_cost {:.3} s'.format(timeCostVO/(len(imgs)-1)))
        #print(transform_matrices)
        #将对[0 1 2]中间1的转换矩阵变成对0的位姿转换
        first_inv_transform = inv_transform_matrices[0]
        final_poses = first_inv_transform[:,:3] @ transform_matrices
        final_poses[:,:,-1:] += first_inv_transform[:,-1:]
        # print('first')
        # print(first_inv_transform)
        print('poses')
        print(final_poses)

        cur_VO_pose=final_poses[1]
        abs_VO_pose[:,:3]=cur_VO_pose[:,:3]@abs_VO_pose[:,:3]
        abs_VO_pose[:,-1:]+=cur_VO_pose[:,-1:]

        print('绝对位姿')
        print(abs_VO_pose)



        if args.output_dir is not None:
            predictions_array[j] = final_poses
            abs_VO_poses[j]=abs_VO_pose.reshape(-1,12)[0]


        ATE, RE = compute_pose_error(sample['poses'], final_poses)
        errors[j] = ATE, RE


        #
        # '''
        # LO部分
        # 以VO的结果作为预估，并计算和真值的差值
        # '''
        # pointclouds=sample['pointclouds']
        # from VOLO import LO
        # #pointcluds是可以直接处理的
        # for i, pc in enumerate(pointclouds):
        #     if i == len(pointclouds)//2:
        #
        #         tgt_pc =pointclouds[i]
        #
        # optimized_transform_matrices=[]
        #
        # timeCostLO=0
        # startTimeLO=time.time()
        # totalIterations=0
        # for i,pc in enumerate(pointclouds):
        #     pose_proposal=np.identity(4)
        #     pose_proposal[:3,:]=transform_matrices[i]
        #     print('======pose proposal for LO=====')
        #     print(pose_proposal)
        #
        #     T,distacnces,iterations=LO(pc,tgt_pc,init_pose=pose_proposal, max_iterations=50, tolerance=0.001,LO='icp')
        #     optimized_transform_matrices.append(T)
        #     totalIterations+=iterations
        #     print('iterations:\n')
        #     print(iterations)
        # timeCostLO=time.time()-startTimeLO
        # optimized_transform_matrices=np.asarray(optimized_transform_matrices)
        #
        # print('*****LO result: time_cost {:.3} s'.format(timeCostLO/(len(pointclouds)-1))+' average iterations: {}'
        #       .format(totalIterations/(len(pointclouds)-1)))
        # # print(optimized_transform_matrices)
        #
        #
        # #TODO 打通VO-LO pipeline: 需要将转换矩阵格式对齐； 评估VO的预估对LO的增益：效率上和精度上； 评估过程可视化
        # #TODO 利用数据集有对应的图像，点云和位姿真值的数据集（Kitti的odomerty）
        #
        # inv_optimized_rot_matrices = np.linalg.inv(optimized_transform_matrices[:,:3,:3])
        # inv_optimized_tr_vectors = -inv_optimized_rot_matrices @ optimized_transform_matrices[:,:3,-1:]
        # inv_optimized_transform_matrices = np.concatenate([inv_optimized_rot_matrices, inv_optimized_tr_vectors], axis=-1)
        #
        # first_inv_optimized_transform = inv_optimized_transform_matrices[0]
        # final_optimized_poses = first_inv_optimized_transform[:,:3] @ optimized_transform_matrices[:,:3,:]
        # final_optimized_poses[:,:,-1:] += first_inv_optimized_transform[:,-1:]
        # # print('first')
        # # print(first_inv_optimized_transform)
        # print('poses')
        # print(final_optimized_poses)
        #
        # if args.output_dir is not None:
        #     predictions_array[j] = final_poses
        #
        # optimized_ATE, optimized_RE = compute_pose_error(sample['poses'], final_optimized_poses)
        # optimized_errors[j] = optimized_ATE, optimized_RE
        #
        # print('==============\n===============\n')

    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['ATE','RE']
    print('')
    print("Results")
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*std_errors))

    optimized_mean_errors = optimized_errors.mean(0)
    optimized_std_errors = optimized_errors.std(0)
    optimized_error_names = ['optimized_ATE','optimized_RE']
    print('')
    print("optimized_Results")
    print("\t {:>10}, {:>10}".format(*optimized_error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*optimized_mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*optimized_std_errors))

    if args.output_dir is not None:
        np.save(output_dir/'predictions.npy', predictions_array)
        np.savetxt(output_dir/'abs_VO_poses.txt',abs_VO_poses)


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:,:,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2)
    ATE = np.linalg.norm((gt[:,:,-1] - scale_factor * pred[:,:,-1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:,:3] @ np.linalg.inv(pred_pose[:,:3])
        s = np.linalg.norm([R[0,1]-R[1,0],
                            R[1,2]-R[2,1],
                            R[0,2]-R[2,0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s,c)

    return ATE/snippet_length, RE/snippet_length


if __name__ == '__main__':
    main()