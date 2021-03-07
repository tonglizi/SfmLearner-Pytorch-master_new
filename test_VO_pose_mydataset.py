# -*- coding: utf-8 -*- 
# @Time : 2020/7/25 17:34 
# @Author : CaiXin
# @File : test_VO_pose.py

import os
import time

import torch
from PIL import Image
from torch.autograd import Variable

from scipy.misc import imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from models import PoseExpNet,DispNetS
from inverse_warp import pose_vec2mat

import argparse

import numpy as np

from utils import tensor2array

np.set_printoptions(precision=4)
from matplotlib.animation import FFMpegWriter

from tqdm import tqdm

# from minisam import *
# how to install minisam: https://minisam.readthedocs.io/install.html
from slam_utils.ScanContextManager import *
from slam_utils.PoseGraphManager import *
from slam_utils.UtilsMisc import *
import slam_utils.UtilsPointcloud as Ptutils
import slam_utils.ICP as ICP
import custom_transforms

parser = argparse.ArgumentParser(
    description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pretrained_posenet", type=str, help="pretrained PoseNet path")
parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=341, type=int, help="Image height")
parser.add_argument("--img-width", default=427, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--sequences", default='09', type=str, nargs='*', help="sequences to test")
parser.add_argument("--output-dir", default=None, type=str,
                    help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)

parser.add_argument('--num_icp_points', type=int, default=5000)  # 5000 is enough for real time

parser.add_argument('--num_rings', type=int, default=20)  # same as the original paper
parser.add_argument('--num_sectors', type=int, default=60)  # same as the original paper
parser.add_argument('--num_candidates', type=int, default=10)  # must be int
parser.add_argument('--try_gap_loop_detection', type=int, default=10)  # same as the original paper

parser.add_argument('--loop_threshold', type=float,
                    default=0.11)  # 0.11 is usually safe (for avoiding false loop closure)


parser.add_argument('--save_gap', type=int, default=300)
parser.add_argument('--isDynamic', type=bool, default=False,help="Only for dynamic scene to test photo mask")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    global tgt_pc, tgt_img
    args = parser.parse_args()
    # from kitti_eval.VOLO_data_utils import test_framework_KITTI as test_framework
    from mydataset_eval.pose_evaluation_utils import test_framework_MYDATASET as test_framework

    weights = torch.load(args.pretrained_posenet)
    seq_length = int(weights['state_dict']['conv1.0.weight'].size(1) / 3)
    pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
    pose_net.load_state_dict(weights['state_dict'], strict=False)

    # *************************可删除*********************************
    # 为了进行Mask评估，这里需要引入disp net
    disp_net = DispNetS().to(device)
    weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    # normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                                         std=[0.5, 0.5, 0.5])
    # valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
    # from datasets.sequence_folders import SequenceFolder
    # val_set = SequenceFolder(
    #     '/home/sda/mydataset/preprocessing/formatted/data/',
    #     transform=valid_transform,
    #     seed=0,
    #     train=False,
    #     sequence_length=3,
    # )
    # val_loader = torch.utils.data.DataLoader(
    #     val_set, batch_size=1, shuffle=False,
    #     num_workers=4, pin_memory=True)
    #
    # intrinsics = None
    # for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):
    #     intrinsics = intrinsics.to(device)
    #     break
    # *************************************************************************

    dataset_dir = Path(args.dataset_dir)
    framework = test_framework(dataset_dir, args.sequences, seq_length)

    print('{} snippets to test'.format(len(framework)))
    errors = np.zeros((len(framework), 2), np.float32)
    optimized_errors = np.zeros((len(framework), 2), np.float32)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()
        predictions_array = np.zeros((len(framework), seq_length, 3, 4))
        abs_VO_poses = np.zeros((len(framework), 12))

    abs_VO_pose = np.identity(4)[:3, :]

    # L和C的转换矩阵,对齐输入位姿到雷达坐标系
    Transform_matrix_L2C = np.identity(4)
    Transform_matrix_L2C[:3, :3] = np.array([[-1.51482698e-02, -9.99886648e-01,  5.36310553e-03],
                                             [-4.65337018e-03, -5.36307196e-03, -9.99969412e-01],
                                             [9.99870070e-01, -1.56647995e-02, -4.48880010e-03]])
    Transform_matrix_L2C[:3, -1:] = np.array([4.29029924e-03, -6.08539196e-02, -9.20346161e-02]).reshape(3, 1)

    Transform_matrix_C2L = np.linalg.inv(Transform_matrix_L2C)

    # *************可视化准备***********************
    num_frames = len(tqdm(framework))
    # Pose Graph Manager (for back-end optimization) initialization
    PGM = PoseGraphManager()
    PGM.addPriorFactor()

    # Result saver
    save_dir = "result/" + args.sequences[0]
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    ResultSaver = PoseGraphResultSaver(init_pose=PGM.curr_se3,
                                       save_gap=args.save_gap,
                                       num_frames=num_frames,
                                       seq_idx=args.sequences[0],
                                       save_dir=save_dir)

    # Scan Context Manager (for loop detection) initialization
    SCM = ScanContextManager(shape=[args.num_rings, args.num_sectors],
                             num_candidates=args.num_candidates,
                             threshold=args.loop_threshold)

    # for save the results as a video
    fig_idx = 1
    fig = plt.figure(fig_idx)
    writer = FFMpegWriter(fps=15)
    video_name = args.sequences[0] + ".mp4"
    num_frames_to_skip_to_show = 5
    num_frames_to_save = np.floor(num_frames / num_frames_to_skip_to_show)
    with writer.saving(fig, video_name, num_frames_to_save):  # this video saving part is optional

        for j, sample in enumerate(tqdm(framework)):

            '''
            VO部分
            并计算和真值的差值
            '''
            imgs = sample['imgs']

            w, h = imgs[0].size
            if (not args.no_resize) and (h != args.img_height or w != args.img_width):
                imgs = [imresize(img, (args.img_height, args.img_width)).astype(np.float32) for img in imgs]

            imgs = [np.transpose(img, (2, 0, 1)) for img in imgs]

            ref_imgs = []
            for i, img in enumerate(imgs):
                img = torch.from_numpy(img).unsqueeze(0)
                img = ((img / 255 - 0.5) / 0.5).to(device)
                if i == len(imgs) // 2:
                    tgt_img = img
                else:
                    ref_imgs.append(img)

            timeCostVO = 0
            startTimeVO = time.time()
            _, poses = pose_net(tgt_img, ref_imgs)
            timeCostVO = time.time() - startTimeVO

            # **************************可删除********************************
            if args.isDynamic:
                '''测试Photo mask的效果'''
                intrinsics = [[279.1911, 0.0000, 210.8265],
                              [0.0000, 279.3980, 172.3114],
                              [0.0000, 0.0000, 1.0000]]
                intrinsics = torch.tensor(intrinsics).unsqueeze(0)
                intrinsics = intrinsics.to(device)

                print('intrinsics')
                print(intrinsics)
                disp = disp_net(tgt_img)
                depth = 1 / disp

                ref_depths = []
                for ref_img in ref_imgs:
                    ref_disparities = disp_net(ref_img)
                    ref_depth = 1 / ref_disparities
                    ref_depths.append(ref_depth)

                from loss_functions2 import photometric_reconstruction_and_depth_diff_loss
                reconstruction_loss, depth_diff_loss, warped_imgs, diff_maps, weighted_masks = photometric_reconstruction_and_depth_diff_loss(
                    tgt_img, ref_imgs, intrinsics,
                    depth, ref_depths,
                    _, poses,
                    'euler',
                    'zeros',
                    isTrain=False)

                im_path = args.output_dir + '/result/' + args.sequences[0] + '/seq_{}/'.format(j)
                if not os.path.exists(im_path):
                    os.makedirs(im_path)
                # save tgt_img
                tgt_img = tensor2array(tgt_img[0]) * 255
                tgt_img = tgt_img.transpose(1, 2, 0)
                img = Image.fromarray(np.uint8(tgt_img)).convert('RGB')
                # img.show()
                if args.output_dir is not None:
                    img.save(im_path + 'tgt.jpg')

                for i in range(len(warped_imgs[0])):
                    warped_img = tensor2array(warped_imgs[0][i]) * 255
                    warped_img = warped_img.transpose(1, 2, 0)
                    img = Image.fromarray(np.uint8(warped_img)).convert('RGB')
                    # img.show()
                    if args.output_dir is not None:
                        img.save(im_path+'src_{}.jpg'.format(i))

                for i in range(len(weighted_masks[0])):
                    weighted_mask = weighted_masks[0][i].cpu().clone().numpy() * 255
                    img = Image.fromarray(weighted_mask)
                    img=img.convert('L')
                    # img.show()
                    if args.output_dir is not None:
                        img.save(im_path+'photomask_{}.jpg'.format(i))
            # ***************************************************************
            poses = poses.cpu()[0]
            poses = torch.cat([poses[:len(imgs) // 2], torch.zeros(1, 6).float(), poses[len(imgs) // 2:]])

            inv_transform_matrices = pose_vec2mat(poses, rotation_mode=args.rotation_mode).numpy().astype(np.float64)

            rot_matrices = np.linalg.inv(inv_transform_matrices[:, :, :3])
            tr_vectors = -rot_matrices @ inv_transform_matrices[:, :, -1:]

            transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)
            print('**********DeepVO result: time_cost {:.3} s'.format(timeCostVO / (len(imgs) - 1)))
            # print(transform_matrices)
            # 将对[0 1 2]中间1的转换矩阵变成对0的位姿转换
            first_inv_transform = inv_transform_matrices[0]
            final_poses = first_inv_transform[:, :3] @ transform_matrices
            final_poses[:, :, -1:] += first_inv_transform[:, -1:]
            # print('first')
            # print(first_inv_transform)
            print('poses')
            print(final_poses)

            # cur_VO_pose取final poses的第2项，则是取T10,T21,T32。。。
            cur_VO_pose = np.identity(4)
            cur_VO_pose[:3, :] = final_poses[1]
            # 引入尺度因子对单目VO输出的位姿进行修正 TODO 尺度因子如何确定
            scale_factor = 7
            cur_VO_pose[:3, -1:] = cur_VO_pose[:3, -1:] * scale_factor
            print("对齐前的帧间位姿")
            print(cur_VO_pose)

            cur_VO_pose = Transform_matrix_C2L @ cur_VO_pose @ np.linalg.inv(Transform_matrix_C2L)
            print("对齐到雷达坐标系帧间位姿")
            print(cur_VO_pose)

            PGM.curr_node_idx = j  # make start with 0
            if (PGM.curr_node_idx == 0):
                PGM.prev_node_idx = PGM.curr_node_idx
                continue
            # update the current (moved) pose
            PGM.curr_se3 = np.matmul(PGM.curr_se3, cur_VO_pose)

            # add the odometry factor to the graph
            # PGM.addOdometryFactor(cur_VO_pose)

            # renewal the prev information
            PGM.prev_node_idx = PGM.curr_node_idx

            # loop detection and optimize the graph
            if (PGM.curr_node_idx > 1 and PGM.curr_node_idx % args.try_gap_loop_detection == 0):
                # 1/ loop detection
                loop_idx, loop_dist, yaw_diff_deg = SCM.detectLoop()
                if (loop_idx == None):  # NOT FOUND
                    pass
                # else:
                #     print("Loop event detected: ", PGM.curr_node_idx, loop_idx, loop_dist)
                #     # 2-1/ add the loop factor
                #     loop_scan_down_pts = SCM.getPtcloud(loop_idx)
                #     loop_transform, _, _ = ICP.icp(curr_scan_down_pts, loop_scan_down_pts,
                #                                    init_pose=yawdeg2se3(yaw_diff_deg), max_iterations=20)
                #     PGM.addLoopFactor(loop_transform, loop_idx)
                #
                #     # 2-2/ graph optimization
                #     PGM.optimizePoseGraph()
                #
                #     # 2-2/ save optimized poses
                #     ResultSaver.saveOptimizedPoseGraphResult(PGM.curr_node_idx, PGM.graph_optimized)

            # save the ICP odometry pose result (no loop closure)
            ResultSaver.saveUnoptimizedPoseGraphResult(PGM.curr_se3, PGM.curr_node_idx)
            if (j % num_frames_to_skip_to_show == 0):
                ResultSaver.vizCurrentTrajectory(fig_idx=fig_idx)
                writer.grab_frame()

            # 绝对位姿
            abs_VO_pose[:, :3] = cur_VO_pose[:3, :3] @ abs_VO_pose[:, :3]
            abs_VO_pose[:, -1:] += cur_VO_pose[:3, -1:]

            print('对齐到雷达坐标系的绝对位姿')
            print(abs_VO_pose)

            if args.output_dir is not None:
                predictions_array[j] = final_poses
                abs_VO_poses[j] = abs_VO_pose.reshape(-1, 12)[0]

            # ATE, RE = compute_pose_error(sample['poses'], final_poses)
            # errors[j] = ATE, RE

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
        error_names = ['ATE', 'RE']
        print('')
        print("Results")
        print("\t {:>10}, {:>10}".format(*error_names))
        print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
        print("std \t {:10.4f}, {:10.4f}".format(*std_errors))

        optimized_mean_errors = optimized_errors.mean(0)
        optimized_std_errors = optimized_errors.std(0)
        optimized_error_names = ['optimized_ATE', 'optimized_RE']
        print('')
        print("optimized_Results")
        print("\t {:>10}, {:>10}".format(*optimized_error_names))
        print("mean \t {:10.4f}, {:10.4f}".format(*optimized_mean_errors))
        print("std \t {:10.4f}, {:10.4f}".format(*optimized_std_errors))

        if args.output_dir is not None:
            #np.save(output_dir / 'predictions.npy', predictions_array)
            np.savetxt(output_dir / 'abs_VO_poses.txt', abs_VO_poses)


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:, :, -1] * pred[:, :, -1]) / np.sum(pred[:, :, -1] ** 2)
    print("scale_factor: %s", scale_factor)
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


if __name__ == '__main__':
    main()
