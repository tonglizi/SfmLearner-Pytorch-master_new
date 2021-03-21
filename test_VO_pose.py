# -*- coding: utf-8 -*- 
# @Time : 2020/7/25 17:34 
# @Author : CaiXin
# @File : test_VO_pose.py

'''
用来测试训练后的Pose net,即VO模型
有位姿图优化PGM模块，但是只做位姿记录，不做优化
开关介绍：
--isDynamic:适用于有动态物体的测试集，能够额外输出光度误差的光度掩码
--isKitti:适用于带有位姿真值的kitti测试集；能够额外输出和真值比较得到误差
'''
import hashlib
import os
import torch
from PIL import Image
from scipy.misc import imresize
from path import Path
from models import PoseExpNet
from inverse_warp import pose_vec2mat
import argparse
import numpy as np
from utils import tensor2array

np.set_printoptions(precision=4)
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
from sympy import *

from slam_utils.PoseGraphManager import *
from slam_utils.UtilsMisc import *

parser = argparse.ArgumentParser(
    description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pretrained_posenet", type=str, help="pretrained PoseNet path")
parser.add_argument("--pretrained-dispnet", type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=341, type=int, help="Image height")
parser.add_argument("--img-width", default=427, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--sequences", default='09', type=str, nargs='*', help="sequences to test")
parser.add_argument("--output-dir", default='./result/', type=str,
                    help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)
parser.add_argument('--try_pose_graph_optimization', type=int, default=30)  # same as the original paper
parser.add_argument('--save_gap', type=int, default=300)
parser.add_argument('--isDynamic', type=bool, default=False, help="Only for dynamic scene to test photo mask")
parser.add_argument('--isKitti', type=bool, default=False,
                    help="Only for KITTI dataset test, if not, then for mydataset")
parser.add_argument('--trainedOnMydataset', type=bool, default=True,
                    help="image downsample size should be consistent with image size trained on the dataset")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def MD5_ID(pretrained_posenet):
    md = hashlib.md5()  # 创建md5对象
    md.update(str(pretrained_posenet).encode(encoding='utf-8'))
    return md.hexdigest()


@torch.no_grad()
def main():
    global tgt_img, disp_net
    args = parser.parse_args()
    '''加载训练后的模型'''
    weights = torch.load(args.pretrained_posenet)
    seq_length = int(weights['state_dict']['conv1.0.weight'].size(1) / 3)
    pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
    pose_net.load_state_dict(weights['state_dict'], strict=False)
    # 网络模型的MD5 ID
    net_ID = MD5_ID(args.pretrained_posenet)
    # L和C的转换矩阵,对齐输入位姿到雷达坐标系
    Transform_matrix_L2C = np.identity(4)
    '''Kitti switch'''
    if args.isKitti:
        from kitti_eval.pose_evaluation_utils import test_framework_KITTI as test_framework
        save_dir = os.path.join(args.output_dir, "kitti", args.sequences[0], 'net_' + net_ID)
        if args.trainedOnMydataset:
            downsample_img_height = args.img_height
            downsample_img_width = args.img_width
        else:
            # on kitti train set
            downsample_img_height = 128
            downsample_img_width = 416

        Transform_matrix_L2C[:3, :3] = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04],
                                                 [1.480249e-02, 7.280733e-04, -9.998902e-01],
                                                 [9.998621e-01, 7.523790e-03, 1.480755e-02]])
        Transform_matrix_L2C[:3, -1:] = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    else:
        from mydataset_eval.pose_evaluation_utils import test_framework_MyData as test_framework
        save_dir = os.path.join(args.output_dir, "mydataset", args.sequences[0], 'net_' + net_ID)
        if args.trainedOnMydataset:
            downsample_img_height = args.img_height
            downsample_img_width = args.img_width
        else:
            # on kitti train set
            downsample_img_height = 128
            downsample_img_width = 416
        Transform_matrix_L2C[:3, :3] = np.array([[-1.51482698e-02, -9.99886648e-01, 5.36310553e-03],
                                                 [-4.65337018e-03, -5.36307196e-03, -9.99969412e-01],
                                                 [9.99870070e-01, -1.56647995e-02, -4.48880010e-03]])
        Transform_matrix_L2C[:3, -1:] = np.array([4.29029924e-03, -6.08539196e-02, -9.20346161e-02]).reshape(3, 1)
    Transform_matrix_L2C = GramSchmidtHelper(Transform_matrix_L2C)
    Transform_matrix_C2L = np.linalg.inv(Transform_matrix_L2C)
    # *************************可删除*********************************
    # 为了进行动态场景下的Mask评估，这里需要引入disp net
    if args.isDynamic:
        from models import DispNetS
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
    '''载入测试数据集'''
    dataset_dir = Path(args.dataset_dir)
    framework = test_framework(dataset_dir, args.sequences, seq_length)
    print('{} snippets to test'.format(len(framework)))
    errors = np.zeros((len(framework), 2), np.float32)
    '''输出到文件夹中的数据'''
    num_poses = len(framework) - (seq_length - 2)
    predictions_array = np.zeros((len(framework), seq_length, 3, 4))
    processing_time = np.zeros((num_poses - 1, 1))
    # 输出文件夹
    save_dir = Path(save_dir)
    print('Output files wiil be saved in： ' + save_dir)
    if not os.path.exists(save_dir): save_dir.makedirs_p()
    # Pose Graph Manager (for back-end optimization) initialization
    PGM = PoseGraphManager()
    PGM.addPriorFactor()
    # Result saver
    num_frames = len(framework)
    ResultSaver = PoseGraphResultSaver(init_pose=PGM.curr_se3,
                                       save_gap=args.save_gap,
                                       num_frames=num_frames,
                                       seq_idx=args.sequences[0],
                                       save_dir=save_dir)

    # for save the results as a video
    fig_idx = 1
    fig = plt.figure(fig_idx)
    writer = FFMpegWriter(fps=15)
    video_path = save_dir + '/' + args.sequences[0] + ".mp4"
    num_frames_to_skip_to_show = 5
    num_frames_to_save = np.floor(num_frames / num_frames_to_skip_to_show)
    with writer.saving(fig, video_path, num_frames_to_save):  # this video saving part is optional
        for j, sample in enumerate(tqdm(framework)):
            '''
            VO部分
            '''
            imgs = sample['imgs']
            w, h = imgs[0].size
            if (not args.no_resize) and (h != downsample_img_height or w != downsample_img_width):
                imgs = [imresize(img, (downsample_img_height, downsample_img_width)).astype(np.float32) for img in imgs]
            imgs = [np.transpose(img, (2, 0, 1)) for img in imgs]

            ref_imgs = []
            for i, img in enumerate(imgs):
                img = torch.from_numpy(img).unsqueeze(0)
                img = ((img / 255 - 0.5) / 0.5).to(device)
                if i == len(imgs) // 2:
                    tgt_img = img
                else:
                    ref_imgs.append(img)

            startTimeVO = time.time()
            _, poses = pose_net(tgt_img, ref_imgs)
            processing_time[j] = (time.time() - startTimeVO) / (seq_length - 1)

            # **************************可删除********************************
            if args.isDynamic:
                '''测试Photo mask的效果'''
                if args.isKitti:
                    intrinsics = [[2.416744631239935472e+02, 0.000000000000000000e+00, 2.041680103059581199e+02],
                                  [0.000000000000000000e+00, 2.462848682666666491e+02, 5.900083200000000261e+01],
                                  [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]]
                else:
                    intrinsics = [[279.1911, 0.0000, 210.8265],
                                  [0.0000, 279.3980, 172.3114],
                                  [0.0000, 0.0000, 1.0000]]
                PhotoMask_Output(_, disp_net, intrinsics, j, poses, ref_imgs, save_dir)
            # ***************************************************************

            final_poses = pose2tf_mat(args.rotation_mode, imgs, poses)
            predictions_array[j] = final_poses
            # rel_VO_pose取final poses的第2项，整体则是取T10,T21,T32。。。
            rel_VO_pose = np.identity(4)
            rel_VO_pose[:3, :] = final_poses[1]
            # 引入尺度因子对单目VO输出的位姿进行修正,并进行坐标系对齐到雷达坐标系
            scale_factor = 7
            rel_VO_pose[:3, -1:] = rel_VO_pose[:3, -1:] * scale_factor
            rel_VO_pose = Transform_matrix_C2L @ rel_VO_pose @ np.linalg.inv(Transform_matrix_C2L)
            rel_VO_pose = GramSchmidtHelper(rel_VO_pose)
            ResultSaver.saveRelativePose(rel_VO_pose)

            PGM.curr_node_idx = j + 1
            PGM.curr_se3 = np.matmul(PGM.curr_se3, rel_VO_pose)
            PGM.addOdometryFactor(rel_VO_pose)
            PGM.prev_node_idx = PGM.curr_node_idx
            ResultSaver.saveUnoptimizedPoseGraphResult(PGM.curr_se3, PGM.curr_node_idx)

            # if (j % num_frames_to_skip_to_show == 0):
            #     ResultSaver.vizCurrentTrajectory(fig_idx=fig_idx)
            #     writer.grab_frame()

            if args.isKitti:
                ATE, RE = compute_pose_error(sample['poses'], final_poses)
                errors[j] = ATE, RE

        '''save output files'''
        if save_dir is not None:
            # np.save(save_dir / 'predictions.npy', predictions_array)
            ResultSaver.saveFinalPoseGraphResult(filename='abs_VO_poses.txt')
            ResultSaver.saveRelativePosesResult(filename='rel_VO_poses.txt')
            np.savetxt(save_dir / 'processing_time.txt', processing_time)
            if args.isKitti:
                np.savetxt(save_dir / 'errors.txt', errors)

        mean_errors = errors.mean(0)
        std_errors = errors.std(0)
        error_names = ['ATE', 'RE']
        print('')
        print("Results")
        print("\t {:>10}, {:>10}".format(*error_names))
        print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
        print("std \t {:10.4f}, {:10.4f}".format(*std_errors))


def pose2tf_mat(rotation_mode, imgs, poses):
    poses = poses.cpu()[0]
    poses = torch.cat([poses[:len(imgs) // 2], torch.zeros(1, 6).float(), poses[len(imgs) // 2:]])
    inv_transform_matrices = pose_vec2mat(poses, rotation_mode=rotation_mode).numpy().astype(np.float64)
    rot_matrices = np.linalg.inv(inv_transform_matrices[:, :, :3])
    tr_vectors = -rot_matrices @ inv_transform_matrices[:, :, -1:]
    transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)
    # 将对[0 1 2]中间1的转换矩阵变成对0的位姿转换:T(0->0),T(1->0),T(2->0)
    first_inv_transform = inv_transform_matrices[0]
    final_poses = first_inv_transform[:, :3] @ transform_matrices
    final_poses[:, :, -1:] += first_inv_transform[:, -1:]
    return final_poses


def PhotoMask_Output(explainability_mask, disp_net, intrinsics, j, poses, ref_imgs, save_dir):
    global tgt_img
    intrinsics = torch.tensor(intrinsics).unsqueeze(0)
    intrinsics = intrinsics.to(device)
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
        explainability_mask, poses,
        'euler',
        'zeros',
        isTrain=False)
    im_path = save_dir + '/seq_{}/'.format(j)
    if not os.path.exists(im_path):
        os.makedirs(im_path)
    # save tgt_img
    tgt_img = tensor2array(tgt_img[0]) * 255
    tgt_img = tgt_img.transpose(1, 2, 0)
    img = Image.fromarray(np.uint8(tgt_img)).convert('RGB')
    img.save(im_path + 'tgt.jpg')

    for i in range(len(warped_imgs[0])):
        warped_img = tensor2array(warped_imgs[0][i]) * 255
        warped_img = warped_img.transpose(1, 2, 0)
        img = Image.fromarray(np.uint8(warped_img)).convert('RGB')
        img.save(im_path + 'src_{}.jpg'.format(i))
    for i in range(len(weighted_masks[0])):
        weighted_mask = weighted_masks[0][i].cpu().clone().numpy() * 255
        img = Image.fromarray(weighted_mask)
        img = img.convert('L')
        img.save(im_path + 'photomask_{}.jpg'.format(i))


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:, :, -1] * pred[:, :, -1]) / np.sum(pred[:, :, -1] ** 2)
    # print("scale_factor: {}".format(scale_factor))
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


def GramSchmidtHelper(transformation):
    a1 = Matrix(transformation[0, :3])
    a2 = Matrix(transformation[1, :3])
    a3 = Matrix(transformation[2, :3])
    so3 = [a1, a2, a3]
    O = GramSchmidt(so3, True)
    O = np.array(O)
    transformation[:3, :3] = O[:3, :3].reshape(3, 3)
    return transformation


if __name__ == '__main__':
    main()
