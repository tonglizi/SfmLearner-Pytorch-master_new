import time
import numpy as np
import torch
from models import PoseExpNet
from point_cloud_processing import prepare_data, icpImpl

#三张图片输入的VO
def VOMultiImg(targetImg,ref_imgs,pretrained_posenet,seq_length,device):
    # #导入预训练模型
    # weights = torch.load(pretrained_posenet)
    # seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
    # pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
    # pose_net.load_state_dict(weights['state_dict'], strict=False)
    # #模型计算
    # _, poses = pose_net(image1, image2)
    # poses = poses.cpu()[0]
    # poses = torch.cat([poses[:len(imgs) // 2], torch.zeros(1, 6).float(), poses[len(imgs) // 2:]])
    #
    # inv_transform_matrices = pose_vec2mat(poses, rotation_mode=args.rotation_mode).numpy().astype(np.float64)
    #
    # rot_matrices = np.linalg.inv(inv_transform_matrices[:, :, :3])
    # tr_vectors = -rot_matrices @ inv_transform_matrices[:, :, -1:]
    #
    # return T,R,t
    pass


#TODO 一般训练出的模型是需要三张输入图像的，中间那张是目标图像：如何改写成两张输入
def VO(image1,image2,pretrained_posenet,device):
    # #导入预训练模型
    # weights = torch.load(pretrained_posenet)
    # seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
    # pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
    # pose_net.load_state_dict(weights['state_dict'], strict=False)
    # #模型计算
    # _, poses = pose_net(image1, image2)
    # poses = poses.cpu()[0]
    # poses = torch.cat([poses[:len(imgs) // 2], torch.zeros(1, 6).float(), poses[len(imgs) // 2:]])
    #
    # inv_transform_matrices = pose_vec2mat(poses, rotation_mode=args.rotation_mode).numpy().astype(np.float64)
    #
    # rot_matrices = np.linalg.inv(inv_transform_matrices[:, :, :3])
    # tr_vectors = -rot_matrices @ inv_transform_matrices[:, :, -1:]
    #
    # return T,R,t
    pass

def LO(points1,points2,iterations,T=None,LO='icp'):
    A = np.asarray(points1.points)
    B = np.asarray(points2.points)
    if LO=='bset_fit':
        T, R, t = icpImpl.best_fit_transform(A, B)
    elif LO=='icp':
        T,_,_=icpImpl.icp(A,B,iterations,T)
    return T

#
# #准备数据
# pointsDir='/'
# imgDir='/'
#
# pointclouds=prepare_data.loadPointCloud(pointsDir)
# imgs=prepare_data.loadImage(imgDir)
#
# assert len(pointclouds)==len(imgs)
# len=len(pointclouds)
#
# #开始遍历
# iterations=50
# method='fusion'
# for i in range(len-1):
#
#     total_time=0;
#     start=time.time()
#
#     if method=='fusion':
#         T,_,_=VO(imgs[i],imgs[i+1])
#         print('fusion_VO@'+i+':'+T)
#         T,_,_=LO(pointclouds[i],pointclouds[i+1],iterations,T)
#         print('fusion_LO@'+i+':'+T)
#     elif method=='LO':
#         T,_,_=LO(pointclouds[i],pointclouds[i+1],2*iterations,T)
#         print('LO@'+i+':'+T)
#     elif method=='VO':
#         T,_,_=VO(imgs[i],imgs[i+1])
#         print('VO@'+i+':'+T)
#     else:
#         pass
#     total_time=time.time()-start
#     print(method+' processing time: '+total_time)