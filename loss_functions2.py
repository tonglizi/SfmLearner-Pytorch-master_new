from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from inverse_warp import *


def photometric_reconstruction_and_depth_diff_loss(tgt_img, ref_imgs, intrinsics,
                                                   tgt_depth, ref_depths, explainability_mask, pose,
                                                   rotation_mode='euler', padding_mode='zeros'):
    def one_scale(tgt_depth):
        assert (pose.size(1) == len(ref_imgs))

        reconstruction_loss = 0
        depth_diff_loss = 0
        b, _, h, w = tgt_depth.size()
        downscale = tgt_img.size(2) / h

        tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
        ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2] / downscale, intrinsics[:, 2:]), dim=1)

        warped_imgs = []
        diff_maps = []

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_depth = ref_depths[i]
            # if type(ref_depth) not in [list, tuple]:
            #     ref_depth = [ref_depth]
            ref_img_warped, _, depth_diff, valid_points = inverse_warp_with_DepthMask(ref_img, tgt_depth[:, 0],
                                                                                      ref_depth[:, 0],
                                                                                      current_pose,
                                                                                      intrinsics_scaled,
                                                                                      rotation_mode, padding_mode)
            diff = (tgt_img_scaled - ref_img_warped) * valid_points.unsqueeze(1).float()
            '''
            加入mask减少动态物体的影响
            mask用深度/光度一致性表示：一致性为1表示损失有效，一致性为0表示损失无效
            '''
            weighted_mask = 1 - (ref_img_warped - tgt_img_scaled).abs() / (ref_img_warped + tgt_img_scaled)
            diff = diff * weighted_mask

            reconstruction_loss += diff.abs().mean()
            depth_diff_loss += depth_diff.abs().mean()
            assert ((reconstruction_loss == reconstruction_loss).item() == 1)
            assert ((depth_diff_loss == depth_diff_loss).item() == 1)

            warped_imgs.append(ref_img_warped[0])
            diff_maps.append(diff[0])

        return reconstruction_loss, depth_diff_loss, warped_imgs, diff_maps

    warped_results, diff_results = [], []
    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(tgt_depth) not in [list, tuple]:
        tgt_depth = list(tgt_depth)

    total_reconstrcution_loss = 0
    total_depth_diff_loss = 0

    for d, mask in zip(tgt_depth, explainability_mask):
        reconstruction_loss, depth_diff_loss, warped, diff = one_scale(d)
        total_reconstrcution_loss += reconstruction_loss
        total_depth_diff_loss += depth_diff_loss
        warped_results.append(warped)
        diff_results.append(diff)
    return total_reconstrcution_loss, total_depth_diff_loss, warped_results, diff_results


def explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = torch.ones_like(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss


def smooth_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()) * weight
        weight /= 2.3  # don't ask me why it works better
    return loss


@torch.no_grad()
def compute_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt) / torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred) ** 2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]
