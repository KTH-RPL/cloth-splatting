#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import imageio
import random
import os
from random import randint

import torch.linalg
import torch_geometric

from scene_reconstruction.dataset import MDNerfDataset
from scene_reconstruction.dataset_readers import SceneInfo
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene_reconstruction.scene import Scene, read_cloth_scene_info

from scene_reconstruction.gaussian_mesh import MultiGaussianMesh
from meshnet.meshnet_network import MeshSimulator, ResidualMeshSimulator

from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams, MeshnetParams
from torch.utils.data import DataLoader

from utils.timer import Timer
from utils.external import *
import wandb

import lpips
from utils.scene_utils import render_training_image

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)


def image_losses(image_tensor, gt_image_tensor, opt: OptimizationParams, mask_tensor=None) -> (torch.Tensor, dict[str, float]):
    loss_dict = dict()

    # Loss
    l1 = l1_loss(image_tensor, gt_image_tensor, mask_tensor)
    # Ll1 = l2_loss(image, gt_image)
    loss_dict['l1'] = l1.item()
    loss = l1

    # TODO Double check if SSIM with mask works.
    if opt.lambda_dssim != 0:
        if mask_tensor is None:
            ssim_val = ssim(image_tensor, gt_image_tensor)
            ssim_loss = 1.0 - ssim_val
        else:
            ssim_map = ssim(image_tensor, gt_image_tensor, return_map=True)
            ssim_loss = ((1.0 - ssim_map) * mask_tensor).mean()
        loss_dict['ssim_loss'] = ssim_loss.item()
        loss += opt.lambda_dssim * ssim_loss

    # if opt.lambda_lpips != 0:
    #     lpipsloss = lpips_loss(image_tensor, gt_image_tensor, lpips_model)
    #     loss += opt.lambda_lpips * lpipsloss

    return loss, loss_dict


def regularization(all_vertice_deform, gaussians, opt: OptimizationParams, static=False):

    n_cams = all_vertice_deform.shape[0]

    loss = torch.zeros([], device="cuda")

    # TODO Gradually add the loss terms back!
    # l_momentum = 0.0
    # if n_cams >= 3:
    #     ## MOMENTUM LOSS
    #     l_momentum = all_means_3D_deform[2, :, :] - 2 * all_means_3D_deform[1, :, :] + all_means_3D_deform[0, :, :]
    #     l_momentum = torch.linalg.norm(l_momentum, dim=-1, ord=1).mean()  # mean l1 norm
    #
    # l_deformation_mag = 0.0
    if n_cams >= 3:
        l_deformation_delta_0 = all_vertice_deform[1, :, :] - all_vertice_deform[0, :, :]
        l_deformation_mag_0 = torch.linalg.norm(l_deformation_delta_0, dim=-1).mean()  # mean l2 norm
        l_deformation_delta_1 = all_vertice_deform[2, :, :] - all_vertice_deform[1, :, :]
        l_deformation_mag_1 = torch.linalg.norm(l_deformation_delta_1, dim=-1).mean()  # mean l2 norm
        l_deformation_mag = 0.5 * (l_deformation_mag_0 + l_deformation_mag_1)
        loss += opt.lambda_deform_mag * l_deformation_mag

    if not static:
        edge_displacement = all_vertice_deform[:, gaussians.mesh.edge_index[1]] - all_vertice_deform[:,
                                                                                  gaussians.mesh.edge_index[0]]
        deformed_norm = torch.linalg.norm(edge_displacement, dim=-1, keepdim=True)
        static_norm = gaussians.edge_norm.unsqueeze(0).expand(n_cams, -1, -1)
        l_rigid = torch.nn.functional.l1_loss(static_norm, deformed_norm)
        loss += opt.lambda_rigid * l_rigid

    # l_iso, l_rigid, l_shadow_mean, l_shadow_delta, l_spring = None, None, None, None, None
    # diff_dimensions = False
    # if stage == "fine" and iteration > user_args.reg_iter:
    #
    #     if o3d_knn_dists is not None and all_means_3D_deform.shape[1] * args.k_nearest != o3d_knn_dists.shape[0]:
    #         diff_dimensions = True
    #     else:
    #         diff_dimensions = False
    #
    #     if iteration % user_args.knn_update_iter == 0 or o3d_knn_dists is None or diff_dimensions:
    #         t_0_pts = get_pos_t0(gaussians).detach().cpu().numpy()
    #         o3d_dist_sqrd, o3d_knn_indices = o3d_knn(t_0_pts, args.k_nearest)
    #         o3d_knn_dists = np.sqrt(o3d_dist_sqrd)
    #         o3d_knn_dists = torch.tensor(o3d_knn_dists, device="cuda").flatten()
    #         o3d_dist_sqrd = torch.tensor(o3d_dist_sqrd, device="cuda").flatten()
    #         knn_weights = torch.exp(-args.lambda_w * o3d_dist_sqrd)
    #
    #         if args.use_wandb and stage == "fine":
    #             wandb.log({"train/o3d_knn_dists": o3d_knn_dists.median()}, step=iteration)
    #         print("updating knn's")
    #
    #     ## ISOMETRIC LOSS
    #     all_l_iso = []
    #
    #     all_l_spring = []
    #
    #     prev_rotations = None
    #     prev_offsets = None
    #     all_l_rigid = []
    #     prev_knn_dists = None
    #     for i in range(n_cams):
    #         # o3d_knn_indices : [N,3], 3 nearest neighbors
    #         # means_3D_deform : [N,3]
    #         # knn_points : [N,3,3]
    #
    #         # compute knn_dists [N,3] distance to KNN
    #         means_3D_deform = all_means_3D_deform[i, :, :]
    #         knn_points = means_3D_deform[o3d_knn_indices]
    #         knn_points = knn_points.reshape(-1, 3)  # N x 3
    #         means_3D_deform_repeated = means_3D_deform.unsqueeze(1).repeat(1, args.k_nearest, 1).reshape(-1,
    #                                                                                                      3)  # N x 3
    #
    #         curr_offsets = knn_points - means_3D_deform_repeated
    #         knn_dists = torch.linalg.norm(curr_offsets, dim=-1)
    #         if args.use_wandb and stage == "fine":
    #             wandb.log({"train/knn_dists": knn_dists.median()}, step=iteration)
    #         # print(knn_dists.shape)
    #         # exit()
    #
    #         l_iso_tmp = torch.mean(knn_dists - o3d_knn_dists)
    #
    #         if prev_knn_dists is not None:
    #             l_spring_tmp = torch.mean(torch.abs(knn_dists - prev_knn_dists))
    #             all_l_spring.append(l_spring_tmp)
    #
    #         prev_knn_dists = knn_dists.clone()
    #
    #         all_l_iso.append(l_iso_tmp)
    #
    #         rotations = all_rotations[i, :, :]
    #         knn_rotations = rotations[o3d_knn_indices].reshape((-1, 4))
    #         knn_rotations_inv = quat_inv(knn_rotations)
    #         if prev_rotations is not None:
    #             # compute rigidity loss
    #             # knn_rotation_matrices : [N,3,3], last two dimensions are rotation matrices
    #             rel_rot = quat_mult(prev_rotations, knn_rotations_inv)
    #             rot = build_rotation(rel_rot)
    #
    #             curr_offset_in_prev_coord = torch.bmm(rot, curr_offsets.unsqueeze(-1)).squeeze(-1)
    #             l_rigid_tmp = weighted_l2_loss_v2(curr_offset_in_prev_coord, prev_offsets, knn_weights)
    #             all_l_rigid.append(l_rigid_tmp)
    #
    #         prev_rotations = knn_rotations.clone()
    #         prev_offsets = curr_offsets.clone()
    #
    #         # print(knn_rotations.shape)
    #         # exit()
    #
    #     l_iso = torch.mean(torch.stack(all_l_iso))
    #     l_spring = torch.mean(torch.stack(all_l_spring))
    #     if user_args.use_wandb and stage == "fine":
    #         wandb.log({"train/l_iso": l_iso}, step=iteration)
    #
    #     if user_args.use_wandb and stage == "fine":
    #         wandb.log({"train/l_spring": l_spring}, step=iteration)
    #
    #     l_rigid = torch.mean(torch.stack(all_l_rigid))
    #     if user_args.use_wandb and stage == "fine":
    #         wandb.log({"train/l_rigid": l_rigid}, step=iteration)
    #
    #     # check if all_shadows is empty
    #     if len(all_shadows) > 0:
    #         all_shadows = torch.cat(all_shadows, 0)
    #         all_shadows_std = torch.tensor(all_shadows_std, device="cuda")
    #
    #         mean_shadow = all_shadows.mean()
    #         shadow_std = all_shadows_std.mean()
    #
    #         l_shadow_mean = mean_shadow  # incentivize a lower shadow mean
    #
    #         l_shadow_delta = 0.0
    #         if n_cams >= 3:
    #             delta_shadow_0 = torch.linalg.norm(all_shadows[1] - all_shadows[0], dim=-1)
    #             delta_shadow_1 = torch.linalg.norm(all_shadows[2] - all_shadows[1], dim=-1)
    #
    #             l_shadow_delta = 1.0 - 0.5 * (delta_shadow_0 + delta_shadow_1)  # incentivize a higher shadow delta
    #
    #         if user_args.use_wandb and stage == "fine":
    #             wandb.log({"train/shadows_mean": mean_shadow, "train/shadows_std": shadow_std,
    #                        "train/l_shadow_mean": l_shadow_mean, "train/l_shadow_delta": l_shadow_delta},
    #                       step=iteration)
    #
    # # add momentum term to loss
    # if user_args.lambda_momentum > 0 and stage == "fine":
    #     loss += user_args.lambda_momentum * l_momentum.mean()
    #
    # # add isometric term to loss
    # if user_args.lambda_isometric > 0 and stage == "fine" and l_iso is not None:
    #     loss += user_args.lambda_isometric * l_iso.mean()
    #
    # if user_args.lambda_rigidity > 0 and stage == "fine" and l_rigid is not None:
    #     loss += user_args.lambda_rigidity * l_rigid.mean()
    #
    # if user_args.lambda_shadow_mean > 0 and stage == "fine" and l_shadow_mean is not None:
    #     loss += user_args.lambda_shadow_mean * l_shadow_mean.mean()
    #
    # if user_args.lambda_shadow_delta > 0 and stage == "fine" and l_shadow_delta is not None:
    #     loss += user_args.lambda_shadow_delta * l_shadow_delta.mean()
    #
    # if user_args.lambda_deformation_mag > 0 and stage == "fine":
    #     loss += user_args.lambda_deformation_mag * l_deformation_mag.mean()
    #
    # if user_args.lambda_spring > 0 and stage == "fine" and l_spring is not None:
    #     loss += user_args.lambda_spring * l_spring.mean()
    #
    # if user_args.use_wandb and stage == "fine":
    #     wandb.log({"train/l_momentum": l_momentum, "train/l_deform_mag": l_deformation_mag}, step=iteration)
    #
    # if stage == "fine" and hyper.time_smoothness_weight != 0:
    #     # tv_loss = 0
    #     tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.plane_tv_weight,
    #                                            hyper.l1_time_planes)
    #     loss += tv_loss

    return loss


def densification(gaussians, iteration, visibility_filter, radii, viewspace_point_tensor_grad, opt: OptimizationParams, cameras_extent):

    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                         radii[visibility_filter])
    gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)


    opacity_threshold = opt.opacity_threshold_fine_init - iteration * (
            opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after) / (
                            opt.densify_until_iter)
    densify_threshold = opt.densify_grad_threshold_fine_init - iteration * (
            opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after) / (
                            opt.densify_until_iter)

    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
        size_threshold = 20 if iteration > opt.opacity_reset_interval else None

        gaussians.densify(densify_threshold, opacity_threshold, cameras_extent, size_threshold)
    if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
        size_threshold = 20 if iteration > opt.opacity_reset_interval else None

        gaussians.prune(densify_threshold, opacity_threshold, cameras_extent, size_threshold)

