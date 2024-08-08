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
import copy

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

from scene_reconstruction.cameras import Camera

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

    if not static and opt.lambda_deform_mag > 0. and n_cams >= 3:
        l_deformation_delta_0 = all_vertice_deform[1, :, :] - all_vertice_deform[0, :, :]
        l_deformation_mag_0 = torch.linalg.norm(l_deformation_delta_0, dim=-1).mean()  # mean l2 norm
        l_deformation_delta_1 = all_vertice_deform[2, :, :] - all_vertice_deform[1, :, :]
        l_deformation_mag_1 = torch.linalg.norm(l_deformation_delta_1, dim=-1).mean()  # mean l2 norm
        l_deformation_mag = 0.5 * (l_deformation_mag_0 + l_deformation_mag_1)
        loss += opt.lambda_deform_mag * l_deformation_mag

    if not static and opt.lambda_rigid > 0:
        edge_displacement = all_vertice_deform[:, gaussians.mesh.edge_index[1]] - all_vertice_deform[:,
                                                                                  gaussians.mesh.edge_index[0]]
        deformed_norm = torch.linalg.norm(edge_displacement, dim=-1, keepdim=True)
        static_norm = gaussians.edge_norm.unsqueeze(0).expand(n_cams, -1, -1)
        l_rigid = torch.nn.functional.l1_loss(static_norm, deformed_norm)
        loss += opt.lambda_rigid * l_rigid

    if not static and opt.lambda_momentum > 0 and n_cams >= 3:
        l_momentum = all_vertice_deform[2, :, :] - 2 * all_vertice_deform[1, :, :] + all_vertice_deform[0, :, :]
        l_momentum = torch.linalg.norm(l_momentum, dim=-1, ord=1).mean()  # mean l1 norm
        loss += opt.lambda_momentum * l_momentum.mean()

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
    # add momentum term to loss
    #
    # # add isometric term to loss
    # if user_args.lambda_isometric > 0 and stage == "fine" and l_iso is not None:
    #     loss += user_args.lambda_isometric * l_iso.mean()
    #
    # if user_args.lambda_shadow_mean > 0 and stage == "fine" and l_shadow_mean is not None:
    #     loss += user_args.lambda_shadow_mean * l_shadow_mean.mean()
    #
    # if user_args.lambda_shadow_delta > 0 and stage == "fine" and l_shadow_delta is not None:
    #     loss += user_args.lambda_shadow_delta * l_shadow_delta.mean()
    #
    # if user_args.lambda_spring > 0 and stage == "fine" and l_spring is not None:
    #     loss += user_args.lambda_spring * l_spring.mean()
    #
    # if stage == "fine" and hyper.time_smoothness_weight != 0:
    #     # tv_loss = 0
    #     tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.plane_tv_weight,
    #                                            hyper.l1_time_planes)
    #     loss += tv_loss

    return loss


def train_step(iteration, viewpoint_cams: list[Camera], gaussians: MultiGaussianMesh, simulator: MeshSimulator,
               meshnet_optimizer, pipeline_params: PipelineParams, opt_params: OptimizationParams,
               cameras_extent, background, static=False, white_background=False, user_args=None):

    gaussians.update_learning_rate(iteration)

    # Every 1000 its we increase the levels of SH up to a maximum degree
    # TODO make this a hyperparameter
    if iteration % 1000 == 0:
        gaussians.oneupSHdegree()

    images = []
    gt_images = []
    radii_list = []
    visibility_filter_list = []
    viewspace_point_tensor_list = []
    all_vertice_deform = []
    masks = [] if viewpoint_cams[0].mask is not None else None

    for viewpoint_cam in viewpoint_cams:

        render_pkg = render(viewpoint_cam, gaussians, simulator, pipeline_params, background,
                            no_shadow=user_args.no_shadow, render_static=static)
        image = render_pkg.render
        images.append(image.unsqueeze(0))
        gt_image = viewpoint_cam.original_image.cuda()
        gt_images.append(gt_image.unsqueeze(0))
        radii_list.append(render_pkg.radii.unsqueeze(0))
        visibility_filter_list.append(render_pkg.visibility_filter.unsqueeze(0))
        viewspace_point_tensor_list.append(render_pkg.viewspace_points)
        if masks is not None:
            masks.append(viewpoint_cam.mask.cuda().unsqueeze(0))
        all_vertice_deform.append(render_pkg.vertice_deform[None, :, :])

    all_vertice_deform = torch.cat(all_vertice_deform, 0)

    radii = torch.cat(radii_list, 0).max(dim=0).values
    visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
    image_tensor = torch.cat(images, 0)
    gt_image_tensor = torch.cat(gt_images, 0)
    mask_tensor = torch.cat(masks, 0) if masks is not None else None

    psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()

    # norm
    loss, loss_dict = image_losses(image_tensor, gt_image_tensor, opt_params, mask_tensor)
    loss += regularization(all_vertice_deform, gaussians, opt_params, static)

    loss.backward()

    viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor_list[0])
    for idx in range(0, len(viewspace_point_tensor_list)):
        viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad

    with torch.no_grad():

        # Densification
        if iteration < opt_params.densify_until_iter:
            densification(gaussians, iteration, visibility_filter, radii,
                          viewspace_point_tensor_grad, opt_params, cameras_extent)

            if iteration % opt_params.opacity_reset_interval == 0 or (
                    white_background and iteration == opt_params.densify_from_iter):
                print("reset opacity")
                gaussians.reset_opacity()

        if iteration % opt_params.bary_cleanup == 0:
            gaussians.cleanup_barycentric_coordinates()

        # Optimizer step
        if iteration < opt_params.iterations:
            if static:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                meshnet_optimizer.zero_grad()
            else:
                gaussians.optimizer.step()
                meshnet_optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                meshnet_optimizer.zero_grad()

    return psnr_, loss, loss_dict


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


class SingleStepOptimizer:

    scene_info: SceneInfo
    initial_mesh: torch_geometric.data.Data
    mesh_predictions: list[torch_geometric.data.Data]
    camera_data: MDNerfDataset

    gaussians: MultiGaussianMesh

    def __init__(self,
                 opt_params: OptimizationParams,
                 pipeline_params: PipelineParams,
                 meshnet_params: MeshnetParams,
                 model_params: ModelParams,
                 args, n_times_max=-1, save_path=None):

        self.args = args
        self.opt_params = opt_params
        self.meshnet_params = meshnet_params
        self.pipeline_params = pipeline_params
        self.model_params = model_params
        self.n_times_max = n_times_max

        self.save_path = save_path if save_path is not None else self.model_params.model_path

        self.static_iterations = opt_params.static_reconst_iteration

        bg_color = [1, 1, 1] if self.model_params.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.last_iters = 0

    def initialize(self):
        (self.scene_info, self.initial_mesh, self.mesh_predictions) = read_cloth_scene_info(self.model_params.source_path,
                                                                                            self.model_params.white_background,
                                                                                            eval=False)
        self.camera_data = MDNerfDataset(self.scene_info.train_cameras, self.args)

        self.gaussians = MultiGaussianMesh(self.model_params.sh_degree)
        self.gaussians.from_mesh(self.initial_mesh, self.scene_info.nerf_normalization['radius'], self.opt_params.gaussian_init_factor)

        self.gaussians.training_setup(self.opt_params)

        # load simulator
        mesh_pos = torch.concat([mesh.pos.unsqueeze(0) for mesh in self.mesh_predictions], dim=0)
        self.simulator = ResidualMeshSimulator(mesh_pos, n_times=self.n_times_max, device='cuda')
        self.simulator.train()

    def update_data(self, n_times=-1):
        self.scene_info, self.initial_mesh, self.mesh_predictions = read_cloth_scene_info(self.model_params.source_path, self.model_params.white_background)
        self.camera_data = MDNerfDataset(self.scene_info.train_cameras, self.args)

        if n_times > 0:
            self.camera_data.ordered_data = self.camera_data.ordered_data[:, :n_times]
            self.camera_data.n_times = n_times
            self.mesh_predictions = self.mesh_predictions[:n_times]

        # load simulator
        mesh_pos = torch.concat([mesh.pos.unsqueeze(0) for mesh in self.mesh_predictions], dim=0)
        self.simulator = ResidualMeshSimulator(mesh_pos, self.n_times_max, device='cuda')
        self.simulator.train()

    def static_reconstruction(self, train_steps=None, bar=True):

        meshnet_optimizer = torch.optim.Adam(self.simulator.parameters(), lr=self.meshnet_params.lr_init)

        self.static_iterations = train_steps if train_steps is not None else self.static_iterations

        print(f"Starting static reconstruction for {self.static_iterations} steps.")
        progress_bar = tqdm(range(1, self.static_iterations), desc=f"Static reconstruction")

        ema_loss_for_log = 0

        for iteration in range(1, self.static_iterations+1):

            viewpoint_cams = [self.camera_data.get_one_item(iteration % len(self.camera_data), 0)]
            psnr_, loss, loss_dict = train_step(iteration, viewpoint_cams, self.gaussians, self.simulator,
                                                meshnet_optimizer, self.pipeline_params, self.opt_params,
                                                self.scene_info.nerf_normalization['radius'], self.background,
                                                static=True, white_background=self.model_params.white_background,
                                                user_args=self.args)
            # Report test and samples of training set
            if iteration % int(self.static_iterations / 4) == 0:
                torch.cuda.empty_cache()
                # individual to get only a single view at a time
                for j in range(3):
                    l1_test = 0.0
                    psnr_test = 0.0
                    l = [self.camera_data.get_one_item(j, 0)]
                    for ele in l:
                        image = torch.clamp(
                            render(ele, self.gaussians, self.simulator, self.pipeline_params,
                                   no_shadow=True, bg_color=self.background).render, 0.0, 1.0)
                        gt_image = torch.clamp(ele.original_image.to("cuda"), 0.0, 1.0)
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()

                        save_path = os.path.join(self.save_path, "test_renders")
                        os.makedirs(save_path, exist_ok=True)
                        save_im = np.transpose(gt_image.detach().cpu().numpy(), (1, 2, 0))
                        save_im = (save_im * 255).astype(np.uint8)
                        imageio.imsave(
                            os.path.join(save_path, f"0_{iteration}_{j}_gt.png"),
                            save_im)
                        save_im = np.transpose(image.squeeze().detach().cpu().numpy(), (1, 2, 0))
                        save_im = (save_im * 255).astype(np.uint8)
                        imageio.imsave(
                            os.path.join(save_path, f"0_{iteration}_{j}_render.png"),
                            save_im)

            ema_loss_for_log = ema_loss_for_log * 0.99 + loss.item() * 0.01
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point": f"{self.gaussians.num_gaussians}"})
                progress_bar.update(10)

        self.last_iters = self.static_iterations
        return self.gaussians, self.simulator

    def update_mesh_predictions(self, train_steps=None, bar=True):

        meshnet_optimizer = torch.optim.Adam(self.simulator.parameters(),
                                             lr=self.meshnet_params.lr_init)

        iterations = train_steps if train_steps is not None else self.opt_params.iterations

        n_times = self.camera_data.n_times

        progress_bar = tqdm(range(1, iterations), desc=f"Time {n_times-1} from step {self.last_iters}")

        ema_loss_for_log = 0
        for iteration in range(self.last_iters+1, self.last_iters+iterations+1):
            if n_times >= 3:
                probabilities = np.linspace(0.5, 1.5, n_times - 2)
                probabilities /= probabilities.sum()
                time_id = random.choices(range(n_times-2), weights=probabilities, k=1)[0]
                viewpoint_cams = [
                    self.camera_data.get_one_item(iteration % len(self.camera_data), time_id - 1 + i) for i in range(3)
                ]
            elif n_times == 2:
                viewpoint_cams = [
                    self.camera_data.get_one_item(iteration % len(self.camera_data), 0),
                    self.camera_data.get_one_item(iteration % len(self.camera_data), 1)
                ]
            elif n_times == 1:
                viewpoint_cams = [
                    self.camera_data.get_one_item(iteration % len(self.camera_data), 0)
                ]
            else:
                ValueError("No cameras to train on")

            psnr_, loss, loss_dict = train_step(iteration, viewpoint_cams, self.gaussians, self.simulator,
                                                meshnet_optimizer, self.pipeline_params, self.opt_params,
                                                self.scene_info.nerf_normalization['radius'], self.background,
                                                static=False, white_background=self.model_params.white_background,
                                                user_args=self.args)
            with torch.no_grad():
            # Report test and samples of training set
                if iteration % int(train_steps / 4) == 0:
                    torch.cuda.empty_cache()
                    # individual to get only a single view at a time
                    for j in range(3):
                            l1_test = 0.0
                            psnr_test = 0.0
                            l = [self.camera_data.get_one_item(j, n_times-1)]
                            for ele in l:

                                image = torch.clamp(
                                    render(ele, self.gaussians, self.simulator, self.pipeline_params,
                                           no_shadow=True, bg_color=self.background).render, 0.0, 1.0)
                                gt_image = torch.clamp(ele.original_image.to("cuda"), 0.0, 1.0)
                                l1_test += l1_loss(image, gt_image).mean().double()
                                psnr_test += psnr(image, gt_image).mean().double()

                                save_path = os.path.join(self.save_path, "test_renders".format(iteration))
                                os.makedirs(save_path, exist_ok=True)
                                save_im = np.transpose(gt_image.detach().cpu().numpy(), (1, 2, 0))
                                save_im = (save_im * 255).astype(np.uint8)
                                imageio.imsave(
                                    os.path.join(save_path, f"{n_times-1}_{iteration}_{j}_gt.png"),
                                    save_im)
                                save_im = np.transpose(image.squeeze().detach().cpu().numpy(), (1, 2, 0))
                                save_im = (save_im * 255).astype(np.uint8)
                                imageio.imsave(
                                    os.path.join(save_path, f"{n_times-1}_{iteration}_{j}_render.png"),
                                    save_im)

            ema_loss_for_log = ema_loss_for_log * 0.99 + loss.item() * 0.01
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point": f"{self.gaussians.num_gaussians}"})
                progress_bar.update(10)

        self.last_iters = iteration

        return self.gaussians, self.simulator

    def save(self):

        iteration = self.static_iterations + self.opt_params.iterations
        print("Saving Gaussians")
        point_cloud_path = os.path.join(self.save_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(point_cloud_path)

        meshnet_path = os.path.join(self.save_path, 'meshnet')
        os.makedirs(meshnet_path, exist_ok=True)
        self.simulator.save(os.path.join(meshnet_path, 'model-' + str(iteration) + '.pt'))
