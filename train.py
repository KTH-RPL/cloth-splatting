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

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene_reconstruction.scene import Scene

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

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def flow_loss(all_projections=None, visibility_filter_list=None, viewpoint_cams=None):
    # flow frame i-1
    flow_0 = all_projections[1] - all_projections[0]
    # mask s.t. only visible points are used for flow
    mask_visibility = visibility_filter_list[0].squeeze(0) & visibility_filter_list[1].squeeze(0)
    # mask s.t. only points that are in [H,W] are used for flow
    mask_in_image = (all_projections[0, :, 0] >= 0) & (all_projections[0, :, 0] < viewpoint_cams[0].image_height) & (
                all_projections[0, :, 1] >= 0) & \
                    (all_projections[0, :, 1] < viewpoint_cams[0].image_width)

    mask = mask_visibility & mask_in_image
    flow_0 = flow_0[mask]
    projections_0 = all_projections[0][mask]
    raft_flow_0 = torch.tensor(viewpoint_cams[0].flow[0], device="cuda")
    raft_flow_0_indexed = raft_flow_0[:, projections_0[:, 0].long(), projections_0[:, 1].long()].T

    ## flow frame i
    flow_1 = all_projections[2] - all_projections[1]
    # mask s.t. only visible points are used for flow
    mask_visibility = visibility_filter_list[1].squeeze(0) & visibility_filter_list[2].squeeze(0)
    # mask s.t. only points that are in [H,W] are used for flow
    mask_in_image = (all_projections[:, 1] >= 0) & (all_projections[:, 1] < viewpoint_cams[1].image_width) & (
                all_projections[:, 2] >= 0) & \
                    (all_projections[:, 2] < viewpoint_cams[2].image_height)

    mask = mask_visibility & mask_in_image
    flow_1 = flow_1[mask]

    # raft_flow_0 shape 2 x H x W
    # all_projections[0] N x 2
    # index raft_flow_0 with all_projections[0]

    print(raft_flow_0_indexed.shape)
    print(flow_0.shape)
    print(raft_flow_0_indexed[:3])
    raft_flow_1 = viewpoint_cams[1].flow


def scene_reconstruction(dataset, opt: OptimizationParams, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians: MultiGaussianMesh, simulator: MeshSimulator, scene: Scene, stage, tb_writer, train_iter, timer, user_args=None):
    first_iter = 0

    # Initialize optimizers
    gaussians.training_setup(opt)

    # TODO Maybe attach this to the meshnet class
    meshnet_optimizer = torch.optim.Adam(
        simulator.parameters(),
        lr=3e-4
    )

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.video_cameras
    o3d_knn_dists, o3d_knn_indices, knn_weights = None, None, None

    for iteration in range(first_iter, final_iter + 1):
        static = opt.static_reconst and iteration < opt.static_reconst_iteration
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, ts = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, simulator, pipe, background, scaling_modifer, render_static=static)[
                        "render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # TODO make this a hyperparameter
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.train_cameras
            batch_size = 1
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size, shuffle=True, num_workers=32,
                                                collate_fn=list)
            loader = iter(viewpoint_stack_loader)
        if opt.dataloader:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader")
                batch_size = 1
                loader = iter(viewpoint_stack_loader)
        else:
            idx = randint(0, len(viewpoint_stack) - 1)  # picking a random viewpoint
            viewpoint_cams = viewpoint_stack[idx]  # returning 3 subsequence timesteps

        if static:
            viewpoint_cams = [viewpoint_stack.get_one_item(iteration % len(viewpoint_stack), 0)]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        all_means_3D_deform = []
        all_projections = []
        all_rotations = []
        all_opacities = []
        all_shadows = []
        all_shadows_std = []
        all_vertice_deform = []
        masks = [] if viewpoint_cams[0].mask is not None else None

        for viewpoint_cam in viewpoint_cams:

            render_pkg = render(viewpoint_cam, gaussians, simulator, pipe, background,
                                no_shadow=user_args.no_shadow, render_static=static)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            gt_image = viewpoint_cam.original_image.cuda()
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            if masks is not None:
                masks.append(viewpoint_cam.mask.cuda().unsqueeze(0))
            all_means_3D_deform.append(render_pkg["means3D_deform"][None, :, :])
            all_vertice_deform.append(render_pkg["vertice_deform"][None, :, :])
            all_projections.append(render_pkg["projections"][None, :, :])
            all_rotations.append(norm_quat(render_pkg["rotations"][None, :, :]))
            all_opacities.append(render_pkg["opacities"][None, :])
            shadows = render_pkg["shadows"]
            if shadows is not None:
                all_shadows.append(shadows[None, :])

            all_shadows_std.append(render_pkg["shadows_std"])

        all_projections = torch.cat(all_projections, 0)
        all_rotations = torch.cat(all_rotations, 0)
        all_opacities = torch.cat(all_opacities, 0)
        all_means_3D_deform = torch.cat(all_means_3D_deform, 0)
        all_vertice_deform = torch.cat(all_vertice_deform, 0)

        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images, 0)
        gt_image_tensor = torch.cat(gt_images, 0)
        mask_tensor = torch.cat(masks, 0) if masks is not None else None
        # Loss
        Ll1 = l1_loss(image_tensor, gt_image_tensor, mask_tensor)
        # Ll1 = l2_loss(image, gt_image)

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()

        # norm

        loss = Ll1

        if user_args.use_wandb and stage == "fine":
            wandb.log({"train/psnr": psnr_, "train/loss": loss}, step=iteration)
            wandb.log({"train/num_gaussians": gaussians.num_gaussians}, step=iteration)

        n_cams = len(viewpoint_cams)

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
            edge_displacement = all_vertice_deform[:, gaussians.mesh.edge_index[1]] - all_vertice_deform[:, gaussians.mesh.edge_index[0]]
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

        # TODO Double check if SSIM with mask works.
        if opt.lambda_dssim != 0:
            if mask_tensor is None:
                ssim_loss = ssim(image_tensor, gt_image_tensor)
                loss += opt.lambda_dssim * (1.0 - ssim_loss)
            else:
                ssim_map = ssim(image_tensor, gt_image_tensor, return_map=True)
                loss += opt.lambda_dssim * ((1.0 - ssim_map) * mask_tensor).mean()
        # if opt.lambda_lpips != 0:
        #     lpipsloss = lpips_loss(image_tensor, gt_image_tensor, lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss

        loss.backward()

        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians.num_gaussians
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point": f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, gaussians, simulator, [pipe, background], stage,
                            user_args=user_args, save_test_images=user_args.save_test_images)
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                point_cloud_path = os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration))
                gaussians.save_ply(point_cloud_path)

                meshnet_path = os.path.join(scene.model_path, 'meshnet')
                os.makedirs(meshnet_path, exist_ok=True)
                simulator.save(os.path.join(meshnet_path, 'model-' + str(iteration) + '.pt'))

                # with open(os.path.join(scene.model_path, "time.txt"), 'a') as file:
                #     file.write(f"{stage} {iteration} {timer.get_elapsed_time()}\n")

            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 1) \
                        or (iteration < 3000 and iteration % 50 == 1) \
                        or (iteration < 10000 and iteration % 100 == 1) \
                        or (iteration < 60000 and iteration % 100 == 1):
                    render_training_image(scene, gaussians, video_cams, render, pipe, background, stage, iteration - 1,
                                          timer.get_elapsed_time())
                    # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration * (
                                opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after) / (
                                            opt.densify_until_iter)
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration * (
                                opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after) / (
                                            opt.densify_until_iter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    print("reset opacity")
                    gaussians.reset_opacity()

            if iteration % opt.bary_cleanup:
                gaussians.cleanup_barycentric_coordinates()

            # Optimizer step
            if iteration < opt.iterations:
                if static:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    meshnet_optimizer.zero_grad()
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    meshnet_optimizer.step()
                    meshnet_optimizer.zero_grad()

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           os.path.join(scene.model_path, "chkpnt" + str(iteration) + ".pth"))


def training(dataset, hyper, opt, pipe, meshnet_params, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, expname, user_args=None):

    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    timer = Timer()

    dataset.model_path = args.model_path

    scene = Scene(dataset, load_coarse=None, user_args=user_args)

    # load simulator
    mesh_pos = torch.concat([mesh.pos.unsqueeze(0) for mesh in scene.mesh_predictions], dim=0)
    simulator = ResidualMeshSimulator(mesh_pos, device='cuda')
    simulator.train()

    gaussians = MultiGaussianMesh(dataset.sh_degree)
    gaussians.from_mesh(scene.initial_mesh, scene.cameras_extent, opt.gaussian_init_factor)

    timer.start()

    if not opt.no_coarse:
        scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, simulator, scene, "coarse", tb_writer,
                             opt.coarse_iterations, timer, user_args=user_args)
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, simulator, scene, "fine", tb_writer,
                         opt.iterations, timer, user_args=user_args)


def prepare_output_and_logger(expname):
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, elapsed, testing_iterations, scene: Scene, gaussians: MultiGaussianMesh, simulator: MeshSimulator,
                    rander_args, stage, user_args=None, save_test_images=True):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test',
                               'cameras': [scene.test_camera_individual[idx % len(scene.test_camera_individual)]
                                           for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras': [
                                  scene.train_camera_individual[idx % len(scene.train_camera_individual)] for
                                  idx in range(10, 5000, 299)]})
        # individual to get only a single view at a time
        for config_id, config in enumerate(validation_configs):
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(
                        render(viewpoint, gaussians, simulator, *rander_args, no_shadow=user_args.no_shadow)[
                            "render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            stage + "/" + config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None],
                            global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                stage + "/" + config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    if save_test_images:
                        save_path = os.path.join(scene.model_path, "test_renders".format(iteration))
                        os.makedirs(save_path, exist_ok=True)
                        save_im = np.transpose(gt_image.detach().cpu().numpy(), (1, 2, 0))
                        save_im = (save_im * 255).astype(np.uint8)
                        imageio.imsave(os.path.join(save_path, f"{iteration}_{config['name']}_{config_id}_gt.png"), save_im)
                        save_im = np.transpose(image.squeeze().detach().cpu().numpy(), (1, 2, 0))
                        save_im = (save_im * 255).astype(np.uint8)
                        imageio.imsave(os.path.join(save_path, f"{iteration}_{config['name']}_{config_id}_render.png"), save_im)

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                if user_args.use_wandb and config['name'] == "test" and stage == "fine":
                    wandb.log({"test/psnr": psnr_test, "test/loss": l1_test}, step=iteration)

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(stage + "/" + config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage + "/" + config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", gaussians.get_opacity, iteration)

            tb_writer.add_scalar(f'{stage}/total_points', gaussians.num_gaussians, iteration)
            #tb_writer.add_scalar(f'{stage}/deformation_rate',
            #                     gaussians._deformation_table.sum() / gaussians.get_xyz().shape[0], iteration)
            #tb_writer.add_histogram(f"{stage}/scene/motion_histogram",
            #                        gaussians._deformation_accum.mean(dim=-1) / 100, iteration, max_bins=500)

        torch.cuda.empty_cache()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    mp = MeshnetParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i * 500 for i in range(0, 120)])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--three_steps_batch", type=bool, default=True)
    parser.add_argument("--save_test_images", type=bool, default=True)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="test_project")
    parser.add_argument("--wandb_name", type=str, default="test_name")

    parser.add_argument("--view_skip", default=1, type=int)
    parser.add_argument("--time_skip", type=int, default=1)

    ###
    # model parameters
    ###

    # disable shadow net
    parser.add_argument("--no_shadow", action="store_true")

    # regularization
    # momentum term
    parser.add_argument("--reg_iter", default=5000, type=int)
    parser.add_argument("--knn_update_iter", default=1000, type=int)
    parser.add_argument("--lambda_momentum", default=0.0, type=float)

    # isometric loss
    parser.add_argument("--lambda_isometric", default=0.0, type=float)


    # shadow loss
    parser.add_argument("--lambda_shadow_mean", default=0.0, type=float)
    parser.add_argument("--lambda_shadow_delta", default=0.0, type=float)

    parser.add_argument("--lambda_momentum_rotation", default=0.0, type=float)

    parser.add_argument("--lambda_spring", default=0.0, type=float)

    parser.add_argument("--lambda_w", default=2000, type=float)
    parser.add_argument("--k_nearest", default=20, type=int)
    parser.add_argument("--single_cam_video", action="store_true",
                        help='Only render from the first camera for the video viz')
    args = parser.parse_args(sys.argv[1:])

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name)
        wandb.config.update(args)

    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), mp.extract(args), args.test_iterations,
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname,
             user_args=args)

    # All done
    print("\nTraining complete.")
