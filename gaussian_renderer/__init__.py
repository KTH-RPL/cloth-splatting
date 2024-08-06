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
from typing import NamedTuple

import torch
import math
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene_reconstruction.gaussian_mesh import MultiGaussianMesh
from meshnet.meshnet_network import ResidualMeshSimulator
from utils.sh_utils import eval_sh


class RenderResults(NamedTuple):
    render: torch.Tensor
    viewspace_points: torch.Tensor
    visibility_filter: torch.Tensor
    radii: torch.Tensor
    depth: torch.Tensor
    means3D_deform: torch.Tensor
    vertice_deform: torch.Tensor
    shadows_mean: torch.Tensor
    shadows_std: torch.Tensor
    projections: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    shadows: torch.Tensor
    vertice_projections: torch.Tensor


def render(viewpoint_camera, pc: MultiGaussianMesh, simulator: ResidualMeshSimulator, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None, log_deform_path=None, no_shadow=False, render_static=False,
           project_vertices=False) -> RenderResults:
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz(), dtype=pc.get_xyz().dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz()
    # add deformation to each points
    # deformation = pc.get_deformation
    means3D = pc.get_xyz()
    means2D = screenspace_points
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
        scales = None
    else:
        cov3D_precomp = None
        scales = pc.get_scaling
    opacity = pc.get_opacity

    shadow_scalars = None

    time = torch.tensor(viewpoint_camera.time).to(pc.mesh.pos.device).repeat(pc.mesh.pos.shape[0], 1)

    if render_static:
        vertice_deform = pc.mesh.pos
        means3D_deform = means3D
        rotations_deform = pc.get_rotation()
    else:
        vertice_deform = simulator(time_vector=time)
        means3D_deform = pc.get_xyz(vertice_deform)
        rotations_deform = pc.get_rotation(vertice_deform)
    rotations_final = rotations_deform
    means3D_final = means3D_deform
    scales_final = scales
    opacity_final = opacity

    # means3D_final = torch.zeros_like(means3D)
    # rotations_final = torch.zeros_like(rotations)
    # scales_final = torch.zeros_like(scales)
    # opacity_final = torch.zeros_like(opacity)
    # means3D_final[deformation_point] =  means3D_deform
    # rotations_final[deformation_point] =  rotations_deform
    # scales_final[deformation_point] =  scales_deform
    # opacity_final[deformation_point] = opacity_deform
    # means3D_final[~deformation_point] = means3D[~deformation_point]
    # rotations_final[~deformation_point] = rotations[~deformation_point]
    # scales_final[~deformation_point] = scales[~deformation_point]
    # opacity_final[~deformation_point] = opacity[~deformation_point]

    if log_deform_path is not None:
            np.savez(log_deform_path,
                     means3D=means3D.cpu().numpy(),
                     means3D_deform=means3D_final.cpu().numpy(),
                     vertice_deform=vertice_deform.cpu().numpy(),
                     rotations=rotations_final.cpu().numpy(),
                     vertice_rotations=pc.get_vertice_rotation(vertice_deform).cpu().numpy())

    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    if no_shadow:
        shadow_scalars = None
    if override_color is None:
        if shadow_scalars is not None: # we compute colors in python to multiply with our shadow scalars
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz() - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if shadow_scalars is not None:
        # shadow_scalars = [N,1]
        # colors_precomp = [N,3]
        # element-wise multiplication
        colors_precomp = colors_precomp * shadow_scalars.repeat(1,3)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)

    def projections(cam, points):
        points_h = torch.cat([points, torch.ones_like(points[:, 0:1])], dim=1).T
        cam_transform = cam.full_proj_transform.to(points_h.device).T

        projections = cam_transform.matmul(points_h)
        projections = projections / projections[3, :]

        projections = projections[:2].T
        H, W = int(cam.image_height), int(cam.image_width)

        projections_cam = torch.zeros_like(projections).to(projections.device)
        projections_cam[:, 0] = ((projections[:, 0] + 1.0) * W - 1.0) * 0.5
        projections_cam[:, 1] = ((projections[:, 1] + 1.0) * H - 1.0) * 0.5
        return projections_cam

    # projecting to cam frame for later use in optic flow
    gaussian_projections = projections(viewpoint_camera, means3D_final)
    vertice_projections = projections(viewpoint_camera, vertice_deform) if project_vertices else None
    shadows_mean = None
    shadows_std = None

    if shadow_scalars is not None:
        shadows_mean = torch.mean(shadow_scalars)
        shadows_std = torch.std(shadow_scalars)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return RenderResults(render=rendered_image,
                         viewspace_points=screenspace_points,
                         visibility_filter=radii > 0,
                         radii=radii,
                         depth=depth,
                         means3D_deform=means3D_final,
                         vertice_deform=vertice_deform,
                         shadows_mean=shadows_mean,
                         shadows_std=shadows_std,
                         projections=gaussian_projections,
                         rotations=rotations_deform,
                         opacities=opacity_final,
                         shadows=shadow_scalars,
                         vertice_projections=vertice_projections)

