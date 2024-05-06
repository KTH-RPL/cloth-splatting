# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from typing import Optional

import h5py
import numpy
import torch
import numpy as np
import torch_geometric.utils

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
# noinspection PyUnresolvedReferences
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

from meshnet.data_utils import compute_mesh, compute_edge_features, farthest_point_sampling
from meshnet.model_utils import NodeType

from scene.gaussian_model import GaussianModel
# TODO Add deformation table to GNN
# TODO Add regularisation terms for GNN
# TODO Figure out standard constraint function
# TODO Add lr scheduler for GNN
# TODO make train_setup function somehow included in the train loop.


class GaussianMesh(GaussianModel):

    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)

        self.node_type = torch.empty(0)
        self.edge_displacement = torch.empty(0)
        self.edge_norm = torch.empty(0)
        self.mesh = torch_geometric.data.Data()

    @torch.no_grad()
    def make_mesh(self, vertices):
        """
        Create the mesh from the gaussians.
        """
        # TODO Check the whole detach and clone mess

        self.mesh = compute_mesh(vertices)
        self.node_type = torch.full(self.mesh.pos.shape[0:1], fill_value=NodeType.CLOTH, device="cuda")

        self.edge_displacement, self.edge_norm = compute_edge_features(self.mesh.pos.clone().detach(),
                                                                       self.mesh.edge_index.clone().detach())
        self.mesh.edge_attr = torch.hstack(
            (self.edge_displacement.clone().detach().to(torch.float32).contiguous(),
             self.edge_norm.clone().detach().to(torch.float32).contiguous())
        )

    def get_xyz(self, deformed_vertices: Optional[torch.Tensor] = None):
        if deformed_vertices is not None:
            return deformed_vertices
        return self._xyz

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        super().create_from_pcd(pcd, spatial_lr_scale)
        self.make_mesh(self.get_xyz())

    def load_ply(self, path):
        super().load_ply(path)
        self.make_mesh(self.get_xyz())

    def prune(self, max_grad, min_opacity, extent, max_screen_size):
        super().prune(max_grad, min_opacity, extent, max_screen_size)
        self.make_mesh(self.get_xyz())

    def densify(self, max_grad, min_opacity, extent, max_screen_size):
        super().densify(max_grad, min_opacity, extent, max_screen_size)
        self.make_mesh(self.get_xyz())


class MultiGaussianMesh(GaussianModel):

    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)

        self.node_type = torch.empty(0)

        self.face_bary = torch.empty(0)          # [n_gaussians, 3]
        self.face_ids = torch.empty(0)           # [n_gaussians, 3]
        self.face_offset = torch.empty(0)        # [n_gaussians, 1]       # TODO Implement face offset

        self.edge_displacement = torch.empty(0)
        self.edge_norm = torch.empty(0)
        self.mesh = torch_geometric.data.Data()

    @torch.no_grad()
    def make_mesh(self, vertices):
        """
        Create the mesh from the gaussians.
        """
        # TODO Check the whole detach and clone mess

        self.mesh = compute_mesh(vertices)
        self.node_type = torch.full(self.mesh.pos.shape[0:1], fill_value=NodeType.CLOTH, device="cuda")

        self.edge_displacement, self.edge_norm = compute_edge_features(self.mesh.pos.clone().detach(),
                                                                       self.mesh.edge_index.clone().detach())
        self.mesh.edge_attr = torch.hstack(
            (self.edge_displacement.clone().detach().to(torch.float32).contiguous(),
             self.edge_norm.clone().detach().to(torch.float32).contiguous())
        )

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.pos_gradient_accum = torch.zeros((self.face_bary.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz().shape[0], 1), device="cuda")

        l = [
            {'params': [self.face_bary], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "face_bary"},
            {'params': [self.face_offset], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "face_offset"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    @property
    def num_gaussians(self) -> int:
        return self.face_ids.shape[0]

    def get_xyz(self, deformed_vertices: Optional[torch.Tensor] = None):
        vertice_ids = self.mesh.face[:, self.face_ids].transpose(0, 1)   # [num_gauss, 3 (vert))]
        if deformed_vertices is not None:
            assert deformed_vertices.shape[0] == self.mesh.pos.shape[0]
            face_pos = deformed_vertices[vertice_ids, :]
        else:
            face_pos = self.mesh.pos[vertice_ids, :]                            # [num_gauss, 3 (vert), 3 (xyz)]

        norm_bary = self.face_bary / self.face_bary.sum(dim=1, keepdim=True)             # [num_gauss, 3 (bary)]
        pos = (norm_bary.unsqueeze(1) @ face_pos).squeeze(1)                             # [num_gauss, 3 (xyz)
        return pos

    def create_mesh_from_vertices(self, vertices, spatial_lr_scale: float):

        self.make_mesh(vertices)

        n_faces = self.mesh.face.shape[1]

        self.face_bary = nn.Parameter(
            ((torch.ones((n_faces, 3), dtype=torch.float, device="cuda")) / 3).requires_grad_(True)
        )
        self.face_offset = nn.Parameter(
            torch.zeros(n_faces, 1, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.face_ids = torch.arange(0, n_faces, dtype=torch.long, device="cuda")

        shs = np.random.random((n_faces, 3)) / 255.0

        point_coords = self.get_xyz().detach().cpu().numpy()
        pcd = BasicPointCloud(points=point_coords, colors=SH2RGB(shs), normals=np.zeros((n_faces, 3)))
        self.create_from_pcd(pcd, spatial_lr_scale)

    def prune_points(self, mask):

        valid_points_mask = ~mask

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self.face_bary = optimizable_tensors["face_bary"]
        self.face_offset = optimizable_tensors["face_offset"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.face_ids = self.face_ids[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self.pos_gradient_accum = self.pos_gradient_accum[valid_points_mask]


    def densification_postfix(self, new_face_bary, new_face_offset, new_face_ids, new_features_dc, new_features_rest,
                              new_opacities, new_scaling, new_rotation):
        d = {"face_bary": new_face_bary,
             "face_offset": new_face_offset,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self.face_bary = optimizable_tensors["face_bary"]
        self.face_offset = optimizable_tensors["face_offset"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.face_ids = torch.cat((self.face_ids, new_face_ids), dim=0)
        self.pos_gradient_accum = torch.zeros((self.face_bary.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.face_bary.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.face_bary.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.face_bary.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)

        new_face_ids = self.face_ids[selected_pts_mask].repeat(N)
        new_face_bary = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.face_bary[selected_pts_mask].repeat(N, 1)
        new_face_offset = self.face_offset[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_face_bary, new_face_offset, new_face_ids, new_features_dc,
                                   new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_face_ids = self.face_ids[selected_pts_mask]
        new_face_bary = self.face_bary[selected_pts_mask]
        new_face_offset = self.face_offset[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_face_bary, new_face_offset, new_face_ids,
                                   new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def construct_list_of_attributes(self):
        l = super().construct_list_of_attributes()
        l.extend(['b1', 'b2', 'b3', 'o', 'id'])
        return l

    def save_ply(self, path):
        save_path = os.path.join(path, "point_cloud.ply")
        mkdir_p(os.path.dirname(save_path))

        face_ids = self.face_ids.unsqueeze(1).detach().cpu().numpy()
        face_bary = self.face_bary.detach().cpu().numpy()
        face_offset = self.face_offset.detach().cpu().numpy()
        xyz = self.get_xyz().detach().cpu().numpy()
        normals = np.zeros_like(face_bary)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(face_ids.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, face_bary, face_offset, face_ids), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(save_path)

        mesh = self.mesh.to_dict()
        with h5py.File(os.path.join(path, "mesh.hdf5"), "w") as f:
            for key, value in mesh.items():
                f.create_dataset(key, data=value.detach().cpu().numpy())

    def load_ply(self, path):
        super().load_ply(path)

        plydata = PlyData.read(os.path.join(path, 'point_cloud.ply'))
        self.face_ids = torch.tensor(plydata.elements[0]['id'], dtype=torch.long, device='cuda')

        face_bary = np.stack((np.asarray(plydata.elements[0]['b1']),
                              np.asarray(plydata.elements[0]['b2']),
                                    np.asarray(plydata.elements[0]['b3'])), axis=1)
        self.face_bary = nn.Parameter(torch.tensor(face_bary, dtype=torch.float, device='cuda').requires_grad_(True))

        self.face_offset = nn.Parameter(torch.tensor(plydata.elements[0]['o'], dtype=torch.float, device='cuda').requires_grad_(True))

        mesh_data = h5py.File(os.path.join(path, 'mesh.hdf5'), 'r')
        self.mesh = torch_geometric.data.Data(
            pos=torch.tensor(mesh_data['pos'][:], device='cuda'),
            norm=torch.tensor(mesh_data['norm'][:], device='cuda'),
            face=torch.tensor(mesh_data['face'][:], device='cuda'),
            edge_index=torch.tensor(mesh_data['edge_index'][:], device='cuda'))

        self.edge_displacement, self.edge_norm = compute_edge_features(self.mesh.pos.clone().detach(),
                                                                       self.mesh.edge_index.clone().detach())
        self.mesh.edge_attr = torch.hstack(
            (self.edge_displacement.clone().detach().to(torch.float32).contiguous(),
             self.edge_norm.clone().detach().to(torch.float32).contiguous())
        )

        self.node_type = torch.full(self.mesh.pos.shape[0:1], fill_value=NodeType.CLOTH, device="cuda")

