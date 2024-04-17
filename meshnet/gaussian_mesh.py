# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import torch_geometric.utils

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

from meshnet.data_utils import compute_mesh, compute_edge_features
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
    def make_mesh(self):
        """
        Create the mesh from the gaussians.
        """
        # TODO Check the whole detach and clone mess

        self.mesh = compute_mesh(self._xyz)

        self.node_type = torch.full(self.mesh.pos.shape[0:1], fill_value=NodeType.CLOTH, device="cuda")

        edge_displacement, edge_norm = compute_edge_features(self.mesh.pos.clone().detach(),
                                                             self.mesh.edge_index.clone().detach())
        self.edge_displacement = edge_displacement
        self.edge_norm = edge_norm

        self.mesh.edge_attr = torch.hstack(
            (edge_displacement.clone().detach().to(torch.float32).contiguous(),
             edge_norm.clone().detach().to(torch.float32).contiguous())
        )

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        super().create_from_pcd(pcd, spatial_lr_scale)
        self.make_mesh()

    def load_ply(self, path):
        super().load_ply(path)
        self.make_mesh()

    def prune(self, max_grad, min_opacity, extent, max_screen_size):
        super().prune(max_grad, min_opacity, extent, max_screen_size)
        self.make_mesh()

    def densify(self, max_grad, min_opacity, extent, max_screen_size):
        super().densify(max_grad, min_opacity, extent, max_screen_size)
        self.make_mesh()


