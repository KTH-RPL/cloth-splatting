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

import os
import random
import json
from typing import NamedTuple
import numpy as np

import torch_geometric.data

from meshnet.data_utils import load_mesh_from_h5py

from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import SceneInfo, readCamerasFromTransforms, getNerfppNorm, read_timeline, \
    generateCamerasFromTransforms, CameraInfo
from meshnet.gaussian_mesh import MultiGaussianMesh
from scene.dataset import FourDGSdataset, MDNerfDataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset
import glob


class MeshSceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    maxtime: int
    initial_mesh: torch_geometric.data.Data
    mesh_predictions: list[torch_geometric.data.Data]


def read_cloth_scene_info(path, white_background, eval, extension=".png", time_skip=None, view_skip=None,
                          single_cam_video=False):
    if not os.path.exists(path):
        raise FileNotFoundError("Path does not exist: {}".format(path))

    timestamp_mapper, max_time = read_timeline(path)
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension,
                                                timestamp_mapper, time_skip=time_skip, view_skip=view_skip,
                                                split='train')
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension,
                                               timestamp_mapper, time_skip=time_skip, view_skip=view_skip, split='test')
    print("Generating Video Transforms")

    video_path = os.path.join(path, "video.json")
    video_cam_infos = None
    if os.path.exists(video_path):
        video_cam_infos = readCamerasFromTransforms(path, "video.json", white_background, extension, timestamp_mapper,
                                                    time_skip=1, view_skip=1, split='video')

    if video_cam_infos is None:
        video_cam_infos = generateCamerasFromTransforms(path, "transforms_train.json", extension, max_time,
                                                        time_skip=time_skip, single_cam_video=single_cam_video)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    mesh_path = os.path.join(path, 'mesh_predictions')
    initial_mesh = load_mesh_from_h5py(os.path.join(path, 'init_mesh.hdf5'))
    mesh_prediction_path = sorted(glob.glob(os.path.join(mesh_path, 'mesh_*.hdf5')))
    mesh_prediction_path = mesh_prediction_path[::time_skip] if time_skip is not None else mesh_prediction_path
    mesh_predictions = [load_mesh_from_h5py(mesh) for mesh in mesh_prediction_path]

    scene_info = MeshSceneInfo(
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        video_cameras=video_cam_infos,
        nerf_normalization=nerf_normalization,
        maxtime=max_time,
        initial_mesh=initial_mesh,
        mesh_predictions=mesh_predictions
    )

    return scene_info, initial_mesh, mesh_predictions


class Scene:
    gaussians: MultiGaussianMesh

    def __init__(self, args: ModelParams, load_iteration=None, shuffle=True,
                 resolution_scales=(1.0), load_coarse=False, user_args=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        time_skip = None
        view_skip = None
        if user_args is not None:
            time_skip = user_args.time_skip
            view_skip = user_args.view_skip

        (scene_info, self.initial_mesh,
         self.mesh_predictions) = read_cloth_scene_info(args.source_path, args.white_background, args.eval,
                                                        time_skip=time_skip, view_skip=view_skip,
                                                        single_cam_video=user_args.single_cam_video)

        self.maxtime = scene_info.maxtime
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        if user_args.three_steps_batch:
            print("Loading Training Cameras, MDNeRF")
            self.train_cameras = MDNerfDataset(scene_info.train_cameras, args)

            print("Loading Test Cameras, MDNeRF")
            self.test_cameras = MDNerfDataset(scene_info.test_cameras, args)
        else:
            print("Loading Training Cameras, 4DGS")
            self.train_cameras = FourDGSdataset(scene_info.train_cameras, args)
            print("Loading Test Cameras, 4DGS")
            self.test_cameras = FourDGSdataset(scene_info.test_cameras, args)

        self.train_camera_individual = FourDGSdataset(scene_info.train_cameras, args)
        self.test_camera_individual = FourDGSdataset(scene_info.test_cameras, args)

        print("Loading Video Cameras")
        self.video_cameras = cameraList_from_camInfos(scene_info.video_cameras, -1, args)

        # TODO Implement the GaussianModel loading
        # if self.loaded_iter:
        #     self.gaussians.load_ply(os.path.join(self.model_path,
        #                                          "point_cloud",
        #                                          "iteration_" + str(self.loaded_iter)))
        # else:
        #     self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
