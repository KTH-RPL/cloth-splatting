from meshnet.data_utils import compute_mesh, farthest_point_sampling
import torch
import numpy as np
import h5py
import os
import shutil
import torch_geometric

# Helper script to transfer the ground-truth trajectory to a sequence of noisy meshes, for evaluation purposes.

def save_mesh(mesh, path, time_id):
    mesh_dict = mesh.to_dict()
    idx = str(time_id).zfill(3)
    with h5py.File(os.path.join(path, f"mesh_predictions/mesh_{idx}.hdf5"), "w") as f:
        for key, value in mesh_dict.items():
            print(key)
            f.create_dataset(key, data=value.detach().cpu().numpy())

#path = '../data/cloth_test/cloth_fold_scene_1_noise'
path = '../data/final_scenes/scene_1_noise'
mesh_path = os.path.join(path, 'mesh_predictions')
if os.path.exists(mesh_path):
    shutil.rmtree(mesh_path)
os.makedirs(mesh_path)

traj = np.load(path + '/final_scene_1_gt_eval.npz')['traj']
# traj = np.load(path + '/trajectory.npz')['arr_0']
traj = traj[:, farthest_point_sampling(traj[0], 200)]

points = torch.tensor(traj[0], dtype=torch.float32)
mesh = compute_mesh(points)
save_mesh(mesh, path, 0)
last_points = points

for time_id, points in enumerate(traj[1:]):
    points = torch.tensor(points, dtype=torch.float32)
    mod_points = 0.10*points + 0.90 * last_points
    mod_points = torch.normal(mod_points, std=0.01)
    mesh = torch_geometric.data.Data(pos=mod_points, edge_index=mesh.edge_index, face=mesh.face)
    mesh = torch_geometric.transforms.GenerateMeshNormals()(mesh)
    save_mesh(mesh, path, time_id+1)
    last_points = mod_points
shutil.copy(os.path.join(mesh_path, f"mesh_000.hdf5"), os.path.join(path, 'init_mesh.hdf5'))
