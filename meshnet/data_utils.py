import numpy as np
import json
import os
import torch
import matplotlib.pyplot as plt
import torch_geometric
from meshnet.viz import plot_mesh, plot_pcd_list, create_gif
import scipy


# function to load json data
def load_traj(data_path):
    traj = np.load(data_path, allow_pickle=True)['traj']
    return traj


def farthest_point_sampling(points, num_samples):
    """
    Selects a subset of points using the Farthest Point Sampling (FPS) algorithm.

    Parameters:
    - points: A NumPy array of shape (N, D) where N is the number of points and D is the dimensionality.
    - num_samples: The number of points to select.

    Returns:
    - A NumPy array of the selected points.
    """
    # Initialize an array to hold indices of the selected points
    selected_indices = np.zeros(num_samples, dtype=int)
    # The first point is selected randomly
    selected_indices[0] = np.random.randint(len(points))
    # Initialize a distance array to track the shortest distance of each point to the selected set
    distances = np.full(len(points), np.inf)

    # Loop to select points
    for i in range(1, num_samples):
        # Update the distances based on the newly added point
        dist_to_new_point = np.linalg.norm(points - points[selected_indices[i - 1]], axis=1)
        distances = np.minimum(distances, dist_to_new_point)
        # Select the point with the maximum distance to the set of selected points
        selected_indices[i] = np.argmax(distances)

    # Return the selected points
    return selected_indices


def process_traj(traj, dt, k=3, delaunay=False, subsample=False, num_samples=300):
    trajectory_data = {"pos": [], "velocity": [], "node_type": [], "edge_index": [], "edge_displacement": [], 'edge_norm': []}

    if subsample:
        sampled_points_indeces = farthest_point_sampling(traj[0], num_samples)
    else:
        sampled_points_indeces = np.arange(traj[0].shape[0])

    mesh = compute_mesh(torch.tensor(traj[0][sampled_points_indeces]))
    # plot_mesh(traj[0][sampled_points_indeces], edge_index.T)

    for time_idx in range(1, traj.shape[0]):
        # Position at current and previous timestep
        pos_current = traj[time_idx][sampled_points_indeces]
        pos_previous = traj[time_idx - 1][sampled_points_indeces]

        # Compute velocity
        velocity = (pos_current - pos_previous) / dt

        # Store data
        trajectory_data["pos"].append(pos_previous)
        trajectory_data["velocity"].append(velocity)
        # so far we only have one node type
        node_type = 0
        # make shape of node type as (velocity.shape[0], 1)
        node_type = np.ones((velocity.shape[0], 1)) * node_type
        trajectory_data["node_type"].append(node_type)

        # compute edge features
        displacement, norm = compute_edge_features(torch.tensor(pos_current, dtype=torch.float), mesh.edge_index)
        trajectory_data["edge_index"].append(mesh.edge_index)
        trajectory_data["edge_displacement"].append(displacement)
        trajectory_data["edge_norm"].append(norm)

    # Handle the first position manually if needed (e.g., set initial velocity to zero)
    trajectory_data["pos"].insert(0, traj[0][sampled_points_indeces])
    trajectory_data["velocity"].insert(0, np.zeros_like(trajectory_data["velocity"][0]))  # Assuming initial velocity is zero
    trajectory_data["node_type"].insert(0, trajectory_data["node_type"][0])
    trajectory_data["edge_index"].insert(0, mesh.edge_index)
    trajectory_data["edge_displacement"].insert(0, trajectory_data["edge_displacement"][0])
    trajectory_data["edge_norm"].insert(0, trajectory_data["edge_norm"][0])

    # iterate over all keys to make it numpy array
    for key in trajectory_data.keys():
        if key in ["edge_index", "edge_displacement", "edge_norm"]:#
            trajectory_data[key] = torch.stack(trajectory_data[key])
        else:
            trajectory_data[key] = np.array(trajectory_data[key])
    return trajectory_data


def compute_mesh(points: torch.Tensor) -> torch_geometric.data.Data:
    """
    Uses a Delaunay triangulation to compute of the mesh.
    The triangulation will only consider x and y coordinates.

    Args:
        points: [n, 3] array of points

    Returns: torch_geometric.data.Data object with pos, face, and edge_index attributes.

    """

    pos = points[:, :2].cpu().numpy()

    tri = scipy.spatial.Delaunay(pos)

    face = torch.from_numpy(tri.simplices).t().contiguous().to(points.device, dtype=torch.long)
    mesh = torch_geometric.data.Data(pos=points, face=face)
    mesh = torch_geometric.transforms.FaceToEdge(remove_faces=False)(mesh)

    return mesh


def compute_edge_features(points, edge_index):
    # Compute relative displacement (edge features)
    displacement = points[edge_index[1]] - points[edge_index[0]]
    norm = torch.norm(displacement, dim=1, keepdim=True)

    return displacement, norm



# if __name__ == '__main__':
#     # Load trajectory
#     traj = load_traj('../data/dataset/final_scene_1_rgb-005/final_scene_1_gt_eval.npz')
#     points = traj[0]
#     mesh = make_mesh(points)
#     trajdge_index = mesh.edge_index
#
#     # create a gif of the trajectory sequence
#     save_data_path = './data/figs'
#     image_files = []
#     for t in range(traj.shape[0]):
#         file_name = os.path.join(save_data_path, f'mesh_t{t}.png')
#         points = traj[t]
#         plot_mesh(points, np.transpose(edge_index), center_plot=np.asarray([0,0,0]), white_bkg=True, save_fig=True, file_name=file_name)
#         image_files.append(file_name)
#
#     # Create GIF
#     gif_path = "trajectory_mesh.gif"
#     create_gif(image_files, './data/gifs/trajectory_mesh.gif', fps=10)
#     # plot the trajectory
#     print(edge_index)

