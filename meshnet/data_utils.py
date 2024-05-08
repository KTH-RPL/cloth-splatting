import numpy as np
import json
import os
import torch
import torch_geometric

import scipy
from scipy.spatial import cKDTree, Delaunay
import glob
from meshnet.viz import plot_mesh, plot_pcd_list, create_gif
import h5py
import math

# function to load json data
def load_traj(data_path):
    traj = np.load(data_path, allow_pickle=True)['traj']
    return traj

def load_sim_traj(data_path, load_keys=['pos', 'vel', 'actions', 'trajectory_params', 'gripper_pos', 'pick', 'place']):
    file_path = glob.glob(os.path.join(data_path, '*h5'))[0]
    with h5py.File(file_path, 'r') as f:
        data = {key: np.array(f[key]) for key in load_keys}
    return data



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



def process_traj(traj, dt, k=3, delaunay=False, subsample=False, num_samples=300, sim_data=False, norm_threshold=0.01):
    trajectory_data = {"pos": [], "velocity": [], "node_type": [], "edge_index": [], "edge_displacement": [], 'edge_norm': [], 'edge_norm': [], 'edge_faces': []}

    if subsample:
        sampled_points_indeces = farthest_point_sampling(traj[0], num_samples)
    else:
        sampled_points_indeces = np.arange(traj[0].shape[0])

    edge_index, faces = compute_edges_index(traj[0][sampled_points_indeces], k=k, delaunay=delaunay, sim_data=sim_data, norm_threshold=norm_threshold)
    # plot_mesh(traj[0][sampled_points_indeces], edge_index.T)

    for time_idx in range(1, traj.shape[0]):
        # Position at current and previous timestep
        pos_current = traj[time_idx][sampled_points_indeces]
        pos_previous = traj[time_idx - 1][sampled_points_indeces]

        # Compute velocity
        velocity = (pos_current - pos_previous) / dt

        # Store data
        trajectory_data["pos"].append(pos_current)
        trajectory_data["velocity"].append(velocity)
        # so far we only have one node type
        node_type = 0
        # make shape of node type as (velocity.shape[0], 1)
        node_type = np.ones((velocity.shape[0], 1)) * node_type
        trajectory_data["node_type"].append(node_type)

        # compute edge features
        displacement, norm = compute_edge_features(torch.tensor(pos_current, dtype=torch.float), edge_index)

        # prunte edges at the first time step
        if time_idx == 1:
            edge_index = edge_index[:,(norm < norm_threshold)[:, 0]]
            displacement, norm = compute_edge_features(torch.tensor(pos_current, dtype=torch.float), edge_index)

        trajectory_data["edge_index"].append(edge_index)
        trajectory_data["edge_displacement"].append(displacement)
        trajectory_data["edge_norm"].append(norm)
        trajectory_data["edge_faces"].append(faces)

    # Handle the first position manually if needed (e.g., set initial velocity to zero)
    trajectory_data["pos"].insert(0, traj[0][sampled_points_indeces])
    trajectory_data["velocity"].insert(0, np.zeros_like(trajectory_data["velocity"][0]))  # Assuming initial velocity is zero
    trajectory_data["node_type"].insert(0, trajectory_data["node_type"][0])
    trajectory_data["edge_index"].insert(0, edge_index)
    trajectory_data["edge_displacement"].insert(0, trajectory_data["edge_displacement"][0])
    trajectory_data["edge_norm"].insert(0, trajectory_data["edge_norm"][0])
    trajectory_data["edge_faces"].insert(0, trajectory_data["edge_faces"][0])

    # iterate over all keys to make it numpy array
    for key in trajectory_data.keys():
        if key in ["edge_index", "edge_displacement", "edge_norm", "edge_faces"]:#
            trajectory_data[key] = torch.stack(trajectory_data[key])
        else:
            trajectory_data[key] = np.array(trajectory_data[key])
    return trajectory_data



def compute_edges_index(points, k=3, delaunay=False, sim_data=False, norm_threshold=0.01):
    if delaunay:
        if sim_data:
            points2d = points[:, [0,2]]
        else:
            points2d = points[:, :2]
        tri = Delaunay(points2d)
        edges = set()
        faces = []
        for simplex in tri.simplices:
            valid_face = True
            current_edges = []

            for i in range(3):
                p1, p2 = simplex[i], simplex[(i + 1) % 3]
                edge = (min(p1, p2), max(p1, p2))
                current_edges.append(edge)
                # Calculate the norm (distance) between the points
                norm = np.linalg.norm(points2d[p1] - points2d[p2])

                # Check if the edge meets the threshold condition
                if norm_threshold is not None and norm > norm_threshold:
                    valid_face = False
                else:
                    edges.add(edge)

            # Add the face if all edges are valid
            if valid_face:
                faces.append(simplex)

        edge_index = np.asarray(list(edges))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        # Convert faces list to a tensor
        faces = torch.tensor(faces, dtype=torch.long).t().contiguous()
        return edge_index, faces
    else:
        # Use a k-D tree for efficient nearest neighbors computation
        tree = cKDTree(points)
        # For simplicity, we find the 3 nearest neighbors; you can adjust this number
        _, indices = tree.query(points, k=k+1)

        # Skip the first column because it's the point itself
        edge_index = np.vstack({tuple(sorted([i, j])) for i, row in enumerate(indices) for j in row[1:]})
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return edge_index


def compute_mesh(points: torch.Tensor) -> torch_geometric.data.Data:
    """
    Uses a Delaunay triangulation to compute of the mesh.
    The triangulation will only consider x and y coordinates.

    Args:
        points: [n, 3] array of points

    Returns: torch_geometric.data.Data object with pos, face, and edge_index attributes.

    """

    pos = points[:, :2].cpu().numpy()

    tri = scipy.spatial.Delaunay(pos, qhull_options='QJ')

    face = torch.from_numpy(tri.simplices).t().contiguous().to(points.device, dtype=torch.long)
    mesh = torch_geometric.data.Data(pos=points, face=face)
    mesh = torch_geometric.transforms.FaceToEdge(remove_faces=False)(mesh)
    mesh = torch_geometric.transforms.GenerateMeshNormals()(mesh)

    return mesh


def compute_edge_features(points, edge_index):
    # Compute relative displacement (edge features)
    displacement = points[edge_index[1]] - points[edge_index[0]]
    norm = torch.norm(displacement, dim=1, keepdim=True)

    return displacement, norm



if __name__ == '__main__':
    # Load trajectory
    traj = load_traj('../data/dataset/final_scene_1_rgb-005/final_scene_1_gt_eval.npz')
    points = traj[0]
    edge_index = compute_edges_index(points, k=3)

    # create a gif of the trajectory sequence
    save_data_path = './data/figs'
    image_files = []
    for t in range(traj.shape[0]):
        file_name = os.path.join(save_data_path, f'mesh_t{t}.png')
        points = traj[t]
        plot_mesh(points, np.transpose(edge_index), center_plot=np.asarray([0,0,0]), white_bkg=True, save_fig=True, file_name=file_name)
        image_files.append(file_name)

    # Create GIF
    gif_path = "trajectory_mesh.gif"
    create_gif(image_files, './data/gifs/trajectory_mesh.gif', fps=10)
    # plot the trajectory
    print(edge_index)

