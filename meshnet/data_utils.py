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

def load_sim_traj(data_path, action_steps=1, load_keys=['pos', 'vel', 'actions', 'trajectory_params', 'gripper_pos', 'pick', 'place']):
    file_path = glob.glob(os.path.join(data_path, '*h5'))[0]
    with h5py.File(file_path, 'r') as f:
        if action_steps  == 1:
            data = {key: np.array(f[key]) for key in load_keys}
        else:
            data = {}
            for key in load_keys:
                if key in ['trajectory_params', 'pick', 'place']:
                    data[key] = np.array(f[key])
                elif key in ['pos', 'vel','gripper_pos',]:
                    # take one evey action steps
                    data[key] = np.array(f[key][::action_steps])
                elif key == 'actions':
                    # take one evey action steps but sum the actions, 
                    actions = np.array(f[key])#.reshape(-1, action_steps, 3)
                    # this code accounts for the scenarios where the number of actions are not divisible by action_steps
                    if actions.shape[0]%action_steps == 0:
                        data[key] = actions.reshape(-1, action_steps, 3).sum(1)
                    else:
                        last_actions = actions[-(actions.shape[0]%action_steps):].sum(0)[None, :]
                        pre_actions = actions[:-(actions.shape[0]%action_steps)].reshape(-1, action_steps, 3).sum(1)
                        data[key] = np.concatenate([pre_actions, last_actions], 0)             

    return data

def get_env_trajs_path(data_paths):
    all_trajs = glob.glob(os.path.join(data_paths, '*'))
    all_trajs.sort()
    
    subfolder =  glob.glob(os.path.join(all_trajs[0], '*'))
    subfolder.sort()
    # if '.hf' not in  glob.glob(os.path.join(all_trajs[0], '*')):
    if all(['.h5' not in subfolder for subfolder in glob.glob(os.path.join(all_trajs[0], '*'))]):
        # expand the list of trajs with all subfolders
        env_all_trajs = []
        for traj in all_trajs:
            new_trajs = glob.glob(os.path.join(traj, '*'))
            new_trajs.sort()
            env_all_trajs.append(new_trajs)
    else:
        env_all_trajs = [all_trajs]
    return env_all_trajs

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

def get_data_traj(data_path, load_keys, params, sim_data=False):  
    dt, k, delaunay, subsample, num_samples, input_length_sequence, action_steps = params             
    traj_data = load_sim_traj(data_path, action_steps, load_keys)
    if sim_data:
        traj_data = flip_trajectory(traj_data, load_keys)
        
    traj = traj_data['pos']
    trajectory_data = process_traj(traj, dt, k, delaunay, 
                                    subsample=subsample, num_samples=num_samples, 
                                    sim_data=False, norm_threshold=0.1) # sim_data is false if we flip the trajectory before
    
    # shit the actions as we store them as a_t, s_t+1
    trajectory_data['actions'] = traj_data['actions'][1:]
    trajectory_data["actions"] = np.concatenate([ np.zeros_like(trajectory_data["actions"][0])[None, :], trajectory_data['actions']],0)

    # gripper data
    trajectory_data['gripper_pos'] = traj_data['gripper_pos']
    trajectory_data['gripper_vel'] = (traj_data['gripper_pos'][1:] - traj_data['gripper_pos'][:-1]) / dt
    trajectory_data["gripper_vel"] = np.concatenate([np.zeros_like(trajectory_data["gripper_vel"][0])[None, :],  trajectory_data['gripper_vel']],0)
    
    # useful info
    trajectory_data['pick'] = traj_data['pick']
    trajectory_data['place'] = traj_data['place']
    trajectory_data['trajectory_params'] = traj_data['trajectory_params']   
    
    # find at time 0 the particle that is the closest to the grasped one by closest euclidean distance
    grasped_particle_idx = np.argmin(np.linalg.norm(trajectory_data['pos'][0] - traj_data['pick'], axis=1))
    # update the node type of the grasped particle
    trajectory_data['node_type'][:, grasped_particle_idx] = 1
    
    trajectory_data['grasped_particle'] = grasped_particle_idx # traj_data['grasped_particle']   
    
    if input_length_sequence > 1:
        for d in ['actions', 'pos', 'velocity', 'gripper_pos', 'gripper_vel', 'node_type', 'edge_index', 'edge_displacement', 'edge_norm']:
            trajectory_data[d] = expand_init_data(trajectory_data[d], input_length_sequence)
            
    return trajectory_data
            
def expand_init_data(data, input_length_sequence):
    # expand the data to have the same length of the input_length_sequence
    # for the first element of the data
    for i in range(input_length_sequence - 1):
        # assert if data is torch or not
        if isinstance(data, torch.Tensor):
            data = torch.cat([data[0:1],  data],0)
        else:
            data = np.concatenate([data[0:1],  data],0)
        # data.insert(0, data[0])
    return data
                    

def flip_trajectory(traj, load_keys):
    flip_keys = ['pos', 'vel', 'actions', 'gripper_pos', 'pick', 'place']

    for k in flip_keys:
        if k in load_keys:
            if traj[k].ndim == 2:
                traj[k] = traj[k][:, [0, 2, 1]]
            elif traj[k].ndim == 3:
                traj[k] = traj[k][:, :, [0, 2, 1]]
            else:
                traj[k] = traj[k][[0, 2, 1]]
                
    return traj
            
            


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
        faces = torch.tensor(np.asarray(faces), dtype=torch.long).t().contiguous()
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

def load_mesh_from_h5py(path):
    mesh_data = h5py.File(path, 'r')
    mesh = torch_geometric.data.Data(
        pos=torch.tensor(mesh_data['pos'][:], device='cuda'),
        norm=torch.tensor(mesh_data['norm'][:], device='cuda'),
        face=torch.tensor(mesh_data['face'][:], device='cuda'),
        edge_index=torch.tensor(mesh_data['edge_index'][:], device='cuda'))
    return mesh


def axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle representation to quaternion.
    Args:
        axis: The axis [n, 3]
        angle: The angle [n]

    Returns: The quaternion [n, 4] (XYZW)
    """
    qxyz = axis * torch.sin(angle / 2).unsqueeze(1)
    qw = torch.cos(angle / 2).unsqueeze(1)
    return torch.cat([qxyz, qw], dim=1)


def vertice_rotation(normals_a: torch.Tensor, normals_b: torch.Tensor) -> torch.Tensor:
    """
    Compute the rotation between sets of normals (elemt-wise).
    Args:
        normals_a: Initial normals [n, 3]
        normals_b: Rotated normals [n, 3]

    Returns: The quaternion [n, 4] (XYZW)

    """
    # Compute cross product to find the rotation axis
    cross_prod = torch.cross(normals_a, normals_b, dim=1)
    # Compute the dot product to find the cosine of the angle
    dot_prod = torch.sum(normals_a * normals_b, dim=1)
    # Compute the angle between the vectors
    angles = torch.acos(torch.clamp(dot_prod, -1.0, 1.0))
    axes = cross_prod / torch.linalg.norm(cross_prod, dim=1, keepdim=True)
    return axis_angle_to_quat(axes, angles)


def compute_barycentric_coordinates(points, triangles):
    """
    Estimates the barycentric coordinates of a set of points with respect to a set of triangles.

    Args:
        points: [n, 3 (xyz)] array of points
        triangles: [m, 3, 3 (xyz)] array of triangles

    Returns: [n, 3] array of barycentric coordinates

    """
    # Triangles vertices
    A = triangles[:, 0, :]
    B = triangles[:, 1, :]
    C = triangles[:, 2, :]

    # Vectors from A to B and A to C
    AB = B - A
    AC = C - A
    AP = points - A

    # Compute the dot products
    dot00 = torch.sum(AC * AC, dim=1)
    dot01 = torch.sum(AC * AB, dim=1)
    dot02 = torch.sum(AC * AP, dim=1)
    dot11 = torch.sum(AB * AB, dim=1)
    dot12 = torch.sum(AB * AP, dim=1)

    # Compute the denominator
    denom = dot00 * dot11 - dot01 * dot01

    # Compute the barycentric coordinates
    v = (dot11 * dot02 - dot01 * dot12) / denom
    w = (dot00 * dot12 - dot01 * dot02) / denom
    u = 1.0 - v - w

    return torch.stack([u, v, w], dim=1)


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

