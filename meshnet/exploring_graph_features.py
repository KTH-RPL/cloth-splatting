
from meshnet.data_utils import load_sim_traj, process_traj, get_env_trajs_path, flip_trajectory
from manipulation.utils.data_collection import get_meshes_paths
import numpy as np
from meshnet.gaussian_sampling import sample_gaussian_means
import os
from torch_geometric.data import Data
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch_geometric.transforms as T
from tqdm import tqdm
import imageio
from torch_scatter import scatter_add
import json


def plot_mesh_with_colors(veritces, edges, colors=None, faces=None, reference_vertex=None, points=None, num_samples_per_triangle=3, keypoints_idx=None,
                             elev=30, azim=30, center_plot=None, return_image=False, white_bkg=False, save_fig=False, file_name='mesh.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points with colors indicating distance
    scatter = ax.scatter(veritces[:, 0], veritces[:, 1], veritces[:, 2], c=colors, marker='o', s=20, alpha=1)
    
     # Plot points with colors indicating distance
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='x', s=20, alpha=1)
    
    if keypoints_idx is not None:   
        for k in keypoints_idx.keys():
            point = veritces[keypoints_idx[k]]
            ax.scatter(point[0], point[1], point[2], c='r', marker='o', s=50, alpha=1)
        # keypoints_pos = veritces[keypoints_idx]
        # ax.scatter(keypoints_pos[:, 0], keypoints_pos[:, 1], keypoints_pos[:, 2], c='g', marker='o', s=50)
    
    # Plot edges
    for edge in edges:
        p1, p2 = veritces[edge[0]], veritces[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='grey', linewidth=0.5)  # Use a neutral color for edges
        
    # Sample and plot points on faces
    # vertex_colors = plt.cm.viridis(distances[reference_vertex] / max_distance)  # Normalize and map to colormap

    for face in faces.T:
        triangle_vertices = veritces[face]
        barycentric_coords = np.random.dirichlet([1, 1, 1], size=num_samples_per_triangle)
        
        barycentric_coords_exp = np.expand_dims(barycentric_coords, axis=2)
        barycentric_coords_exp = np.repeat(barycentric_coords_exp, 3, axis=2)

        # add one dimension to the vertices and repeat for each num_sample_per_triange
        triangle_vertices = np.expand_dims(triangle_vertices, axis=0)
        triangle_vertices = np.repeat(triangle_vertices, num_samples_per_triangle, axis=0)

        # get the sample points as multiplitcation of the barycentric coordinates and the vertices
        sampled_points = np.einsum('ijm,ijm->ijm', barycentric_coords_exp, triangle_vertices)
        sampled_points = sampled_points.sum(axis=-2)


        # Determine colors of sampled points by interpolating vertex colors
        sampled_colors =  np.repeat(colors[None, face], num_samples_per_triangle, axis=0)
        weighted_colors = np.einsum('ij,ijk->ik', barycentric_coords, sampled_colors)
        ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], c=np.clip(weighted_colors, a_max=1, a_min=0), marker='o', s=5)

    
    # Set axis limits based on the points' range
    if center_plot is None:
        min_values = veritces.min(axis=0)
        max_values = veritces.max(axis=0)
        center_plot = (min_values + max_values) / 2
        max_range = (max_values - min_values).max()
        ax.set_xlim(center_plot[0] - max_range / 2, center_plot[0] + max_range / 2)
        ax.set_ylim(center_plot[1] - max_range / 2, center_plot[1] + max_range / 2)
        ax.set_zlim(center_plot[2] - max_range / 2, center_plot[2] + max_range / 2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # elev and azim
    ax.view_init(elev=elev, azim=azim)

    # Adding a color bar to indicate distance scales
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.08)
    cbar.set_label('Distance from vertex {}'.format(reference_vertex))

    if white_bkg:
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # remove grid and axis
        ax.axis('off')
        ax.grid(False)
        
    if return_image:
        canvas = fig.canvas
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)
        plt.close()
        return img

    if save_fig:
        plt.savefig(file_name)  # Save the figure to a file
        plt.close(fig)  # Close the figure to free memory
    else:
        plt.show()


def plot_mesh_with_distances(veritces, edges, distances=None, faces=None, reference_vertex=None, points=None, num_samples_per_triangle=3, keypoints_idx=None,
                             elev=30, azim=30, center_plot=None, return_image=False, white_bkg=False, save_fig=False, file_name='mesh.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate colors based on distance to the reference vertex
    max_distance = np.nanmax(distances[reference_vertex])  # Avoid considering inf values
    colors = plt.cm.viridis(distances[reference_vertex] / max_distance)  # Normalize and map to colormap
    
    # Plot points with colors indicating distance
    scatter = ax.scatter(veritces[:, 0], veritces[:, 1], veritces[:, 2], c=colors, marker='o', s=20, alpha=1)
    
     # Plot points with colors indicating distance
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='x', s=20, alpha=1)
    
    if keypoints_idx is not None:   
        for k in keypoints_idx.keys():
            point = veritces[keypoints_idx[k]]
            ax.scatter(point[0], point[1], point[2], c='r', marker='o', s=50, alpha=1)
        # keypoints_pos = veritces[keypoints_idx]
        # ax.scatter(keypoints_pos[:, 0], keypoints_pos[:, 1], keypoints_pos[:, 2], c='g', marker='o', s=50)
    
    # Plot edges
    for edge in edges:
        p1, p2 = veritces[edge[0]], veritces[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='grey', linewidth=0.5)  # Use a neutral color for edges
        
    # Sample and plot points on faces
    # vertex_colors = plt.cm.viridis(distances[reference_vertex] / max_distance)  # Normalize and map to colormap

    for face in faces.T:
        triangle_vertices = veritces[face]
        barycentric_coords = np.random.dirichlet([1, 1, 1], size=num_samples_per_triangle)
        
        barycentric_coords_exp = np.expand_dims(barycentric_coords, axis=2)
        barycentric_coords_exp = np.repeat(barycentric_coords_exp, 3, axis=2)

        # add one dimension to the vertices and repeat for each num_sample_per_triange
        triangle_vertices = np.expand_dims(triangle_vertices, axis=0)
        triangle_vertices = np.repeat(triangle_vertices, num_samples_per_triangle, axis=0)

        # get the sample points as multiplitcation of the barycentric coordinates and the vertices
        sampled_points = np.einsum('ijm,ijm->ijm', barycentric_coords_exp, triangle_vertices)
        sampled_points = sampled_points.sum(axis=-2)


        # Determine colors of sampled points by interpolating vertex colors
        sampled_colors =  np.repeat(colors[None, face], num_samples_per_triangle, axis=0)
        weighted_colors = np.einsum('ij,ijk->ik', barycentric_coords, sampled_colors)
        ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], c=np.clip(weighted_colors, a_max=1, a_min=0), marker='o', s=5)

    
    # Set axis limits based on the points' range
    if center_plot is None:
        min_values = veritces.min(axis=0)
        max_values = veritces.max(axis=0)
        center_plot = (min_values + max_values) / 2
        max_range = (max_values - min_values).max()
        ax.set_xlim(center_plot[0] - max_range / 2, center_plot[0] + max_range / 2)
        ax.set_ylim(center_plot[1] - max_range / 2, center_plot[1] + max_range / 2)
        ax.set_zlim(center_plot[2] - max_range / 2, center_plot[2] + max_range / 2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # elev and azim
    ax.view_init(elev=elev, azim=azim)

    # Adding a color bar to indicate distance scales
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.08)
    cbar.set_label('Distance from vertex {}'.format(reference_vertex))

    if white_bkg:
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # remove grid and axis
        ax.axis('off')
        ax.grid(False)
        
    if return_image:
        canvas = fig.canvas
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)
        plt.close()
        return img

    if save_fig:
        plt.savefig(file_name)  # Save the figure to a file
        plt.close(fig)  # Close the figure to free memory
    else:
        plt.show()

def floyd_warshall(n, edge_index, positions):
    # Initialize the distance matrix with infinity
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0)

    # Fill initial distances based on direct connections
    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i]
        dist[u, v] = dist[v, u] = 1 # np.linalg.norm(positions[u] - positions[v])  # Assuming undirected graph

    # Floyd-Warshall algorithm to compute shortest paths
    for k in tqdm(range(n)):
        for i in range(n):
            for j in range(n):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    return dist

def compute_distance_matrix(positions, edge_index):
    # Assuming the 'pos' are the node positions and 'edge_index' contains the edge information
    n = positions.shape[0]  # Number of vertices

    # Calculate the distance matrix
    distance_matrix = floyd_warshall(n, edge_index, positions)
    
    return distance_matrix

def plot_mesh_with_keypoints(veritces, edges,  keypoints,  elev=30, azim=30, center_plot=None, return_image=False, white_bkg=False, save_fig=False, file_name='mesh.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate colors based on distance to the reference vertex
    # Plot points with colors indicating distance
    scatter = ax.scatter(veritces[:, 0], veritces[:, 1], veritces[:, 2], c='b', marker='o', alpha=0.2, s=20)
    
     # Plot points with colors indicating distance
     # define a unique color for each keypoint
    num_keypoints = len(keypoints.keys())
    keypoints_colors = list(plt.cm.viridis(np.linspace(0, 1, num_keypoints)))
    # same list but with gist_rainbow colormap
    keypoints_colors = list(plt.cm.gist_rainbow(np.linspace(0, 1, num_keypoints)))
    
    i = 0
    for k in keypoints.keys():
        point = veritces[keypoints[k]]
        ax.scatter(point[0], point[1], point[2], color=keypoints_colors[i], marker='x', alpha=1, s=100, label=k, linewidths=2)
        i += 1
    

    # Plot edges
    for edge in edges:
        p1, p2 = veritces[edge[0]], veritces[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='grey', linewidth=0.5)  # Use a neutral color for edges
        
    # Sample and plot points on faces
    # vertex_colors = plt.cm.viridis(distances[reference_vertex] / max_distance)  # Normalize and map to colormap

    # Set axis limits based on the points' range
    if center_plot is None:
        min_values = veritces.min(axis=0)
        max_values = veritces.max(axis=0)
        center_plot = (min_values + max_values) / 2
        max_range = (max_values - min_values).max()
        ax.set_xlim(center_plot[0] - max_range / 2, center_plot[0] + max_range / 2)
        ax.set_ylim(center_plot[1] - max_range / 2, center_plot[1] + max_range / 2)
        ax.set_zlim(center_plot[2] - max_range / 2, center_plot[2] + max_range / 2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # elev and azim
    ax.view_init(elev=elev, azim=azim)

    # put the legend away from the plot
    plt.legend()
    ax.legend(loc='center left', bbox_to_anchor=(0.86, 0.65))

    if white_bkg:
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # remove grid and axis
        ax.axis('off')
        ax.grid(False)
        
    if return_image:
        canvas = fig.canvas
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)
        plt.close()
        return img

    if save_fig:
        plt.savefig(file_name)  # Save the figure to a file
        plt.close(fig)  # Close the figure to free memory
    else:
        plt.show()
        
def get_edges(traj_processed, t, transform, pos):
    faces = traj_processed['edge_faces'][t]
    # create graph
    graph = Data(
    face=faces.to(torch.int64).contiguous(),
    pos=torch.from_numpy(pos).to(torch.float32).contiguous(),
    )
    # obtain edges
    graph = transform(graph)
    edge_index = graph.edge_index
    
    return edge_index

def create_laplacian(edge_index, num_nodes):
    # Create the degree matrix
    row, col = edge_index
    deg = scatter_add(torch.ones_like(row), row, dim=0, dim_size=num_nodes)

    # Create the adjacency matrix as a sparse tensor
    adj = torch.zeros((num_nodes, num_nodes))
    adj[row, col] = 1

    # Laplacian Matrix
    lap = torch.diag(deg) - adj
    return lap

def features_diffusion(pos, keypoints_idx, keypoints_features, L, feature_dim=4, steps=100, epsilon=0.01):
    # Example to setup the graph - placeholder for your actual data setup
    num_nodes = pos.shape[0]
    features = torch.zeros((num_nodes, feature_dim))  # 4 for RGB colors

    # Set initial conditions for specific nodes
    torch_keypoint_features = torch.from_numpy(keypoints_features).float()
    features[keypoints_idx] = torch_keypoint_features



    # Parameters for diffusion
    epsilon = epsilon
    steps = steps

    # Diffusion process
    node_changes = []
    all_features = []
    for _ in range(steps):
        update = epsilon * torch.matmul(L, features)
        features = features - update
        # Reapply fixed color conditions
        features[keypoints_idx] = torch_keypoint_features
        # 
        norm_update = torch.norm(update[:, :3], dim=1)
        # keep only the 10 hihgest change to average
        highest = torch.topk(norm_update, 10, largest=True)
        node_changes.append(highest[0].mean().numpy())
        all_features.append(features.numpy())
        
    return all_features, node_changes


def apply_softmax(distances):
    # Apply softmax on the negative of the distances to make closer distances have higher weights
    # Note: Use a small epsilon to avoid division by zero for self-distances
    epsilon = 1e-5
    inverse_distances = 1 / (distances + epsilon)
    return np.exp(inverse_distances) / np.sum(np.exp(inverse_distances), axis=1, keepdims=True)



if __name__=='__main__':
    obj_type = 'TOWEL'#'TOWEL''TSHIRT'
    mesh_idx = 1        # 0, 1
    folder_idx = "{:05}".format(mesh_idx)
    dataset = 'test_dataset_0415'
    data_paths = f'./sim_datasets/{dataset}/{obj_type}/{folder_idx}' 
    flat_mesh_dataset = '0411_train'
    env_all_trajs = get_env_trajs_path(data_paths)    
    mesh_paths = get_meshes_paths(obj_type, flat_mesh_dataset)
    gif_path = os.path.join('.', 'data', 'features_gif', dataset, obj_type, folder_idx)
    os.makedirs(gif_path, exist_ok=True)
    
    action_steps = 1
    dt = 1,
    k = 3
    delaunay = True
    subsample = True
    num_samples = 150
    sim_data = True
    load_keys=['pos']
    original_mesh_path =mesh_paths[mesh_idx]
    
    transform = T.FaceToEdge()
    
    for all_trajs in env_all_trajs:
        for data_path in all_trajs:   
            traj_data = load_sim_traj(data_path, action_steps, load_keys)
            if sim_data:
                traj_data = flip_trajectory(traj_data, load_keys)
                
            
            original_mesh_path =mesh_paths[mesh_idx]
            keypoints = json.load(open(original_mesh_path.replace(".obj", ".json")))["keypoint_vertices"]    
            # create a dictionary with the keypoint index and the position
            keypoints_pos = {k: traj_data['pos'][0][keypoints[k]] for k in keypoints.keys()}  
                
            traj = traj_data['pos']
            traj_processed = process_traj(traj, dt, k, delaunay, 
                            subsample=subsample, num_samples=num_samples, 
                            sim_data=False, norm_threshold=0.1)
            
    
            
            if subsample:
                # find the new indeces of hte keypoints by the idx that is closes to the original keypoint position
                keypoints_idx = {k: np.argmin(np.linalg.norm(traj_processed['pos'][0] - keypoints_pos[k], axis=1)) for k in keypoints.keys()}
                keypoints = {k: keypoints_idx[k] for k in keypoints.keys()}
            keypoints_idx = [keypoints[k] for k in keypoints.keys()]
                
            num_keypoints = len(keypoints.keys())
            # same list but with gist_rainbow colormap
            keypoints_colors = np.asarray(list(plt.cm.gist_rainbow(np.linspace(0, 1, num_keypoints))))
                
            # obj_traj_path = os.path.join(data_path, 'obj')
            # os.makedirs(obj_traj_path, exist_ok=True)
            imgs = []
            return_image = True
            plot_keypoints = False
            d_matrix = None
            for t, pos in enumerate(traj_processed['pos']):
                faces = traj_processed['edge_faces'][t]
                if t == 0:
                    edge_index = get_edges(traj_processed, t, transform, pos)
                    if plot_keypoints:
                        plot_mesh_with_keypoints(pos, edge_index.T,  keypoints,  elev=90, azim=90, center_plot=None, return_image=False, white_bkg=True, save_fig=False, file_name='mesh.png')

                    
                    # get distance matrix
                    d_matrix = compute_distance_matrix(pos, edge_index.numpy())
                    d_keypoints_matrix = d_matrix[:, [keypoints[k] for k in keypoints.keys()]]/d_matrix.max()
                    d_keypoints_matrix_soft = 1 - d_keypoints_matrix
                    d_keypoints_matrix_soft = apply_softmax(d_keypoints_matrix_soft)
                    
                    weighted_colors = np.einsum('ij,jm->im', d_keypoints_matrix_soft, keypoints_colors)/num_keypoints
                    weighted_colors[:, 3] = 1
                    
                    # Create the Laplacian matrix
                    L = create_laplacian(edge_index, pos.shape[0])
    
                    min_colors = np.repeat(keypoints_colors[None], d_matrix.shape[0], 0)[np.arange(d_matrix.shape[0]), np.argmin(d_keypoints_matrix, axis=1)]
                    all_diffused_colors, color_changes = features_diffusion(pos, keypoints_idx, keypoints_colors, L, steps=1000, epsilon=0.1)
                    diffused_colors = all_diffused_colors[-1]
                    # change matoplotlib  backend
                    # plt.switch_backend('Agg')
                    # switch back to visualization
                    # plt.switch_backend('TkAgg')
                    # create a fig with the diffusion process
                    diff_images = []
                    for i, colors_t in enumerate(all_diffused_colors[::10]):
                        ref = 1
                        num_samples_per_triangle = 50
                        img = plot_mesh_with_colors(pos, edge_index.T, colors=colors_t, faces=faces.numpy(), reference_vertex=ref, points=np.asarray([pos[ref]]), num_samples_per_triangle=num_samples_per_triangle, keypoints_idx=keypoints,
                             elev=55, azim=-120, center_plot=None, return_image=True, white_bkg=True, save_fig=False, file_name='mesh.png')
                        diff_images.append(img)
                    imageio.mimsave(f'{gif_path}/mesh_RGB_diffusion_process_{num_samples_per_triangle}.gif', diff_images, loop=0)
                    
                    
                
                # plot mesh
                ref = 1
                num_samples_per_triangle = 0
                if plot_keypoints:
                    # img = plot_mesh_with_distances(pos, edge_index.T, d_matrix, faces.numpy(), num_samples_per_triangle=num_samples_per_triangle, keypoints_idx=keypoints, 
                    #                             reference_vertex=ref, points=np.asarray([pos[ref]]), center_plot=None, white_bkg=True, elev=55, azim=-120, return_image=return_image) 
                    
                    # img = plot_mesh_with_colors(pos, edge_index.T, colors=min_colors, faces=faces.numpy(), reference_vertex=ref, points=np.asarray([pos[ref]]), num_samples_per_triangle=num_samples_per_triangle, keypoints_idx=keypoints,
                    #          elev=55, azim=-120, center_plot=None, return_image=True, white_bkg=True, save_fig=False, file_name='mesh.png')
                    
                    img = plot_mesh_with_colors(pos, edge_index.T, colors=diffused_colors, faces=faces.numpy(), reference_vertex=ref, points=np.asarray([pos[ref]]), num_samples_per_triangle=num_samples_per_triangle, keypoints_idx=keypoints,
                             elev=55, azim=-120, center_plot=None, return_image=True, white_bkg=True, save_fig=False, file_name='mesh.png')
                else:
                    # img = plot_mesh_with_distances(pos, edge_index.T, d_matrix, faces.numpy(), num_samples_per_triangle=num_samples_per_triangle, keypoints_idx=None, 
                    #         reference_vertex=ref, points=np.asarray([pos[ref]]), center_plot=None, white_bkg=True, elev=55, azim=-120, return_image=return_image) 
                    
                    # img = plot_mesh_with_colors(pos, edge_index.T, colors=min_colors, faces=faces.numpy(), reference_vertex=ref, points=np.asarray([pos[ref]]), num_samples_per_triangle=num_samples_per_triangle, keypoints_idx=None,
                    #          elev=55, azim=-120, center_plot=None, return_image=True, white_bkg=True, save_fig=False, file_name='mesh.png')
                    
                    img = plot_mesh_with_colors(pos, edge_index.T, colors=diffused_colors, faces=faces.numpy(), reference_vertex=ref, points=np.asarray([pos[ref]]), num_samples_per_triangle=num_samples_per_triangle, keypoints_idx=None,
                             elev=55, azim=-120, center_plot=None, return_image=True, white_bkg=True, save_fig=False, file_name='mesh.png')
                imgs.append(img)
                # plot_mesh_with_distances(traj_processed['pos'][-1], edge_index.T, d_matrix, faces.numpy(), num_samples_per_triangle=0, reference_vertex=ref, points=np.asarray([traj_processed['pos'][-1][ref]]), center_plot=None, white_bkg=False, save_fig=False, file_name='mesh.png')
            
            # create and save gif from images
            # imageio.mimsave(f'{gif_path}/mesh_samples_{num_samples_per_triangle}.gif', imgs, loop=0)
            # imageio.mimsave(f'{gif_path}/mesh_RGB_samples_{num_samples_per_triangle}.gif', imgs, loop=0)
            imageio.mimsave(f'{gif_path}/mesh_RGB_diffusion_samples_{num_samples_per_triangle}.gif', imgs, loop=0)
            
            # same but with looping gif
            
            