import numpy as np
from gym.spaces import Box
import pyflex
from manipulation.envs.gym_env import GymEnv
from manipulation.envs.cloth_env import ClothEnv
from copy import deepcopy
import h5py
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from meshnet.exploring_graph_features import compute_distance_matrix, features_diffusion, create_laplacian, plot_mesh_with_colors, plot_mesh_with_distances
from meshnet.data_utils import process_traj, farthest_point_sampling, compute_edges_index
from pyflex_utils.se3 import SE3Container
from pyflex_utils.utils import (
    create_pyflex_cloth_scene_config, 
    DeformationConfig, 
    load_cloth_mesh_in_simulator, 
    PyFlexStepWrapper, 
    ClothParticleSystem,
    wait_until_scene_is_stable,
    ParticleGrasperObserver,
    ParticleGrasper
)
import pathlib
import json
from manipulation.envs.utils import (
    compute_intrinsics,
    get_matrix_world_to_camera
)
import random
import os 
import argparse
from manipulation.utils.trajectory_gen import (
    visualize_multiple_trajectories,
    visualize_trajectory_and_actions,
    generate_bezier_trajectory,
    compute_actions_from_trajectory,
    get_action_traj
)
from manipulation.utils.data_collection import get_meshes_paths, store_data_by_name, sample_cloth_params
from tqdm import tqdm
from multiprocessing import Process
from meshnet.viz import plot_mesh
import matplotlib.pyplot as plt

def update_observations(data_dict, obs, action, grasp, camera_names):
    for cam in camera_names:
        data_dict[f'{cam}_rgbd'].append(obs[cam]['rgbd'])
    data_dict["pos"].append(obs["pos"])
    data_dict["vel"].append(obs["vel"])
    data_dict["gripper_pos"].append(obs["gripper_pos"])
    data_dict["done"].append(obs["done"])
    data_dict['actions'].append(action)
    data_dict['grasp'].append(grasp)
    
    return data_dict
    
class HalfFoldConfig:
    def __init__(self, ):
        self.num_pick_places = 2
        self.keypoints = ['corner0', 'corner1', 'corner2', 'corner3'] 
        self.actions = {'pick': ['corner0', 'corner3'], 'place': ['corner1', 'corner2']}
        
        # trajectory params, in the future we might optimize them
        self.height = [0.1]*self.num_pick_places
        self.tilt = [0]*self.num_pick_places
        self.vel = [2.0]*self.num_pick_places
        self.traj_dt = [0.01]*self.num_pick_places
 
def pick_and_place(env, actions, args):
    for a in actions:
        obs, _, _, _ = env.step(np.asarray(a), record_continuous_video=False, img_size=args.img_size)
    env.action_tool.release_particle()
    wait_until_scene_is_stable(pyflex_stepper=env.cloth_system.pyflex_stepper, max_steps=500, tolerance=0.05)
    obs = env._get_obs()
    return obs

# def process_graph(pos, edge_index, get_point_distances=False):
#     # Create the Laplacian matrix
#     L = create_laplacian(torch.from_numpy(edge_index), pos.shape[0]).numpy()
    
#     if get_point_distances:
#         # this is also a feature that can be used to compute similarities if we only use one keypoint
#         d_matrix = compute_distance_matrix(pos, edge_index)
#         return L, d_matrix
#     else:
#         return L, _
    

               
def process_obs(obs, demo_data, args):
    rgb, points = obs['camera_0']['rgbd'][:, :, :3], obs["pos"]
    
    if demo_data['graph_ids'] is None:
        # todo: get graph out of the points, this will be the tracked graph in the full pipeline
        if args.subsample:
            sampled_points_indeces = farthest_point_sampling(points, args.num_samples)
        else:
            sampled_points_indeces = np.arange(points.shape[0])

        graph_pos = points[sampled_points_indeces]
        _, faces = compute_edges_index(graph_pos, k=args.knn, delaunay=args.delaunay, sim_data=True, norm_threshold=0.1)
        # get edges with torch geometric
        graph_data = Data(pos=torch.tensor(graph_pos, dtype=torch.float32), face=torch.tensor(faces, dtype=torch.long))
        graph_data = T.FaceToEdge(remove_faces=False)(graph_data)
        edge_index = graph_data.edge_index
        
        demo_data['graph_ids'] = sampled_points_indeces
        demo_data['edge_index'] = edge_index
        demo_data['faces'] = faces
        demo_data['graph_keypoints_ids'] = np.array([np.argmin(np.linalg.norm(graph_pos - points[keypoint], axis=1)) for keypoint in demo_data['keypoints_ids']]) 
        demo_data['laplacian'] = create_laplacian(edge_index, graph_pos.shape[0]).numpy()
    else:
        graph_pos = points[demo_data['graph_ids']]
    return rgb, points, graph_pos, demo_data

def update_data(demo_data, rgb, graph_pos, points, env):
    demo_data['images'].append(rgb)
    demo_data['pos'].append(points)
    demo_data['graph'].append(graph_pos)
    coverage = env.compute_coverage()
    demo_data['coverage'].append(coverage)
    return demo_data

def load_demo(path):
    with h5py.File(path, 'r') as f:
        data = {key: np.array(f[key]) for key in f.keys()}
    return data
    
def imitate_demo(args):
    mesh_paths = get_meshes_paths(args.object, args.flat_mesh_dataset)
    demo = load_demo(f'{args.dataset_path}/{args.dataset_name}/{args.object}/{args.demo}/data.h5')  
    demo['laplacian'] = create_laplacian(torch.from_numpy(demo['edge_index']), demo['graph'][0].shape[0]).numpy()
    
    # cloth parameters
    deformation_config = DeformationConfig()
    static_friction, dynamic_friction, particle_friction, drag, stretch_stiffness, bend_stiffness = sample_cloth_params(deformation_config)

    mesh_id = args.mesh_id
    # check if you are using the same cloth or not
    if mesh_id == demo['mesh_id']:
        print('Using the same cloth to imitate')
         
    target_mesh_path = mesh_paths[mesh_id]
    
    env_kwargs = create_pyflex_cloth_scene_config(
            cloth_type=args.object,
            env_id=0,
            traj_id=0,
            mesh_id=mesh_id,
            target_mesh_path=target_mesh_path,     
            action_mode = args.action_mode, # ['line', 'square', 'circular']
            render= True,
            headless = args.headless,
            recording=False,
            deformation_config  = deformation_config,
            dynamic_friction = dynamic_friction, 
            particle_friction = particle_friction,
            static_friction= static_friction, 
            stretch_stiffness=stretch_stiffness,
            bend_stiffness=bend_stiffness,
            drag = drag,
            particle_radius = 0.01,  # keep radius close to particle rest distances in mesh to avoid weird behaviors
            solid_rest_distance = 0.01, # mesh triangulation -> approx 1cm edges lengths
            )
    
    # get demo configurations, this is needed for the hyperparameters of the trajectory
    if args.demo == 'fold_half':
        demo_config = HalfFoldConfig()
                  
    data_save_path = f'{args.dataset_path}/{args.dataset_name}/{args.object}/{args.demo}_imitation/'
    os.makedirs(data_save_path, exist_ok=True)
    print(f"Saving data to {data_save_path}")
    
    env = ClothEnv(num_steps_per_action=1, **env_kwargs)
    env.reset()
    
    imitation_data = {'images':[], 'pick':[], 'place':[], 'graph': [],'graph_ids': None, 'edge_index': None, 'faces': None, 'laplacian': None,
                      'pos':[], 'trajectory_params':[], 'keypoints_ids':[], 'coverage':[]}
    imitation_data['keypoints_ids'] = list(env.keypoints.values())

    for i in range(demo_config.num_pick_places):
        obs = env._get_obs()
        print(list(obs.keys()))
        rgb, points, graph_pos, imitation_data = process_obs(obs, imitation_data, args)   # if first call, this initialize the graph, edges, faces and laplacian
        imitation_data = update_data(imitation_data, rgb, graph_pos, points, env)
        
        plt.imshow(obs['top_view']['rgbd'][:, :, :3]/255.)
        plt.savefig(f'./test_top_view_{i}.png')
        
        # debug polts        
        # plot_mesh(graph_pos, demo_data['edge_index'].T, center_plot=None, white_bkg=False, save_fig=False, file_name='mesh.png')
        # plt.imshow(rgb/255.)
        # plt.show()
        
        ################################## Example of pixel to image and inverse ################
        # TO DEBUG, didnt have time to check
        depth = obs['top_view']['rgbd'][:, :, 3]
        pixel = (depth.shape[0]//2, depth.shape[1]//2)
        position = env.pixel_to_3d(pixel, depth, camera_name='top_view')        # leave this camera name for now
        
        pixel_back = env.project_to_image(position, camera_name='top_view')
        
        # # plot the image with the pixel_back in red as well
        plt.imshow(obs['top_view']['rgbd'][:, :, :3]/255.)
        plt.scatter(pixel_back[0], pixel_back[1], c='r')
        plt.show()
        
        position_keypoint = deepcopy(imitation_data['graph'][0][imitation_data['graph_keypoints_ids'][0], :])
        # m to mm
        # position_keypoint /= 1000
        pixel_keypoint = env.project_to_image(deepcopy(position_keypoint), camera_name='top_view')
        plt.imshow(obs['top_view']['rgbd'][:, :, :3]/255.)
        plt.scatter(pixel_keypoint[0], pixel_keypoint[1], c='r')
        plt.show()
        
        ################################# pick and place selection ################################
        
        if args.policy == 'random':
            pick_idx = env.get_random_pick() # returns an idx not a position, so we need to get the 3D position
            pick = env.cloth_system.get_positions()[pick_idx]
            place =  pick + env.get_random_place(pick_idx)           

        elif args.policy == 'demo':
            pick_key = demo_config.actions['pick'][i]       # this is the semantic name of the keypoint
            pick = env.cloth_system.get_positions()[env.keypoints[pick_key]]
            
            place_key = demo_config.actions['place'][i]
            place = env.cloth_system.get_positions()[env.keypoints[place_key]]  
            
        elif args.policy == 'semantic_graph_imitation':
            # TODO: extract graph features for both graphs
            if args.keypoints == 'gt_keypoints':
                demo_keypoints_id = demo['graph_keypoints_ids']
                graph_keypoints_id = imitation_data['graph_keypoints_ids']
                
            # https://github.com/ShirAmir/dino-vit-features
                
            if args.features == 'rgb':
                num_keypoints = len(demo_keypoints_id)
                # keypoint features as gist_rainbow colormap
                keypoints_features = np.asarray(list(plt.cm.gist_rainbow(np.linspace(0, 1, num_keypoints))))
                
                # can be extended to semantic features from DINO, which requires PCA 
                
            # diffuse the features of the keypoints to the other nodes
            demo_all_diffused_colors, demo_color_changes = features_diffusion(demo['graph'][i], demo_keypoints_id, keypoints_features, torch.from_numpy(demo['laplacian']), steps=1000, epsilon=0.1)
            demo_diffused_colors = demo_all_diffused_colors[-1]
            
            all_diffused_colors, color_changes = features_diffusion(imitation_data['graph'][i], graph_keypoints_id, keypoints_features, torch.from_numpy(imitation_data['laplacian']), steps=1000, epsilon=0.1)
            diffused_colors = all_diffused_colors[-1]
            
            # debugging
            num_samples_per_triangle = 0 # this samples points on the mesh faces to densify the mesh
            # img = plot_mesh_with_colors(demo['graph'][i], demo['edge_index'].T, colors=demo_diffused_colors, faces=demo['faces'], reference_vertex=0, points=np.asarray([demo['pick'][i]]), num_samples_per_triangle=num_samples_per_triangle, keypoints_idx=None,
            #                             elev=55, azim=-120, center_plot=None, return_image=False, white_bkg=True, save_fig=False, file_name='mesh.png')
            
            # img = plot_mesh_with_colors(imitation_data['graph'][i], imitation_data['edge_index'].T, colors=diffused_colors, faces=imitation_data['faces'], reference_vertex=0, points=np.asarray([demo['pick'][i]]), num_samples_per_triangle=num_samples_per_triangle, keypoints_idx=None,
            #                 elev=55, azim=-120, center_plot=None, return_image=False, white_bkg=True, save_fig=False, file_name='mesh.png')
            
            # use a similarity metric w.r.t. demoed pick and place positions as value map
            # first find the closest point to the pick and place positions in the demo mesh            
            demo_pick_id = np.argmin(np.linalg.norm(demo['graph'][i] - demo['pick'][i], axis=1))
            demo_place_id = np.argmin(np.linalg.norm(demo['graph'][i] - demo['place'][i], axis=1))
            
            pick_feature = demo_diffused_colors[demo_pick_id]
            place_feature = demo_diffused_colors[demo_place_id]
            
            # now for each point in the imitation graph compute the distance to the pick and place features and normalize them
            
            pick_distance_metric = np.linalg.norm(diffused_colors - pick_feature, axis=1)
            place_distance_metric = np.linalg.norm(diffused_colors - place_feature, axis=1)
            
            # normalize to 0 and 1
            pick_distance_metric = (pick_distance_metric - np.min(pick_distance_metric))/(np.max(pick_distance_metric) - np.min(pick_distance_metric))
            place_distance_metric = (place_distance_metric - np.min(place_distance_metric))/(np.max(place_distance_metric) - np.min(place_distance_metric))
            
            # now we can use the distance in the feature space to pick the closest point to the pick and place positions as a value function
            pick_value = 1 - pick_distance_metric
            place_value = 1 - place_distance_metric     
            
            # plot the value map for debugging purposes 
            # img = plot_mesh_with_colors(imitation_data['graph'][i], imitation_data['edge_index'].T, colors=pick_value, faces=imitation_data['faces'], reference_vertex=0, points=np.asarray([demo['pick'][i]]), 
            #                       num_samples_per_triangle=num_samples_per_triangle, keypoints_idx=None,save_fig=False, file_name='mesh.png')

            # sample pick and place based on the value map       
            pick_idx = np.argmax(pick_value)
            place_idx = np.argmax(place_value)
            
            pick = imitation_data['graph'][i][pick_idx]
            place = imitation_data['graph'][i][place_idx]
        
        # store to use them as reference in the future
        imitation_data['pick'].append(pick)
        imitation_data['place'].append(place)        
        
        
        ########################## execution ########################################
        
        # predefined trajectory hyperparameters, in the future we might optimize them
        height, tilt, velocity, dt = demo_config.height[i], demo_config.tilt[i], demo_config.vel[i], demo_config.traj_dt[i]        
        imitation_data['trajectory_params'].append([height, tilt, velocity, dt])
        
        
        # based on the 3D position of the pick, find the closest particle to pick
        pick_idx = np.argmin(np.linalg.norm(points - pick, axis=1))        
        env.action_tool.grasp_particle(pick_idx)
        
        # generate trajectory and actions
        trajectory, actions  = get_action_traj(pick, place, height, tilt, velocity, dt, sim_data=True)
        obs = pick_and_place(env, actions, args)

    rgb, points, graph_pos, imitation_data = process_obs(obs, imitation_data, args)
    imitation_data = update_data(imitation_data, rgb, graph_pos, points, env)

    # TODO: compute some statistics, normalized coverage
        
    store_data_by_name(data_names=list(imitation_data.keys()), data=imitation_data, path=f'{data_save_path}data.h5')
    print(f'Saved demo in {data_save_path}data.h5')
    

      


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--save_data', type=int, default=1, help='Whether to save the data')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=360, help='Size of the recorded videos')
    parser.add_argument('--object', type=str, default='TOWEL', help='Object to load, choices are TOWEL, TSHIRT, SHORTS')
    parser.add_argument('--flat_mesh_dataset', type=str, default='0411_train', help='Dataset of meshes [dev, 00-final, 0411_test, 0411_train]')
    
    # parser.add_argument('-start_mesh_id', type=int, default=0, help='Id of the mesh we want to start generating from')
    # parser.add_argument('--num_cloths', type=int, default=100, help='Number different cloth in the dataset')
    # parser.add_argument('--num_trajectories', type=int, default=50, help='Number of trajectories to generate per cloth')
    # parser.add_argument('--dataset_name', type=str, default='train_dataset_0415', help='Name of the dataset')
    
    parser.add_argument('--demo', type=str, default='fold_half', help='Name of the task to execute')
    parser.add_argument('--policy', type=str, default='semantic_graph_imitation', help='Type of policy to use, choices are [random, demo, semantic_graph_imitation, feature__matching_DINO(implement)]') # , feature_matching---- bc
    parser.add_argument('--keypoints', type=str, default='gt_keypoints', help='Type of semantic to use, choices are [gt_keypoints, --- DINO-BB(implement)]') # pred_keypoints, 
    parser.add_argument('--features', type=str, default='rgb', help='Type of semantic to use, choices are [rgb, DINO(implement)]')  # distances?
    parser.add_argument('--densify', type=int, default=0, help='whether to densify the mesh or not to compute the value map.')

    parser.add_argument('-mesh_id', type=int, default=10, help='Id of the mesh we want to use to generating from')
    parser.add_argument('--num_cloths', type=int, default=1, help='Number different cloth in the dataset')
    parser.add_argument('--num_trajectories', type=int, default=1, help='Number of trajectories to generate per cloth')
    parser.add_argument('--dataset_name', type=str, default='0508_test', help='Name of the dataset')
    
    parser.add_argument('--dataset_path', type=str, default='./manipulation/demos', help='Name of the dataset')
    parser.add_argument('--action_mode', type=str, default='circular', help='how to sample the trajectory, still need to implement variations')
    
    # Data Processing
    parser.add_argument('--knn', type=int, default=10, help='Number of neighbor to construct the graph.')
    parser.add_argument('--delaunay', type=int, default=1, help='Whether to use delaunay to traingulation or not.')
    parser.add_argument('--subsample', type=int, default=1, help='Whether to subsample or not the initial set of points.')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of points to subsample. Default 300')
    
    args = parser.parse_args()
    

  
    imitate_demo(args)
    
 