import numpy as np
from gym.spaces import Box
import pyflex
from manipulation.envs.gym_env import GymEnv
from manipulation.envs.cloth_env import ClothEnv
from torch_geometric.data import Data
import torch
import torch_geometric.transforms as T
from copy import deepcopy
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

# transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])

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
        
    def compute_pp(self, cloth_points, keypoints, i):
        pick_key = self.actions['pick'][i]       # this is the semantic name of the keypoint
        pick = cloth_points[keypoints[pick_key]]
        
        place_key = self.actions['place'][i]
        place = cloth_points[keypoints[place_key]]   
        
        return pick_key, pick, place_key, place
    
class ShortsFoldConfig:
    def __init__(self, ):
        self.num_pick_places = 2
        self.keypoints = ['waist_left', 'waist_right', 'pipe_right_outer', 'pipe_right_inner', 'crotch', 'pipe_left_inner', 'pipe_left_outer']
        self.actions = {'pick': ['mid_pipe_left', 'mid_pipe_right'], 'place': ['corner1', 'corner2']}
        
        # trajectory params, in the future we might optimize them
        self.height = [0.1]*self.num_pick_places
        self.tilt = [0]*self.num_pick_places
        self.vel = [2.0]*self.num_pick_places
        self.traj_dt = [0.01]*self.num_pick_places
        
    def compute_pp(self, cloth_points, keypoints, i):
        pick_key = self.actions['pick'][i]       # this is the semantic name of the keypoint
        pick = cloth_points[keypoints[pick_key]]
        
        place_key = self.actions['place'][i]
        place = cloth_points[keypoints[place_key]]   
        
        return pick_key, pick, place_key, place
        
        
 
def pick_and_place(env, actions, args):
    for a in actions:
        obs, _, _, _ = env.step(np.asarray(a), record_continuous_video=False, img_size=args.img_size)
    env.action_tool.release_particle()
    wait_until_scene_is_stable(pyflex_stepper=env.cloth_system.pyflex_stepper, max_steps=500, tolerance=0.05)
    obs = env._get_obs()
    return obs
               
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
        edge_index = graph_data.edge_index.numpy()
        
        demo_data['graph_ids'] = sampled_points_indeces
        demo_data['edge_index'] = edge_index
        demo_data['faces'] = faces
        # find keypoints on graph
        demo_data['graph_keypoints_ids'] = np.array([np.argmin(np.linalg.norm(graph_pos - points[keypoint], axis=1)) for keypoint in demo_data['keypoints_ids']]) 
        
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
    
def demo_data_collection(args):
    mesh_paths = get_meshes_paths(args.object, args.flat_mesh_dataset)
    
    
    # cloth parameters
    deformation_config = DeformationConfig()
    static_friction, dynamic_friction, particle_friction, drag, stretch_stiffness, bend_stiffness = sample_cloth_params(deformation_config)

    mesh_id = args.mesh_id
    target_mesh_path =mesh_paths[mesh_id]
    
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
    
    # set demo details and configurations
    if args.demo == 'fold_half':
        demo_config = HalfFoldConfig()
    if args.demo == 'fold_shorts':
        demo_config = ShortsFoldConfig()
        
          
    # saving path
    data_save_path = f'{args.dataset_path}/{args.dataset_name}/{args.object}/{args.demo}/'
    os.makedirs(data_save_path, exist_ok=True)
    print(f"Saving data to {data_save_path}")
    
    env = ClothEnv(num_steps_per_action=1, **env_kwargs)
    env.reset()
    
    print(list(env.keypoints.keys()))
    
    demo_data = {'images':[], 'mesh_id': None, 'pick':[], 'place':[], 'graph': [],'graph_ids': None, 
                 'edge_index': None, 'faces': None, 'pos':[], 'trajectory_params':[], 'keypoints_ids':[], 
                 'graph_keypoints_ids':[], 'coverage':[]}
    
    # TODO: put explanation of namings
    
    demo_data['keypoints_ids'] = list(env.keypoints.values())
    demo_data['mesh_id'] = mesh_id

    for i in range(demo_config.num_pick_places):
        obs = env._get_obs()
        rgb, points, graph_pos, demo_data = process_obs(obs, demo_data, args)   # if first call, this initialize the graph, edges and faces
        demo_data = update_data(demo_data, rgb, graph_pos, points, env)
        
        # debug polts
        
        # plot_mesh(graph_pos, demo_data['edge_index'].T, center_plot=None, white_bkg=False, save_fig=False, file_name='mesh.png')
        # plt.imshow(rgb/255.)
        # plt.show()
        
        ################################# pick and place selection ################################
        # random policy
        # pick_idx = env.get_random_pick() # returns an idx not a position, so we need to get the 3D position
        # pick = env.cloth_system.get_positions()[pick_idx]
        # place =  pick + env.get_random_place(pick_idx)    
        
        # demo policy
        # pick_key = demo_config.actions['pick'][i]       # this is the semantic name of the keypoint
        # pick = env.cloth_system.get_positions()[env.keypoints[pick_key]]
        
        # place_key = demo_config.actions['place'][i]
        # place = env.cloth_system.get_positions()[env.keypoints[place_key]]   
        
        pick_key, pick, place_key, place = demo_config.compute_pp(env.cloth_system.get_positions(), env.keypoints, i)
        
        # store to use them as reference in the future
        demo_data['pick'].append(pick)
        demo_data['place'].append(place)        
        
        
        ########################## execution ########################################
        
        # predefined trajectory hyperparameters, in the future we might optimize them
        height, tilt, velocity, dt = demo_config.height[i], demo_config.tilt[i], demo_config.vel[i], demo_config.traj_dt[i]        
        demo_data['trajectory_params'].append([height, tilt, velocity, dt])
        
        
        # based on the 3D position of the pick, find the closest particle to pick
        pick_idx = np.argmin(np.linalg.norm(points - pick, axis=1))        
        env.action_tool.grasp_particle(pick_idx)
        
        # generate trajectory and actions
        trajectory, actions  = get_action_traj(pick, place, height, tilt, velocity, dt, sim_data=True)
        obs = pick_and_place(env, actions, args)

    rgb, points, graph_pos, demo_data = process_obs(obs, demo_data, args)
    demo_data = update_data(demo_data, rgb, graph_pos, points, env)

        
    store_data_by_name(data_names=list(demo_data.keys()), data=demo_data, path=f'{data_save_path}data.h5')
    print(f'Saved demo in {data_save_path}data.h5')
    

      


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--headless', type=int, default=1, help='Whether to run the environment with headless rendering')
    parser.add_argument('--save_data', type=int, default=1, help='Whether to save the data')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=360, help='Size of the recorded videos')
    parser.add_argument('--object', type=str, default='SHORTS', help='Object to load, choices are TOWEL, TSHIRT, SHORTS')
    parser.add_argument('--flat_mesh_dataset', type=str, default='0411_train', help='Dataset of meshes [dev, 00-final, 0411_test, 0411_train]')
    
    # parser.add_argument('-start_mesh_id', type=int, default=0, help='Id of the mesh we want to start generating from')
    # parser.add_argument('--num_cloths', type=int, default=100, help='Number different cloth in the dataset')
    # parser.add_argument('--num_trajectories', type=int, default=50, help='Number of trajectories to generate per cloth')
    # parser.add_argument('--dataset_name', type=str, default='train_dataset_0415', help='Name of the dataset')
    
    parser.add_argument('--demo', type=str, default='fold_shorts', help='Name of the dataset. Options: [fold_half, fold_shorts]')
    parser.add_argument('-mesh_id', type=int, default=0, help='Id of the mesh we want to use to generating from')
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
    

  
    demo_data_collection(args)
    
 