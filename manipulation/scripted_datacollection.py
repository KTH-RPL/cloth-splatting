import numpy as np
from gym.spaces import Box
import pyflex
from manipulation.envs.gym_env import GymEnv
from manipulation.envs.cloth_env import ClothEnv
from copy import deepcopy
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
from multiprocessing import Process
from manipulation.utils.data_collection import get_meshes_paths, store_data_by_name, sample_cloth_params
from tqdm import tqdm

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

def collect_trajectory(args, 
                       pick_keypoint,
                       place_keypoint,
                       height,
                        tilt,
                        velocity,
                        dt,
                       **kwargs):
    
    data_dict = {
        "init_rgbd": [],        # multiple viewpoints of the initial scene
        #"rgbd": [],             # rgbds from few cameras at each time step
        "pos": [],              # particle positions at each time step
        "vel": [],              # particle velocities at each time step
        "grasp": [],            # whether the particle is grasped at each time step
        "gripper_pos": [],      # position of the gripper at each time step
        "done": [],
        "grasped_particle": [],
        "pick": [],
        "place": [],
        "keypoints_ids": [],
        "trajectory_params": [],
        "trajectory": [],
        "actions": [],
        "cloth_params": [],
        
    }
    
    mesh_idx = kwargs["mesh_id"]
    traj_idx = kwargs["traj_id"]
    data_dict["grasped_particle"] = list(env.keypoints.values())[pick_keypoint]
    all_params = ['dynamic_friction', 'particle_friction', 'static_friction', 'stretch_stiffness', 'bend_stiffness', 'drag']
    data_dict["cloth_params"] = [kwargs["scene_config"][param] for param in all_params]      # TODO: transform it into array
    
    if args.save_data:
        idx = "{:05}".format(mesh_idx)
        idx_traj = "{:05}".format(traj_idx)
        # TODO: change naming
        data_save_path = f'{args.dataset_path}/{args.dataset_name}/{kwargs["cloth_type"]}/{idx}/{idx_traj}/'
        os.makedirs(data_save_path, exist_ok=True)
        print(f"Saving data to {data_save_path}")
        
    env = ClothEnv(num_steps_per_action=1, **kwargs)
    env.reset()
    
    data_dict['keypoints_ids'] = list(env.keypoints.values())
    
    pick_idx = env.get_keypoint_pick(keypoint_idx=pick_keypoint)
    pick = env.cloth_system.get_positions()[pick_idx]
    place =  pick + env.get_keypoint_place(pick_idx, keypoint_idx=place_keypoint)    
    
    # create and save camera params (intrinsics, extrinsics)
    env.save_cameras(output_dir_cam=data_save_path)
    for cam in env.camera_names:
        data_dict[f'{cam}_rgbd'] = []    

    
    data_dict['pick'] = pick
    data_dict['place'] = place
    data_dict['trajectory_params'] = [height, tilt, velocity, dt]
    
    # generate trajectory and actions
    trajectory, actions  = get_action_traj(pick, place, height, tilt, velocity, dt, sim_data=True)
    visualize_trajectory_and_actions(trajectory[:, [0,2,1]], actions[:, [0,2,1]])
    
    data_dict['trajectory'] = trajectory
    # data_dict['actions'] = actions
    
    ############## Execute the fold
    #TODO: init rgbds from multiple camera views

    env.action_tool.grasp_particle(pick_idx)
    obs = env._get_obs()
    data_dict = update_observations(data_dict, obs, action=np.zeros(3), grasp=1, camera_names=env.camera_names)
    positions = [env.cloth_system.get_positions()[pick_idx]]
    
    for i in range(actions.shape[0]):
        a = actions[i]
        obs, _, _, _ = env.step(a, record_continuous_video=False, img_size=args.img_size)
        data_dict = update_observations(data_dict, obs, action=a, grasp=1, camera_names=env.camera_names)
        positions.append(env.cloth_system.get_positions()[pick_idx])
        
        # print('******************************************************************')
        # print(f'Current position: {positions[i]}, wanted position: {trajectory[i]}')
        # print(f'Current action: {positions[i] - positions[i-1]}, wanted action: {a}')
        # print(f'Error from desired: {trajectory[i] - positions[i]}')
        
    env.action_tool.release_particle()
    wait_until_scene_is_stable(pyflex_stepper=env.cloth_system.pyflex_stepper, max_steps=500, tolerance=0.05)
    obs = env._get_obs()
    obs['done'] = True
    data_dict = update_observations(data_dict, obs, action=np.zeros(3), grasp=0, camera_names=env.camera_names)
    
    if args.save_data:
        store_data_by_name(data_names=list(data_dict.keys()), data=data_dict, path=f'{data_save_path}data.h5')
        
    print()
    
    
def full_data_collection(args):
    mesh_paths = get_meshes_paths(args.object, args.flat_mesh_dataset)
    num_cloths = args.num_cloths
    num_trajectories = args.num_trajectories
    
    deformation_config = DeformationConfig()
    
    for c in range(num_cloths):
        mesh_id = c
        target_mesh_path =mesh_paths[mesh_id]
        
        with tqdm(total=num_trajectories, desc=f'Cloth {c+1}/{num_cloths} ({(c/num_cloths)*100:.2f}%)') as pbar:
            for t in tqdm(range(num_trajectories)):
                pbar.update()
                # do we want to randomize all the times or only once for each cloth?
                static_friction, dynamic_friction, particle_friction, drag, stretch_stiffness, bend_stiffness = sample_cloth_params(deformation_config)
            
                env_kwargs = create_pyflex_cloth_scene_config(
                            cloth_type=args.object,
                            env_id=0,
                            traj_id=t,
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
                
                collect_trajectory(args, **env_kwargs)
    
        
# def get_pick_place(**env_kwargs):
#     env = ClothEnv(num_steps_per_action=1, **env_kwargs)
#     env.reset()
#     pick_idx = env.get_random_pick()
#     pick = env.cloth_system.get_positions()[pick_idx]
#     place =  pick + env.get_random_place(pick_idx)    
    
#     return pick_idx, pick, place

def worker(args,     
        pick_keypoint,
        place_keypoint,
        height,
        tilt,
        velocity,
        dt,
        env_kwargs):
    
    collect_trajectory(args,     
        pick_keypoint,
        place_keypoint,
        height,
        tilt,
        velocity,
        dt,
        **env_kwargs)
    
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--save_data', type=int, default=1, help='Whether to save the data')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--object', type=str, default='TOWEL', help='Object to load, choices are TOWEL, TSHIRT, SHORTS')
    parser.add_argument('--flat_mesh_dataset', type=str, default='0411_train', help='Dataset of meshes [dev, 00-final, 0411_test, 0411_train]')

    parser.add_argument('--num_cloths', type=int, default=1, help='Number different cloth in the dataset')
    parser.add_argument('--num_trajectories', type=int, default=5, help='Number of trajectories to generate per cloth')
    parser.add_argument('--dataset_path', type=str, default='./sim_datasets', help='Name of the dataset')
    parser.add_argument('--dataset_name', type=str, default='0415_debug', help='Name of the dataset')
    parser.add_argument('--action_mode', type=str, default='circular', help='how to sample the trajectory, still need to implement variations')
    
    args = parser.parse_args()
    
    # full_data_collection(args)
    
    ########### cloth params generation ##############
    mesh_paths = get_meshes_paths(args.object, args.flat_mesh_dataset)
    
    
    deformation_config = DeformationConfig()
    deformation_config.grasp_keypoint_vertex_probability = 1
    deformation_config.place_keypoint_vertex_probability = 1
    
    static_friction = np.random.uniform(0.3, 1.0)
    dynamic_friction = np.random.uniform(0.3, 1.0)
    particle_friction = np.random.uniform(0.3, 1.0)
    
    # drag is important to create some high frequency wrinkles
    drag = np.random.uniform(deformation_config.max_drag / 5, deformation_config.max_drag)    
    
    stretch_stiffness = np.random.uniform(0.5, deformation_config.max_stretch_stiffness)
    bend_stiffness = np.random.uniform(0.01, deformation_config.max_bending_stiffness)
    
    

    pick_keypoint = 0
    place_keypoint = 5
    
    if args.object == 'TOWEL':
        place_keypoint = 2
    
    #################################
    for i in range(args.num_cloths):    
        # select mesh at random
        mesh_id = i
        target_mesh_path =mesh_paths[mesh_id]

        env_kwargs = create_pyflex_cloth_scene_config(
        cloth_type=args.object,
        env_id=0,
        traj_id=0,
        mesh_id=mesh_id,
        target_mesh_path=target_mesh_path,      # TODO: set
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

        for traj in range(args.num_trajectories):    
            height = random.uniform(deformation_config.min_fold_height, deformation_config.max_fold_height) # random betweem 0.05 and 0.15
            tilt = np.radians(random.randint(deformation_config.min_fold_tilt, deformation_config.max_fold_tilt)) # random between -45 and 45 degrees
            velocity = deformation_config.fixed_fold_vel
            dt = deformation_config.traj_dt  
            
            env_kwargs['traj_id'] = traj
            
            p = Process(target=worker, args=(args, pick_keypoint, place_keypoint, height, tilt, velocity, dt, env_kwargs))
            p.start()
            p.join()
            
            # collect_trajectory(args,     
            #                 pick_keypoint,
            #                 place_keypoint,
            #                 height,
            #                 tilt,
            #                 velocity,
            #                 dt,
            #                 **env_kwargs)
    
    