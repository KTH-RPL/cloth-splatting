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
from meshnet.viz import plot_pcd_list
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
    get_action_traj,
    Action_Sampler,
    visualize_sampled_traj
)
from manipulation.utils.data_collection import get_meshes_paths, store_data_by_name, sample_cloth_params
from tqdm import tqdm
from multiprocessing import Process

def update_observations(data_dict, obs, action, grasp, camera_names):
    # for cam in camera_names:
    #     data_dict[f'{cam}_rgbd'].append(obs[cam]['rgbd'])
    data_dict["pos"].append(obs["pos"])
    data_dict["vel"].append(obs["vel"])
    data_dict["gripper_pos"].append(obs["gripper_pos"])
    data_dict["done"].append(obs["done"])
    data_dict['actions'].append(action)
    data_dict['grasp'].append(grasp)
    
    return data_dict

def collect_trajectory(args, **kwargs):
    
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
        "trajectory_params": [],
        "trajectory": [],
        "actions": [],
        "cloth_params": [],
        "keypoints_ids": [],    # ids of the keypoints in the mesh
        
    }
    
    mesh_idx = kwargs["mesh_id"]
    traj_idx = kwargs["traj_id"]
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
    
    # create and save camera params (intrinsics, extrinsics)
    env.save_cameras(output_dir_cam=data_save_path)
    for cam in env.camera_names:
        data_dict[f'{cam}_rgbd'] = []
    
    # sample random trajectory params
    deformation_config = kwargs["deformation_config"]
    pick_idx = env.get_random_pick()
    pick = env.cloth_system.get_positions()[pick_idx]
    place =  pick + env.get_random_place(pick_idx)    
    height = random.uniform(deformation_config.min_fold_height, deformation_config.max_fold_height) # random betweem 0.05 and 0.15
    tilt = np.radians(random.randint(deformation_config.min_fold_tilt, deformation_config.max_fold_tilt)) # random between -45 and 45 degrees
    velocity = deformation_config.fixed_fold_vel
    dt = deformation_config.traj_dt  
    
    data_dict['pick'] = pick
    data_dict['place'] = place
    data_dict['trajectory_params'] = [height, tilt, velocity, dt]
    
    # generate SMOOTH trajectory and actions
    # with probability 0.1 generate smooth actions
    smoot_actions = random.random() < 0.1
    if smoot_actions:
        trajectory, actions  = get_action_traj(pick, place, height, tilt, velocity, dt, sim_data=True)
    else:
        # Generate random trajectory
        traj_len = random.randint(15, 25)
        velocity = random.uniform(0.05, 0.1)
        
        # approximate the velocity up to the second decimal
        velocity = round(velocity, 2)
        
        action_repetition = random.randint(1, 3)
        trajectory_sampler = Action_Sampler(
                    N=traj_len,  # trajectory length
                    velocity=velocity,
                    c_threshold=0.,
                    noise_sigma=0.01,
                    action_repetition=action_repetition,
                    pp_dir=place - pick,
                    place=place,
                    starting_point=pick,
                    grid_size=0.01,
                    sampling_mean=None,
                    fixed_trajectory=None,
                    invert_yz=True)
        trajectory, actions = trajectory_sampler.sample_trajectory(starting_point=pick, 
                                                        target_point=place, 
                                                        return_actions=True)
    
    
    data_dict['trajectory'] = trajectory
    
    # data_dict['actions'] = actions
    
    ############## Execute the fold
    #TODO: init rgbds from multiple camera views

    env.action_tool.grasp_particle(pick_idx)
    obs = env._get_obs()
    data_dict = update_observations(data_dict, obs, action=np.zeros(3), grasp=1, camera_names=env.camera_names)
    
    for a in actions:
        a = np.asarray(a)
        obs, _, _, _ = env.step(a, record_continuous_video=False, img_size=args.img_size)
        data_dict = update_observations(data_dict, obs, action=a, grasp=1, camera_names=env.camera_names)
        
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
        mesh_id = c + args.start_mesh_id
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
                
                p = Process(target=worker, args=(args, env_kwargs))
                p.start()
                p.join()
                
                # collect_trajectory(args, **env_kwargs)
    
def worker(args, env_kwargs):
    
    collect_trajectory(args,  **env_kwargs)
      


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--headless', type=int, default=1, help='Whether to run the environment with headless rendering')
    parser.add_argument('--save_data', type=int, default=1, help='Whether to save the data')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    
    # TODO: remember that for now images are not saved!
    parser.add_argument('--img_size', type=int, default=360, help='Size of the recorded videos')
    parser.add_argument('--object', type=str, default='TOWEL', help='Object to load, choices are TOWEL, TSHIRT, SHORTS')
    parser.add_argument('--flat_mesh_dataset', type=str, default='0411_train', help='Dataset of meshes [dev, 00-final, 0411_test, 0411_train]')
    
    # parser.add_argument('-start_mesh_id', type=int, default=0, help='Id of the mesh we want to start generating from')
    # parser.add_argument('--num_cloths', type=int, default=100, help='Number different cloth in the dataset')
    # parser.add_argument('--num_trajectories', type=int, default=50, help='Number of trajectories to generate per cloth')
    # parser.add_argument('--dataset_name', type=str, default='train_dataset_0415', help='Name of the dataset')
    
    parser.add_argument('-start_mesh_id', type=int, default=0, help='Id of the mesh we want to start generating from')
    parser.add_argument('--num_cloths', type=int, default=1, help='Number different cloth in the dataset')
    # parser.add_argument('--dataset_name', type=str, default='test_dataset_0415', help='Name of the dataset')
    
    # parser.add_argument('--num_trajectories', type=int, default=1000, help='Number of trajectories to generate per cloth')
    # parser.add_argument('--dataset_name', type=str, default='train_dataset_0702', help='Name of the dataset')
    
    parser.add_argument('--num_trajectories', type=int, default=100, help='Number of trajectories to generate per cloth')
    parser.add_argument('--dataset_name', type=str, default='test_dataset_0702', help='Name of the dataset')
    
    parser.add_argument('--dataset_path', type=str, default='./sim_datasets', help='Name of the dataset')
    parser.add_argument('--action_mode', type=str, default='circular', help='how to sample the trajectory, still need to implement variations')
    
    args = parser.parse_args()
    
    full_data_collection(args)
    
    ############## ALL OBJETCT COLLECTION: Needed for test ##############
    
    # for cloth in ['TOWEL', 'TSHIRT', 'SHORTS']:
    #     args.object = cloth    
    #     full_data_collection(args)
    
    ########### cloth params generation testing ##############
#     mesh_paths = get_meshes_paths(args.object, args.flat_mesh_dataset)
    
#     # select mesh at random
#     mesh_id = np.random.randint(len(mesh_paths))
#     target_mesh_path =mesh_paths[mesh_id]
    
#     deformation_config = DeformationConfig()
    
#     static_friction = np.random.uniform(0.3, 1.0)
#     dynamic_friction = np.random.uniform(0.3, 1.0)
#     particle_friction = np.random.uniform(0.3, 1.0)
    
#     # drag is important to create some high frequency wrinkles
#     drag = np.random.uniform(deformation_config.max_drag / 5, deformation_config.max_drag)    
    
#     stretch_stiffness = np.random.uniform(0.5, deformation_config.max_stretch_stiffness)
#     bend_stiffness = np.random.uniform(0.01, deformation_config.max_bending_stiffness)
    
#     #################################

#     env_kwargs = create_pyflex_cloth_scene_config(
#     cloth_type=args.object,
#     env_id=0,
#     traj_id=0,
#     mesh_id=mesh_id,
#     target_mesh_path=target_mesh_path,      # TODO: set
#     action_mode = args.action_mode, # ['line', 'square', 'circular']
#     render= True,
#     headless = args.headless,
#     recording=False,
#     deformation_config  = deformation_config,
#     dynamic_friction = dynamic_friction, 
#     particle_friction = particle_friction,
#     static_friction= static_friction, 
#     stretch_stiffness=stretch_stiffness,
#     bend_stiffness=bend_stiffness,
#     drag = drag,
#     particle_radius = 0.01,  # keep radius close to particle rest distances in mesh to avoid weird behaviors
#     solid_rest_distance = 0.01, # mesh triangulation -> approx 1cm edges lengths
# )
    
#     collect_trajectory(args, **env_kwargs)
    
    