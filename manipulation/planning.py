import numpy as np
from manipulation.envs.gym_env import GymEnv
from manipulation.envs.cloth_env import ClothEnv
from copy import deepcopy
import torch
import meshnet.dataloader_sim as data_loader  
import copy
from manipulation.utils.planning_utils import set_seeds
import os
from manipulation.utils.data_collection import store_data_by_name
import argparse
from meshnet.generate_mesh_predictions import save_mesh, get_mesh_data
from manipulation.planner.mpc import MPC
from manipulation.fold_rendering.renderer import Renderer
from meshnet.viz import plot_mesh, plot_pcd_list
import matplotlib.pyplot as plt
from prettytable import PrettyTable 
import shutil
from manipulation.utils.planning_utils import(
    init_data_dict,
    update_observations,
    array_data_dict,
    get_cloth_params,
    load_goal_meshes,
    load_model,
    init_data_dict_obs,
    get_goal_fold,
    plot_final_mesh,
    move_rendered_images,
    empty_folder
    )
from manipulation.utils.trajectory_gen import (
    get_action_traj,
    visualize_sampled_traj
)
from pyflex_utils.utils import (
    wait_until_scene_is_stable,
)

import imageio
import random
import os
from random import randint

import torch.linalg

from scene_reconstruction.train_utils import regularization, image_losses, densification, train_step
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, RenderResults
import sys
from scene_reconstruction.scene import Scene

from scene_reconstruction.gaussian_mesh import MultiGaussianMesh
from meshnet.meshnet_network import MeshSimulator, ResidualMeshSimulator

from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams, MeshnetParams
from torch.utils.data import DataLoader

from utils.timer import Timer
from utils.external import *
import wandb

from scene_reconstruction.train_utils import SingleStepOptimizer

import lpips
from utils.scene_utils import render_training_image

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


import matplotlib
matplotlib.use("Tkagg")

def closed_loop_planning(args, tiral=0):
    set_seeds()
    
    # set up the cloth parameters and the image renderer
    env_kwargs = get_cloth_params(args)
    
    goal_trajectory = args.traj_idx
    
    A = args.candidates
    H = args.horizon
    action_repetition = args.action_repetition
    velocity = args.velocity
    traj_len = args.traj_len
    
    modality = args.modality
    obs_modalities = {'mpc-oracle': 'gt', 'mpc-ol': 'open_loop', 'fixed': 'gt', 'random': 'gt', 'mpc-cs': 'cloth_splatting', 'mpc-oracle-noise': 'cloth_splatting'}
    obs_modality = obs_modalities[modality]
    
    exp_folder = "./manipulation/experiment_results/"
    experiment_name = f"{args.object}/{modality}/A={A}_H={H}_vel={velocity}_ar={action_repetition}_traj_len={traj_len}/goal_idx={goal_trajectory}/trial={tiral}"
    results_dir = os.path.join(exp_folder, experiment_name)
    results_dir_gaussians = os.path.join(results_dir, 'gaussians')
    results_dir_rendering = os.path.join(results_dir, 'rendering')
    results_dir_rendering = os.getcwd() + results_dir_rendering[1:]
    results_dir_predictions = os.path.join(results_dir_rendering, 'mesh_predictions')
    if os.path.exists(results_dir_rendering):
        shutil.rmtree(results_dir_rendering)
    os.makedirs(results_dir_rendering, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_dir_gaussians, exist_ok=True)
    os.makedirs(results_dir_predictions, exist_ok=True)
    print(f"Saving results to {results_dir}")
    print(f"Saving gaussians to {results_dir_gaussians}")
    print(f"Saving mesh predictions to {results_dir_predictions}")
    print(f"Saving rendering to {results_dir_rendering}")
    
    # create a temporary folder to store the rendered images, then move evertything to the final folder results_dir_rendering
    obj_folder = './tmp'
    os.makedirs(obj_folder, exist_ok=True)
    
    if args.render_images:
        renderer = Renderer(env_kwargs["target_mesh_path"], obj_folder)

    original_model_path = copy.deepcopy(args.gnn_model_path)
    original_data_path = copy.deepcopy(args.data_path)
    meshnet = load_model(args)
    args.data_path = original_data_path
    args.gnn_model_path = original_model_path

    env = ClothEnv(num_steps_per_action=1, **env_kwargs)
    env.reset()
    
    if args.object == 'SHORTS':
        pick =  env._get_cloth_positions()[env.get_keypoint_pick(0)]
        pick[2], pick[1] = pick[1], pick[2]
        goal_place = env._get_cloth_positions()[env.get_keypoint_pick(1)]
        goal_place[2], goal_place[1] = goal_place[1], goal_place[2]
        
    if args.object == 'TSHIRT':
        pick =  env._get_cloth_positions()[env.get_keypoint_pick(11)]
        pick[2], pick[1] = pick[1], pick[2]
        goal_place = env._get_cloth_positions()[env.get_keypoint_pick(4)]
        goal_place[2], goal_place[1] = goal_place[1], goal_place[2]
        
    if args.object == 'TOWEL':
    
        pick =  env._get_cloth_positions()[env.get_keypoint_pick(3)]
        pick[2], pick[1] = pick[1], pick[2]
        goal_place = env._get_cloth_positions()[env.get_keypoint_pick(0)]
        goal_place[2], goal_place[1] = goal_place[1], goal_place[2]
    # plot_pcd_list([env._get_cloth_positions()[::10], env._get_cloth_positions()[env.get_keypoint_pick(keypoint_idx):env.get_keypoint_pick(keypoint_idx)+1]])
    
    
    # # This for now loads the last mesh of the trajectory idx of this sample
    final_particles, init_particles, _, _, _, _ = load_goal_meshes(args)
    goal_particles = get_goal_fold(init_particles, torch.from_numpy(pick).to(torch.float32), torch.from_numpy(goal_place).to(torch.float32))
    
    # load traj data
    init_full = env._get_cloth_positions()
    flipped = copy.deepcopy(init_full)
    # flipped[:, 1], flipped[:, 2] = init_full[:, 2], init_full[:, 1]
    flipped_pick = copy.deepcopy(pick)
    flipped_pick[1], flipped_pick[2] = pick[2], pick[1]
    flipped_goal = copy.deepcopy(goal_place)
    flipped_goal[1], flipped_goal[2] = goal_place[2], goal_place[1]
    goal_particles_mesh = get_goal_fold(torch.from_numpy(copy.deepcopy(flipped)), torch.from_numpy(flipped_pick).to(torch.float32), torch.from_numpy(flipped_goal).to(torch.float32))

    # goal_particles_mesh = get_goal_fold(torch.from_numpy(copy.deepcopy(flipped)), torch.from_numpy(pick).to(torch.float32), torch.from_numpy(goal_place).to(torch.float32))
    


    # initialize and update observations
    data_dict = init_data_dict()
    data_dict["keypoints_ids"] = list(env.keypoints.values())
    
    # invert x and y for the simulation
    data_dict["pick"] = copy.deepcopy(pick)
    data_dict['pick'][1], data_dict['pick'][2] =  data_dict['pick'][2],  data_dict['pick'][1]
    data_dict["place"] = copy.deepcopy(goal_place)
    data_dict['place'][1], data_dict['place'][2] = data_dict['place'][2], data_dict['place'][1]

    # initialize empty dataloader for meshnet
    ds = data_loader.SamplesClothSimDataset(
        data_path=None,
        input_length_sequence=args.input_sequence_length,
        FLAGS=args,
        dt=args.dt,
        knn=args.knn,
        delaunay=True * args.delaunay,
        subsample=True * args.subsample,
        num_samples=args.num_samples,
        sim_data=True,
    )

    # pick the particle, remember to swap y and z for the simulation
    pick_swapped = pick.copy()
    pick_swapped[1], pick_swapped[2] = pick[2], pick[1]
    pick_idx = env.get_closest_point(pick_swapped)
    env.action_tool.grasp_particle(pick_idx)
    obs = env._get_obs()
    
    data_dict = init_data_dict_obs(data_dict, obs, env.camera_names)  
    # update the dataset 
    goal_particles = ds.collect_observation(array_data_dict(data_dict), first=True, modality='gt')
    
    
    mesh = get_mesh_data(torch.from_numpy(flipped).to(torch.float32), ds._data[0]['edge_faces'][0])
    save_mesh(mesh, results_dir_rendering, name=f"full_init_mesh.hdf5")
    
    flipped_goal = copy.deepcopy(goal_particles_mesh)
    # flipped_goal[:, 1], flipped_goal[:, 2] = goal_particles_mesh[:, 2], goal_particles_mesh[:, 1]
    mesh = get_mesh_data(flipped_goal.to(torch.float32), ds._data[0]['edge_faces'][0])
    save_mesh(mesh, results_dir_rendering, name=f"goal_mesh.hdf5")
    
        
    # for i in range(len(data_dict['pos'])):
    data_dict['predicted_pos'].append(ds._data[0]['pos'][0])
    data_dict['refined_pos'].append(ds._data[0]['pos'][0])  

    
    if args.modality in ['mpc-cs']:
        max_step = 30
        iteration_per_time =  1000
        total_steps = opt_param.static_reconst_iteration + iteration_per_time * max_step
        opt_param.iterations = total_steps
        opt_param.pruning_until_iter = total_steps * 0.8
        opt_param.densify_until_iter = total_steps * 0.8
        opt_param.position_lr_max_steps = total_steps * 0.8
        # upload the firts graph, 
        # This has to be done every time that ds gets updated
        init_mesh = get_mesh_data(torch.from_numpy(ds._data[0]['pos'][0]).to(torch.float32), ds._data[0]['edge_faces'][0])
        save_mesh(init_mesh, results_dir_rendering, name=f"init_mesh.hdf5")
        
        # upload mesh predictions: skip the first one that is a repetition of the initial mesh
        for i, pos in enumerate(data_dict['refined_pos']):
            mesh = get_mesh_data(torch.from_numpy(pos).to(torch.float32), ds._data[0]['edge_faces'][0])
            save_mesh(mesh, results_dir_predictions, name=f"mesh_{i:03}.hdf5")
        
        # render the initial scene, but skip the first one that is a repetition of the initial mesh
        renderer.process_obj_traj(data_dict, time_start=1, sim_data=True)
        renderer.obj_to_rgb()
        
        source_dir = os.path.join(obj_folder, 'planning_datasets', 'splits', 'train')
        destination_dir = os.path.join(results_dir_rendering, 'train')
        json_file_path = os.path.join(obj_folder, 'planning_datasets', 'splits')
        json_destination_path = os.path.join(results_dir_rendering)
        os.makedirs(destination_dir, exist_ok=True)
        move_rendered_images(source_dir, destination_dir, json_file_path, json_destination_path)

        # TODO: intilize also masks for the RW case
            
        # Initialize GS
        model_param.source_path = results_dir_rendering
        gs_trainer = SingleStepOptimizer(
        opt_params=opt_param,
        pipeline_params=pipeline_param,
        meshnet_params=meshnet_param,
        model_params=model_param,
        args=user_args,
        save_path=results_dir_gaussians,
        )
        
        gs_trainer.initialize()
        gs_trainer.static_reconstruction()
    


    # initialize planning variables
    t = 0
    max_iterations = 30
    done = False
    
    if modality == 'fixed':
        if args.object == 'TSHIRT':
            height = 0.35
        else:
            height = 0.15
        tilt = 0
        dt = 1
        trajectory, candidate_actions  = get_action_traj(pick, goal_place, height, tilt, velocity, dt, sim_data=False)
        candidate_actions = np.array(candidate_actions).reshape(1, -1, 3)
        # visualize_sampled_traj([trajectory])
    else:
        # initialize the MPC   
        mpc = MPC(meshnet, A=A, H=H, input_sequence_length=args.input_sequence_length)
        mpc.init_sampler(velocity=velocity, action_repetition=action_repetition, pick=pick, goal_place=goal_place, traj_len=traj_len, invert_yz=False)
        trajectories, candidate_actions = mpc.sample_candidate_actions()
        # visualize_sampled_traj(trajectories)

    full_traj_len = len(candidate_actions[0])
    costs = []
    while not done and t < max_iterations:
        print(f"Planning iteration {t}")
        
        if modality == 'random':
            # if t == 0:
            best_action_idx = np.random.choice(A)
            best_actions = candidate_actions[best_action_idx]
            cost_action = 0
        elif modality == 'fixed':
            best_action_idx = 0
            best_actions = candidate_actions[best_action_idx]
            cost_action = 0
        elif modality in ['mpc-oracle', 'mpc-oracle-noise', 'mpc-cs']:
            # Model rollout and cost computation
            model_rollouts = mpc.model_rollout(ds, t=t, regularization_steps=0)        
            best_action_idx, best_actions, cost_action  = mpc.compute_cost(model_rollouts, goal_particles)
        elif modality == 'mpc-ol':
            # Model rollout and cost computation only the first iteration
            if t == 0:
                model_rollouts = mpc.model_rollout(ds, t=t, regularization_steps=0)        
                best_action_idx, best_actions, cost_action  = mpc.compute_cost(model_rollouts, goal_particles)
            else:
                best_actions = best_actions[action_repetition:]
        
        # DEBUG add this line to execute the first trajectory without planning this line that is for debug
        # best_actions = candidate_actions[0, iteration:iteration+1]

        # Execute number of actions corresponding to the action repetition and store the observations from the simulator
        all_obs = []
        for i, action in enumerate(best_actions[: action_repetition]):
            flipped_action = action.copy()  # create a copy of action
            flipped_action[1], flipped_action[2] = action[2], action[1]  # swap y and z
            # TODO: needs to give a flipped action to the simuilator (y and z)
            print(f"Best action: {action}")        
            obs, _, _, _ = env.step(flipped_action, record_continuous_video=False, img_size=args.img_size)
            all_obs.append(copy.deepcopy(obs))
        
        
        # shift candidate actions
        if modality in ['fixed']:
            candidate_actions = candidate_actions[:, action_repetition:] 
        else:
            gripper_pos = copy.deepcopy(obs["gripper_pos"])
            # flip gripper pos
            gripper_pos[1], gripper_pos[2] = gripper_pos[2], gripper_pos[1]
            trajectories, candidate_actions = mpc.update_candidates(gripper_pos=gripper_pos, action_repetition=action_repetition)
            # print(f"New candidate actions time {t}: {candidate_actions.shape}")

        
            
        # update predicted positions to refine them with GS in case of mpc-cs
        model_predictions = [None for _ in range(action_repetition)]
        if modality in ['mpc-ol','mpc-oracle', 'mpc-cs', 'mpc-ol']:
            # store all data in the desired folder and substitute with GS
            model_predictions = [model_rollouts[i][best_action_idx].cpu() for i in range(min(action_repetition, len(model_rollouts)))]
            
        # remove the initial duplicate
        if t == 0:
            for k in ['pos', 'vel', 'actions', 'grasp', 'done', 'gripper_pos']:
                data_dict[k] = [data_dict[k][0]]
            
        for i, action in enumerate(best_actions[: action_repetition]):
            # the actions for the simulator and the dataloader should be flipped
            flipped_action = action.copy()  # create a copy of action
            flipped_action[1], flipped_action[2] = action[2], action[1]  # swap y and z
            if model_predictions[i] is not None:
                data_dict = update_observations(data_dict, all_obs[i], action=flipped_action, grasp=1, predicted_pos=model_predictions[i].numpy(), refined_pos=None,  camera_names=env.camera_names)
            else:
                data_dict = update_observations(data_dict, all_obs[i], action=flipped_action, grasp=1, predicted_pos=None, refined_pos=None,  camera_names=env.camera_names)
        
            
        ######################## REFINEMENTS #####################################################
        if args.modality in ['mpc-cs']:
            init_mesh = get_mesh_data(torch.from_numpy(ds._data[0]['pos'][0]).to(torch.float32), ds._data[0]['edge_faces'][0])
            save_mesh(init_mesh, results_dir_rendering, name=f"init_mesh.hdf5")
            
            # upload refined meshes as history and then GNN predictions: skip the first one that is a repetition of the initial mesh
            total_meshes = 0
            for i, pos in enumerate(data_dict['refined_pos']):
                mesh = get_mesh_data(torch.from_numpy(pos).to(torch.float32), ds._data[0]['edge_faces'][0])
                save_mesh(mesh, results_dir_predictions, name=f"mesh_{total_meshes:03}.hdf5")
                total_meshes += 1
            # total_meshes = 0
            # for i, pos in enumerate(ds._data[0]['pos']):
            #     mesh = get_mesh_data(torch.from_numpy(pos).to(torch.float32), ds._data[0]['edge_faces'][0])
            #     save_mesh(mesh, results_dir_predictions, name=f"mesh_{total_meshes:03}.hdf5")
            #     total_meshes += 1
                
                
            for i, pos in enumerate(model_predictions):
                mesh = get_mesh_data(pos.to(torch.float32), ds._data[0]['edge_faces'][0])
                save_mesh(mesh, results_dir_predictions, name=f"mesh_{total_meshes:03}.hdf5")
                total_meshes += 1
            
            # render the initial scene        
            renderer.process_obj_traj(data_dict, time_start=-action_repetition, sim_data=True)
            renderer.obj_to_rgb()
            
            source_dir = os.path.join(obj_folder, 'planning_datasets', 'splits', 'train')
            destination_dir = os.path.join(results_dir_rendering, 'train')
            json_file_path = os.path.join(obj_folder, 'planning_datasets', 'splits')
            json_destination_path = os.path.join(results_dir_rendering)
            os.makedirs(destination_dir, exist_ok=True)
            move_rendered_images(source_dir, destination_dir, json_file_path, json_destination_path)
                
                
            ########## CS refinements ############
            # TODO: put the if condition 
            gs_trainer.update_data()
            _, simulator = gs_trainer.update_mesh_predictions(iteration_per_time)
        
            
            ####################################################################################
            
            refined_meshes = []
            for n in range(simulator.n_times):
                time_var = torch.tensor(n, dtype=torch.float32, device=meshnet._device).reshape(1, -1)
                # TODO: do we need batch dimensions?
                prediction = simulator(time_var * simulator.time_delta)
                refined_meshes.append(prediction.detach().cpu().numpy())
                
            # reset refined meshes with the updated ones
            data_dict['refined_pos'] = refined_meshes
        ############################################################################################
            

        goal_particles = ds.collect_observation(array_data_dict(data_dict), first=False, modality=obs_modality)
        
        # Compute current distance from goal
        cost = (torch.from_numpy(ds._data[0]['gt_pos'][-1]) - goal_particles).pow(2).mean()
        costs.append(cost)
        
        t += 1
        if t == full_traj_len - 1 or len(candidate_actions[0]) == 0:
            done = True
        
    # TODO: finish the execution by releasing the grasp and compute the final cost/reward
    env.action_tool.release_particle()
    wait_until_scene_is_stable(pyflex_stepper=env.cloth_system.pyflex_stepper, max_steps=500, tolerance=0.05)
    obs = env._get_obs()
    
    data_dict = update_observations(data_dict, obs, action=np.zeros(3), grasp=0, predicted_pos=None, refined_pos=None,  camera_names=env.camera_names)
    
    goal_particles = ds.collect_observation(array_data_dict(data_dict), first=False, modality='gt')
    
    final_observation = ds._data[0]['gt_pos'][-1]
    faces_final = ds._data[0]['edge_faces'][-1]
    grasped_particle = ds._data[0]['grasped_particle']
    
    # mse between state and goal
    final_cost = (torch.from_numpy(final_observation) - goal_particles).pow(2).mean()
    print(f"Final reward: {final_cost}")
    
    costs.append(final_cost)
    # plot the overall costs 
    plt.figure()
    plt.plot(costs)
    plt.title("Costs over time")
    plt.xlabel("Time")
    plt.ylabel("Cost")    
    plt.savefig(os.path.join(results_dir, "costs.png"))
    
    # save costs
    np.save(os.path.join(results_dir, "costs.npy"), np.array(costs))
    # save final cost
    np.save(os.path.join(results_dir, "final_cost.npy"), np.array(final_cost))
    
    mesh_fig = plot_final_mesh(goal_particles, pick, goal_place, final_observation, faces_final, faces_final, return_fig=True)
    # save fig
    mesh_fig.savefig(os.path.join(results_dir, "final_mesh.png"))
    
    # save some final data for debug and visualization
    # data to save: goal, pick, place, pos, gt_pos, predicted_pos, refined_pos, faces, edge_index, actions, grasped_particle
    data_dict.update({'goal_particles': goal_particles,
                      'gt_pos': ds._data[0]['gt_pos'][args.input_sequence_length-1:],
                      'edge_index': ds._data[0]['edge_index'][args.input_sequence_length-1:],
                      'edge_faces': ds._data[0]['edge_faces'][args.input_sequence_length:],})
    store_data_by_name(data_names=list(data_dict.keys()), data=data_dict, path=f'{results_dir}/all_data.h5')
    if modality in ['mpc-cs']:
        gs_trainer.save()
        
    print("Done")
    
    return final_cost
        
def run_batch_experiment(args, iters=10):
    
    args.headless = 1
    results = {}
    for m in [ 'mpc-oracle',  'mpc-ol', 'random']:#['random', 'fixed', 'mpc-oracle',  'mpc-ol']
        results.update({m: []})
        
    if args.object == 'SHORTS':
        args.traj_len = 16
        args.velocity = 0.05
        args.horizon = 10
        args.mesh_id = 1
        
    if args.object == 'TSHIRT':
        args.traj_len = 16
        args.velocity = 0.05
        args.horizon = 10
        
    for mod in results.keys():
        args.modality = mod
        
        for i in range(iters):
            final_cost = closed_loop_planning(args, tiral=i)
            results[mod].append(final_cost)
            
    # print a table of the results showing mean and std of the final costs per each method (results.keys())
    table = PrettyTable()
    table.field_names = ["Method", "Mean", "Std"]
    for key, value in results.items():
        v = np.asarray(value)
        table.add_row([key, np.mean(v), np.std(v)])
        
    print(table)
    save_path = f"./manipulation/experiment_results/{args.object}/results_baselines_{args.action_repetition}.txt"
    
    # save table in the text file as well
    with open(save_path, 'w') as f:
        f.write(str(table))
        
    print(f"Results saved to {save_path}")
    
        
def run_mpc_ablation_experiment(args, iters=10):
    
    args.modality = 'mpc-oracle'
    
    args.headless = 1
    results = {}
    
    A = [100, 200]
    H = [6, 12]
    action_steps = [1, 2]
    velocity = [0.05, 0.08]
    
    for a in A:
        args.candidates = a
        for h in H:
            args.horizon = h
            for ar in action_steps:
                args.action_repetition = ar
                for v in velocity:
                    args.velocity = v
                    name = f"A={a}_H={h}_ar={ar}_v={v}"
                    results.update({name: []})
                    
                    for i in range(iters):
                        final_cost = closed_loop_planning(args, tiral=i)
                        results[name].append(final_cost)
            
    # print a table of the results showing mean and std of the final costs per each method (results.keys())
    table = PrettyTable()
    table.field_names = ["Method", "Mean", "Std"]
    for key, value in results.items():
        v = np.asarray(value)
        # round results
        
        table.add_row([key, np.round(np.mean(v),4), np.round(np.std(v),4)])
        
    print(table)
    save_path = f"./manipulation/experiment_results/{args.object}/ablation_mpc_results_2.txt"
    
    # save table in the text file as well
    with open(save_path, 'w') as f:
        f.write(str(table))
        
    print(f"Results saved to {save_path}")
            
    
def run_cs_experiment(args, iters=10):
    
    args.modality = 'mpc-cs'
    
    args.headless = 1
    results = {}
    
    A = [100]
    H = [6]
    action_steps = [2]
    velocity = [0.05] #, 0.05]
    
    if args.object == 'SHORTS':
        args.traj_len = 16
        args.velocity = 0.05
        velocity = [0.05]
        args.horizon = 10
        H = [10]
        args.mesh_id = 1
        
    if args.object == 'TSHIRT':
        args.traj_len = 16
        args.velocity = 0.05
        velocity = [0.05]
        args.horizon = 10
        H = [10]
    
    for a in A:
        args.candidates = a
        for h in H:
            args.horizon = h
            for ar in action_steps:
                args.action_repetition = ar
                for v in velocity:
                    args.velocity = v
                    name = f"A={a}_H={h}_ar={ar}_v={v}"
                    results.update({name: []})
                    
                    for i in range(iters):
                        final_cost = closed_loop_planning(args, tiral=i)
                        results[name].append(final_cost)
            
    # print a table of the results showing mean and std of the final costs per each method (results.keys())
    table = PrettyTable()
    table.field_names = ["Method", "Mean", "Std"]
    for key, value in results.items():
        v = np.asarray(value)
        # round results
        
        table.add_row([key, np.round(np.mean(v),4), np.round(np.std(v),4)])
        
    print(table)
    save_path = f"./manipulation/experiment_results/{args.object}/mpc_cs_results.txt"
    
    # save table in the text file as well
    with open(save_path, 'w') as f:
        f.write(str(table))
        
    print(f"Results saved to {save_path}")
            
    


if __name__ == "__main__":
    

    
    parser = argparse.ArgumentParser(description="Process some integers.")
    
    ################# CS ARGS ######################
    
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    mp = MeshnetParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i * 500 for i in range(0, 120)])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="arguments/cloth_splatting/continual.py")
    parser.add_argument("--three_steps_batch", type=bool, default=True)
    parser.add_argument("--save_test_images", type=bool, default=True)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="test_project")
    parser.add_argument("--wandb_name", type=str, default="test_name")

    parser.add_argument("--view_skip", default=1, type=int)
    parser.add_argument("--time_skip", type=int, default=1)

    ###
    # model parameters
    ###

    # disable shadow net
    parser.add_argument("--no_shadow", action="store_true")

    # regularization
    # momentum term
    parser.add_argument("--reg_iter", default=5000, type=int)
    parser.add_argument("--knn_update_iter", default=1000, type=int)
    # parser.add_argument("--lambda_momentum", default=0.0, type=float)

    # isometric loss
    parser.add_argument("--lambda_isometric", default=0.0, type=float)

    # shadow loss
    parser.add_argument("--lambda_shadow_mean", default=0.0, type=float)
    parser.add_argument("--lambda_shadow_delta", default=0.0, type=float)
    parser.add_argument("--lambda_momentum_rotation", default=0.0, type=float)
    parser.add_argument("--lambda_spring", default=0.0, type=float)
    parser.add_argument("--lambda_w", default=2000, type=float)
    parser.add_argument("--k_nearest", default=20, type=int)
    parser.add_argument("--single_cam_video", action="store_true", help='Only render from the first camera for the video viz')

    ################## PLANNING ARGS ######################

    parser.add_argument("--modality", type=str, default='mpc-cs', help="Planning modality, the choices are [random, fixed, mpc-oracle, mpc-oracle-noise, mpc-cs, mpc-ol]")
    parser.add_argument("--candidates", type=int, default=100, help="NUmber of candidate actions to sample")
    parser.add_argument("--horizon", type=int, default=6, help="Prediction Horizon of the MPC")
    parser.add_argument("--action_repetition", type=int, default=2, help="Number of actions to exectute after planning with MPC")
    parser.add_argument("--render_images", type=int, default=1, help="Whether to render the images with blender or not")
    parser.add_argument("--velocity", type=int, default=0.05, help="Velocity of the execution (action norm)")
    parser.add_argument("--traj_len", type=int, default=10, help="Length of the trajectory ot sample")

    ################## EXPERIMENTS #######################

    parser.add_argument("--headless", type=int, default=0, help="Whether to run the environment with headless rendering")
    parser.add_argument("--save_data", type=int, default=1, help="Whether to save the data")
    parser.add_argument("--save_video_dir", type=str, default="./data/", help="Path to the saved video")
    parser.add_argument("--img_size", type=int, default=360, help="Size of the recorded videos")
    parser.add_argument("--object", type=str, default="TOWEL", help="Object to load, choices are TOWEL, TSHIRT, SHORTS")
    parser.add_argument("--flat_mesh_dataset", type=str, default="0411_train", help="Dataset of meshes [dev, 00-final, 0411_test, 0411_train]")
    parser.add_argument("-mesh_id", type=int, default=2, help="Id of the mesh we want to use to generating from")
    parser.add_argument("--num_cloths", type=int, default=1, help="Number different cloth in the dataset")
    parser.add_argument("--num_trajectories", type=int, default=1, help="Number of trajectories to generate per cloth")
    parser.add_argument("--dataset_name", type=str, default="0508_test", help="Name of the dataset")
    parser.add_argument("--action_mode", type=str, default="circular", help="how to sample the trajectory, still need to implement variations")

    ############## MESHNET PARAMS #########################

    parser.add_argument("--mode", type=str, default="rollout", choices=["train", "valid", "rollout"], help="Train model, validation or rollout evaluation.")
    parser.add_argument("--model_file", type=str, default="cloth-splatting-SIM-curr0-astep3-propagation15-noise0.0-history2-batch32/model-190.pt", help='Model filename (.pt) to resume from. Can also use "latest" to default to newest file.')
    parser.add_argument("--data_path", type=str, default="./sim_datasets/test_dataset_0415/", help="The dataset directory.")
    # parser.add_argument('--object', type=str, default='TOWEL', help='The dataset directory.')
    # parser.add_argument('--mesh_idx', type=int, default=0, help='Which mesh to load for prediction.')
    parser.add_argument("--traj_idx", type=int, default=6, help="Which trajectory to unroll for predictions.")
    parser.add_argument("--berzelius", type=int, default=0, help="Whether it is running on berzelius or not.")
    parser.add_argument("--wandb", type=int, default=0, help="Whether it is using wandb to log or not.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size.")
    parser.add_argument("--data_name", type=str, default="final_scene_1_gt_eval", help="Name of the dataset file.")
    parser.add_argument("--gnn_model_path", type=str, default="./data/berzelius/model_checkpoint_sim/", help="The path for saving checkpoints of the model.")
    parser.add_argument("--output_path", type=str, default="./data/berzelius/rollouts_pos_sim/", help="The path for saving outputs (e.g. rollouts).")
    parser.add_argument("--train_state_file", type=str, default=None, help='Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.')
    parser.add_argument("--cuda_device_number", type=int, default=None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")
    parser.add_argument("--rollout_filename", type=str, default="rollout", help="Name saving the rollout")
    parser.add_argument("--ntraining_steps", type=int, default=int(2e2), help="Number of training steps.")
    parser.add_argument("--nsave_steps", type=int, default=10, help="Number of steps at which to save the model.")

    # Model parameters and training details
    parser.add_argument("--input_sequence_length", type=int, default=2, help="Length of the sequence in input, default 1.")
    parser.add_argument("--future_sequence_length", type=int, default=1, help="Length of the sequence in input, default 1.")
    parser.add_argument("--curriculum", type=int, default=0, help="Whether to use curriculum learning or not, where curriculum is the # of future steps to predict.")
    parser.add_argument("--action_steps", type=int, default=3, help="Number of actions to predict. Default 1.")
    parser.add_argument("--message_passing", type=int, default=15, help="Number of message passing steps. Default 15")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Noise standard deviation.")
    parser.add_argument("--node_type_embedding_size", type=int, default=1, help="Number of different types of nodes. So far only 1.")
    parser.add_argument("--dt", type=float, default=1.0, help="Simulator delta time.")
    parser.add_argument("--loss_report_step", type=int, default=1, help="Number of steps at which to report the loss.")
    parser.add_argument("--normalize", type=int, default=1, help="Whether to use data normalization or not.")

    # Data Processing
    parser.add_argument("--knn", type=int, default=10, help="Number of neighbors to construct the graph.")
    parser.add_argument("--delaunay", type=int, default=1, help="Whether to use delaunay triangulation or not.")
    parser.add_argument("--subsample", type=int, default=1, help="Whether to subsample or not the initial set of points.")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of points to subsample. Default 300")


    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)


    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    model_param = lp.extract(args)
    hidden_param = hp.extract(args)
    opt_param = op.extract(args)
    pipeline_param = pp.extract(args)
    meshnet_param = mp.extract(args)
    user_args = args

    # closed_loop_planning(args)
    
    run_batch_experiment(args, iters=10)
    
    # run_mpc_ablation_experiment(args, iters=10)
        
    # run_cs_experiment(args, iters=10)

