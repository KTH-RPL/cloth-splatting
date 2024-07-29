from meshnet.data_utils import load_sim_traj, process_traj, get_env_trajs_path, flip_trajectory
from manipulation.utils.data_collection import get_meshes_paths
import numpy as np
import os
from manipulation.fold_rendering.data_to_obj import process_obj_traj
import argparse
from manipulation.fold_rendering.obj_to_rgb import obj_to_rgb
import shutil
import subprocess

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--scale',type=float,default=1.0)
    parser.add_argument('-r_x','--res_x',type=int,default=800)
    parser.add_argument('-r_y','--res_y',type=int,default=800)
    parser.add_argument('--frame_start',type=int,default=0,required=False)
    parser.add_argument('--frame_end',type=int,default=0,required=False)
    parser.add_argument('-o','--output',type=str,default='output')
    parser.add_argument('-r','--results',help='path to results folder',default='results')
    parser.add_argument('--train_poses',help='Give a json file here to manually set poses',type=str,default='./manipulation/asset/pose_jsons/fold_train.json')
    parser.add_argument('--test_poses',help='Give a json file here to manually set poses',type=str,default='./manipulation/asset/pose_jsons/fold_test.json')
    parser.add_argument('-v','--views',type=int,default=12,required=False)
    parser.add_argument('--stop_motion',type=int,default=0,required=False)
    parser.add_argument('-m','--meta',type=str,default='results/splits')
    parser.add_argument('--material_path',type=str,default='/home/omniverse/workspace/cloth-splatting/manipulation/materials/small_red_square.jpg')
    parser.add_argument('--max_n_frames',type=int,default=None)
    parser.add_argument('-fps','--fps',type=int,default=None)
    parser.add_argument('--z_bias',default=None,type=float)
    parser.add_argument('--bg_scene', nargs='?', default=None, help="Path to the scene.blend file")
    parser.add_argument('-split','--split',type=float,default=1.0)
    parser.add_argument('--object',type=str,default='TOWEL')
    parser.add_argument('--mesh_id',type=int,default=0)
    parser.add_argument('--traj_id',type=int,default=0)
    
    
    
    # RGB
    parser.add_argument('-d','--depth',type=int,default=0)
    parser.add_argument('--format',type=str,choices=['png','exr'],default='png')
    
    # DEPTYH
    # parser.add_argument('-d','--depth',type=int,default=1)
    # parser.add_argument('--format',type=str,choices=['png','exr'],default='exr')

    args = parser.parse_args()    
    
    if args.depth:
        args.meta = args.meta + '_depth'
    
    dataset = 'test_dataset_0415'
    obj_type = args.object # THSIRT TOWEL
    mesh_idx = args.mesh_id        # different meshes of the same object
    traj_idx = args.traj_id
    
    # for traj_idx in range(0, 1, 1):
    folder_idx = "{:05}".format(mesh_idx)
    data_paths = f'./sim_datasets/{dataset}/{obj_type}/{folder_idx}'    # different folds of the same object
    flat_mesh_dataset = '0411_train'
    env_all_trajs = get_env_trajs_path(data_paths)    
    mesh_paths = get_meshes_paths(obj_type, flat_mesh_dataset)
    
    action_steps = 1
    dt = 1,
    k = 3
    delaunay = True
    subsample = False
    num_samples = 300
    sim_data = True
    load_keys=['pos']
    original_mesh_path = mesh_paths[mesh_idx]
    

    data_path = env_all_trajs[0][traj_idx]
    obj_traj_path = process_obj_traj(original_mesh_path, data_path, action_steps,load_keys, sim_data=True)
    
    # rendering the meshes to rgb images
    if not os.path.exists(args.meta):
        os.makedirs(args.meta)
    else:
        shutil.rmtree(args.meta)
        os.makedirs(args.meta)
    
    args.output = data_path
    # check if data already exis, in case skip it
    splits = ['train','test',]
    json_files = [args.train_poses,args.test_poses]
    for split, json_file in zip(splits,json_files):
        obj_to_rgb(args, obj_traj_path, split, json_file)
        
    # move the results/split folder into args.output  
    # check if path already exhists

    if not os.path.exists(os.path.join(args.output, args.meta.split('/')[-1])):    
        shutil.move(args.meta, args.output) 
    else:
        shutil.rmtree(os.path.join(args.output, args.meta.split('/')[-1]))
        shutil.move(args.meta, args.output)
        
    shutil.move(os.path.join(args.output, 'trajectory.npz'), os.path.join(args.output, 'splits', 'trajectory.npz'))
            
    # # TODO: generate meshnet predi/codections and save them as graphs 
    # print("Generating meshnet predictions...")
    # subprocess.run(["python3", "./meshnet/generate_mesh_predictions.py", "--data_path", data_path, "--object", obj_type, 
    #                 "--mesh_idx", str(mesh_idx), "--traj_idx", str(traj_idx)])
        
    print("Generation DONE!")
    print()
    