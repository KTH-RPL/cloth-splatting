import bpy

import json
import os
import pathlib
from typing import List
import argparse
import bpy
import numpy as np
from manipulation.fold_rendering.gen_poses import gen_poses
import glob
import shutil 
import copy
from manipulation.fold_rendering.render_poses_frame import render_poses_frames
    
def get_json(results_path):
    json_path = os.path.join(results_path,'dnerf.json')
    # open json file from json_path 
    data = json.load(open(json_path))

    return data

def filter_json(json_data,split):
    for i in range(len(json_data['frames'])):
        json_data['frames'][i]['file_path'] = './'+split+'/'+json_data['frames'][i]['file_path'].split('/')[-1]
    return json_data


def obj_to_rgb(args, obj_folder, split, json_file):

    tmp_args = copy.deepcopy(args)
    tmp_args.results = os.path.join(args.results,split)
    tmp_args.poses = os.path.join(args.meta,'poses_'+split+'.json')
    
    
    obj_paths = glob.glob(f'{obj_folder}/*.obj')
    obj_paths.sort()
    tmp_args.frame_start = 0
    tmp_args.frame_end = len(obj_paths)-1

    gen_poses(tmp_args,json_file=json_file)
    render_poses_frames(tmp_args, obj_paths=obj_paths)

    json_data = get_json(tmp_args.results)
    json_data_filtered = filter_json(json_data,split)

    # breakpoint()
    with open(os.path.join(tmp_args.meta,'transforms_'+split+'.json'), 'w') as out_file:
        json.dump(json_data_filtered, out_file, indent=4)
    
    # make folder for split
    if not os.path.exists(os.path.join(tmp_args.meta,split)):
        os.makedirs(os.path.join(tmp_args.meta,split))
    else:
        shutil.rmtree(os.path.join(tmp_args.meta,split))
        os.makedirs(os.path.join(tmp_args.meta,split))

    # copy images to split folder
    for i in range(len(json_data_filtered['frames'])):
    # for i in range(2):
        raw_file_path = json_data_filtered['frames'][i]['file_path']
        pacnerf_path = os.path.join(args.results,split,'pacnerf','data',raw_file_path.split('/')[-1])
        shutil.copyfile(pacnerf_path,os.path.join(args.meta,split,raw_file_path.split('/')[-1]))
    
    if args.depth:
        shutil.copyfile(os.path.join(args.results,split,'pacnerf/data/depth.npz'),os.path.join(args.meta,split,'depth.npz'))

    if split == 'train':
        n_views = args.split*args.views
        args.views = np.ceil(n_views).astype(int)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--scale',type=float,default=1.0)
    parser.add_argument('-r_x','--res_x',type=int,default=800)
    parser.add_argument('-r_y','--res_y',type=int,default=800)
    parser.add_argument('--frame_start',type=int,default=0,required=False)
    parser.add_argument('--frame_end',type=int,default=0,required=False)
    parser.add_argument('-o','--output',type=str,default='output')
    parser.add_argument('-r','--results',help='path to results folder',default='results')
    parser.add_argument('-d','--depth',type=int,default=0)
    parser.add_argument('--train_poses',help='Give a json file here to manually set poses',type=str,default='pose_jsons/lego_train.json')
    parser.add_argument('--test_poses',help='Give a json file here to manually set poses',type=str,default='pose_jsons/lego_test.json')
    parser.add_argument('-v','--views',type=int,default=5,required=False)
    parser.add_argument('--stop_motion',type=int,default=0,required=False)
    parser.add_argument('-m','--meta',type=str,default='results/splits')
    parser.add_argument('--format',type=str,choices=['png','exr'],default='png')
    args = parser.parse_args()
    
    #ensure repeatability
    args.meta = os.path.join(args.results,'splits')
    np.random.seed(seed=69)


    obj_path = [ glob.glob('/home/omniverse/workspace/cloth-splatting/sim_datasets/test_dataset_0415/TOWEL/00000/00000/obj/*.obj')]
    obj_path.sort()
    
    rgb_path = "./test_output.png"
    
    splits = ['train','test',]
    json_files = [args.train_poses,args.test_poses]
    # json_files = [args.train_poses,args.test_poses,args.val_poses]
    
    obj_paths = glob.glob('./obj_folder/obj/*.obj')
    obj_paths.sort()
    for split, json_file in zip(splits,json_files):
        obj_to_rgb(args, obj_paths, split, json_file)
                
    # rgb_path = "./test_output.png"
    
    # splits = ['train','test',]
    # json_files = [args.train_poses,args.test_poses,args.val_poses]
    # # splits = ['val']
    # # json_files = [args.val_poses]

    # for split, json_file in zip(splits,json_files):
    
    #     tmp_args = copy.deepcopy(args)
    #     tmp_args.results = os.path.join(args.results,split)
    #     tmp_args.poses = os.path.join(args.meta,'poses_'+split+'.json')


    #     gen_poses(tmp_args,json_file=json_file)
    #     render_poses_frames(tmp_args)
        
    #     json_data = get_json(tmp_args.results)
    #     json_data_filtered = filter_json(json_data,split)
        
        
    #             # breakpoint()
    #     with open(os.path.join(tmp_args.meta,'transforms_'+split+'.json'), 'w') as out_file:
    #         json.dump(json_data_filtered, out_file, indent=4)
        
    #     # make folder for split
    #     if not os.path.exists(os.path.join(tmp_args.meta,split)):
    #         os.makedirs(os.path.join(tmp_args.meta,split))
    #     else:
    #         shutil.rmtree(os.path.join(tmp_args.meta,split))
    #         os.makedirs(os.path.join(tmp_args.meta,split))

    #     # copy images to split folder
    #     for i in range(len(json_data_filtered['frames'])):
    #     # for i in range(2):
    #         raw_file_path = json_data_filtered['frames'][i]['file_path']
    #         pacnerf_path = os.path.join(args.results,split,'pacnerf','data',raw_file_path.split('/')[-1])
    #         shutil.copyfile(pacnerf_path,os.path.join(args.meta,split,raw_file_path.split('/')[-1]))
        
    #     if args.depth:
    #         shutil.copyfile(os.path.join(args.results,split,'pacnerf/data/depth.npz'),os.path.join(args.meta,split,'depth.npz'))

    #     if split == 'train':
    #         n_views = args.split*args.views
    #         args.views = np.ceil(n_views).astype(int)
    #     # obj_to_rgb(obj_path, rgb_path)