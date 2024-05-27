import sys, os
import json
import bpy
import numpy as np
import argparse
import matplotlib.pyplot as plt 
import shutil
import copy
DEBUG = False


        

def gen_poses(args,json_file=None):
    
    if args.format == 'exr':
        extension = '.exr'
    elif args.format == 'png':
        extension = '.png'
    else:
        raise ValueError('Invalid format')

    fp = args.results
    if not os.path.exists(fp):
            os.makedirs(fp) 

    json_data = None
    if json_file is not None:
        print("Loading json file: ", json_file)
        f = open(json_file)
        json_data = json.load(f)
        frames = json_data['frames']
    else:
        raise FileNotFoundError

    def listify_matrix(matrix):
        matrix_list = []
        for row in matrix:
            matrix_list.append(list(row))
        return matrix_list
    
    
    num_frames = args.frame_end - args.frame_start + 1 # assume same number of frames for both scenes

    if args.max_n_frames is not None:
        num_frames = min(num_frames,args.max_n_frames)

    print("frame_start: ", args.frame_start)
    print("frame_end: ", args.frame_end)
    
    print("num_frames: ", num_frames)



    # bpy.data.objects['Camera'].data.angle_x = json_data['camera_angle_x']
    fl = 0.5*args.res_x/np.tan(0.5*json_data['camera_angle_x'])

    intrinsics = np.zeros((3,3))
    intrinsics[0,0] = fl
    intrinsics[1,1] = fl
    intrinsics[0,2] = 0.5*args.res_x
    intrinsics[1,2] = 0.5*args.res_y
    intrinsics[2,2] = 1

    angle_x = 2*np.arctan2(0.5*args.res_x,fl)
    angle_y = 2*np.arctan2(0.5*args.res_y,fl)

    out_data = {
        'camera_angle_x': angle_x,
        'camera_angle_y': angle_y,
        'fl_x': fl,
        'fl_y': fl,
        'w': args.res_x,
        'h': args.res_y,
        'cx': args.res_x / 2,
        'cy': args.res_y / 2,
        'n_frames': num_frames,
    }
    dnerf_data = copy.deepcopy(out_data)

    pacnerf_data = []
    
    if args.fps is not None:
        fps = args.fps
    else:
        fps = bpy.context.scene.render.fps
    out_data['frames'] = []
    dnerf_data['frames'] = []

    # breakpoint()
    for i, frame in enumerate(frames):
        if i < args.views:
            frame['transform_matrix'] = np.array(frame['transform_matrix'])
            frame['transform_matrix'][:3,3] = frame['transform_matrix'][:3,3] * args.scale
            
            if args.z_bias is not None:        
                frame['transform_matrix'][2,3] += args.z_bias
            frame['transform_matrix'] = frame['transform_matrix'].tolist()
            frame_data = {
                    'file_path': 'r_'+str(i),
                    'rotation': frame['rotation'],
                    'transform_matrix': frame['transform_matrix']
                }
            out_data['frames'].append(frame_data)
            
            if args.bg_scene is not None:
                # image with no cloth visible
                pac_nerf_frame_data = {
                'file_path': os.path.join('./data','r_'+str(i)+'_-1'+extension),
                'time':-1./fps,
                'c2w': frame['transform_matrix'][:3],
                'intrinsic': listify_matrix(intrinsics)
                }
                pacnerf_data.append(pac_nerf_frame_data)

                dnerf_frame_data = {
                        'file_path': os.path.join('./data','r_'+str(i)+'_-1'+extension),
                        'time': -1./fps,
                        'transform_matrix':  frame['transform_matrix'],
                        'type':'bg'
                    }
                
                dnerf_data['frames'].append(dnerf_frame_data)

            for j in range(num_frames):
                # pacnerf json
                pac_nerf_frame_data = {
                    'file_path': os.path.join('./data','r_'+str(i)+'_'+str(j)+extension),
                    'time':float(j)/float(fps),
                    'c2w': frame['transform_matrix'][:3],
                    'intrinsic': listify_matrix(intrinsics)
                }
                pacnerf_data.append(pac_nerf_frame_data)

                if num_frames > 1 :
                    time = float(j)/(float(num_frames)-1.0)
                else:
                    time = 0.0

                dnerf_frame_data = {
                    'file_path': os.path.join('./data','r_'+str(i)+'_'+str(j)+extension),
                    'time': time,
                    'transform_matrix':  frame['transform_matrix'],
                    'type':'wrap'
                }
                dnerf_data['frames'].append(dnerf_frame_data)


    if not DEBUG:

        with open(os.path.join(args.poses), 'w') as out_file:
            json.dump(out_data, out_file, indent=4)
        
        with open(os.path.join(fp,'pacnerf.json'), 'w') as out_file:
            json.dump(pacnerf_data, out_file, indent=4)
        with open(os.path.join(fp,'dnerf.json'), 'w') as out_file:
            json.dump(dnerf_data, out_file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', nargs='?', default="cloth/scenes/cloth_scaled.blend", help="Path to the scene.blend file")
    # parser.add_argument('output_dir', nargs='?', default="cloth/output/free_flag", help="Path to where the final files will be saved ")
    # parser.add_argument('camera', nargs='?', default="examples/resources/camera_positions", help="Path to the camera file")
    parser.add_argument('--frame_start',type=int,default=-1,required=False)
    parser.add_argument('--frame_end',type=int,default=-1,required=False)
    parser.add_argument('-v','--views',type=int,default=-1,required=False)
    parser.add_argument('-p','--poses',type=str,default='poses.json')
    parser.add_argument('-s','--skip',type=int,default=1,required=False)
    args = parser.parse_args()

    gen_poses(args)
