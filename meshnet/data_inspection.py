from meshnet.viz import plot_mesh, plot_pcd_list, create_gif, plot_mesh_and_points
import sys
import numpy as np
import os
import glob
from meshnet.data_utils import load_sim_traj, process_traj
import matplotlib.pyplot as plt
# change matplotlib backend to save images
plt.switch_backend('agg')
import cv2


if __name__ == '__main__':
    dataset = '0415_debug'     # train_dataset_0414
    dataset = 'train_dataset_0415'     # train_dataset_0414
    obj = 'TOWEL' #'TSHIRT'
    env_ids = ['00000', '00001', '00002', '00003', '00004']
    # env_ids = ['00000']
    for env_id in env_ids:
        data_paths = f'./sim_datasets/{dataset}/{obj}/{env_id}'
        load_keys=['pos',  'gripper_pos', 'camera_0_rgbd']
        data = []
        all_trajs = glob.glob(os.path.join(data_paths, '*'))
        all_trajs.sort()
        traj_id = 2
        for traj_id in range(min(len(all_trajs), 3)):
            data_path = all_trajs[traj_id]
            traj_data = load_sim_traj(data_path, load_keys)
            
            traj_processed = process_traj(traj_data['pos'], dt=0.1, k=3, delaunay=True, subsample=True, num_samples=300, sim_data=True, norm_threshold=0.1)
            # plot_mesh(points = traj_processed['pos'][0], edges=traj_processed['edge_index'][0].T, center_plot=None, white_bkg=False, save_fig=False, file_name='mesh.png')
            # plot_mesh_and_points(mesh_points=traj_processed['pos'][0][:, [0,2,1]], edges=traj_processed['edge_index'][0].T, points=traj_data['gripper_pos'][:, [0, 2, 1]], center_plot=None, white_bkg=False, save_fig=False, file_name='mesh.png')
                # for all rgbd images, temporarly save them and then create a gif
            mesh_images = []
            for i in range(len(traj_processed['pos'])):
                # create png image
                os.makedirs(f'./data/figs/{obj}/{dataset}/{env_id}/{traj_id:05}/mesh', exist_ok=True)
                file_name = os.path.join('./data/figs', obj, dataset, f'{env_id}', f'{traj_id:05}', 'mesh', f'img_{i}.png')
                plot_mesh_and_points(mesh_points=traj_processed['pos'][i][:, [0,2,1]], edges=traj_processed['edge_index'][i].T, points=traj_data['gripper_pos'][:, [0, 2, 1]], center_plot=None, white_bkg=True, save_fig=True, file_name=file_name)

                # plot_mesh(points = traj_processed['pos'][i][:, [0,2,1]], edges=traj_processed['edge_index'][i].T, center_plot=None, white_bkg=True, save_fig=True, file_name=file_name)
                mesh_images.append(file_name)
                plt.close()
                
            save_data_path = f'./data/gifs/{obj}/{dataset}/{env_id}/{traj_id:05}/mesh'
            os.makedirs(save_data_path, exist_ok=True)
            create_gif(mesh_images, os.path.join(save_data_path, 'gif.gif'), fps=10)
            
            # for all rgbd images, temporarly save them and then create a gif
            images = []
            for i in range(traj_data['camera_0_rgbd'].shape[0]):
                image = traj_data['camera_0_rgbd'][i]
                # create png image
                os.makedirs(f'./data/figs/{obj}/{dataset}/{env_id}/{traj_id:05}/rgbd/', exist_ok=True)
                file_name = os.path.join('./data/figs', obj, dataset, f'{env_id}', f'{traj_id:05}', 'rgbd', f'img_{i}.png')
                img = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_name, img)
                # plt.figure()
                # plt.imshow(image[:, :, :3]/255)
                # plt.savefig(file_name)
                images.append(file_name)
                # plt.close()
                
            save_data_path = f'./data/gifs/{obj}/{dataset}/{env_id}/{traj_id:05}/rgbd/'
            os.makedirs(save_data_path, exist_ok=True)
            create_gif(images, os.path.join(save_data_path, 'gif.gif'), fps=10)
        
    
    sys.path.append('..')