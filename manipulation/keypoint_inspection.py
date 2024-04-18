from meshnet.viz import plot_mesh, plot_pcd_list, create_gif, plot_mesh_and_points
import sys
import numpy as np
import os
import glob
from meshnet.data_utils import load_sim_traj, process_traj
import matplotlib.pyplot as plt
import cv2
# change matplotlib backend to save images
# plt.switch_backend('agg')


if __name__ == '__main__':
    # get first images for each one of the tshirts
    dataset = 'train_dataset_0414'     # train_dataset_0414
    obj = 'TSHIRT'
    mesh_ids = ['00000', '00001', '00002', '00003', '00004']
    for mesh_id in mesh_ids:
        data_paths = f'./sim_datasets/{dataset}/{obj}/{mesh_id}'
        load_keys=['camera_0_rgbd'] 
        data = []
        all_trajs = glob.glob(os.path.join(data_paths, '*'))
        all_trajs.sort()
        iteration_id = 0
        data_path = all_trajs[iteration_id]
        traj_data = load_sim_traj(data_path, load_keys)
        image = traj_data['camera_0_rgbd'][0]
        os.makedirs(f'./data/figs/{obj}/{dataset}/{mesh_id}/{iteration_id:05}/rgbd', exist_ok=True)
        file_name = os.path.join('./data/figs', obj, dataset, f'{mesh_id}', f'{iteration_id:05}', 'rgbd', f'img_0.png')
        # save image as png not with matplotlib
        image = image[:, :, :3]
        # image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_name, image)
        
    
    
    