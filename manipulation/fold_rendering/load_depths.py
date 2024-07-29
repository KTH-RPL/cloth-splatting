import os
import glob
import pyroexr
import numpy as np

def load_exr(path):
    exr = pyroexr.load(path)
    shape = exr.channels()['B'].shape
    img = np.zeros((shape[0],shape[1],1))
    img[:,:,0] = exr.channels()['R'] 
    img = img[np.newaxis,...]
    return img

# TODO: provide the path to the .exr folder containing the depth images you want to load
path_to_depths = './sim_datasets/test_dataset_0415/TOWEL/00000'
depth_files = glob.glob(os.path.join(path_to_depths, '*.exr'))
depth_files.sort()


depths = None
for j, depth_file in enumerate(depth_files):
    if depths is None:
        depths = load_exr(depth_file)
    else:
        depths = np.concatenate((depths,load_exr(depth_file)),axis=0)