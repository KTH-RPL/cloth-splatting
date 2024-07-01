from meshnet.data_utils import load_sim_traj, process_traj, get_env_trajs_path, flip_trajectory
from manipulation.utils.data_collection import get_meshes_paths
import numpy as np
import os

# this requires to keep in memory the orignal mesh, which should not be a problem?
def create_obj_with_new_vertex_positions_the_hacky_way(
    new_vertex_positions: np.ndarray, file_path: str, target_file_path: str
):
    """trimesh keeps messing with faces and vertices, so this function does a one-on-one replacement of the vertex positions in the obj file."""
    with open(file_path, "r") as f:
        lines = f.readlines()
    vertex_idx = 0
    for line_idx, line in enumerate(lines):
        if line.startswith("v "):
            lines[
                line_idx
            ] = f"v {new_vertex_positions[vertex_idx][0]} {new_vertex_positions[vertex_idx][1]} {new_vertex_positions[vertex_idx][2]}\n"
            vertex_idx += 1
    with open(target_file_path, "w") as f:
        f.writelines(lines)
        
def export_mesh_to_obj(vertices, faces, filename):
    with open(filename, 'w') as file:
        # Write vertices
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # Write faces (OBJ format uses 1-based indexing)
        for face in faces:
            face_str = ' '.join(str(idx + 1) for idx in face)
            file.write(f"f {face_str}\n")
            
def process_obj_traj(original_mesh_path, data_path, action_steps,load_keys, sim_data=False):
    """Process the trajectory data to get the mesh data and save the obj files into data_path/obj folder."""  
    traj_data = load_sim_traj(data_path, action_steps, load_keys)
    if sim_data:
        traj_data = flip_trajectory(traj_data, load_keys)
        
    traj = traj_data['pos']
    # save positions as npz file in data_path
    np.savez(os.path.join(data_path, 'trajectory.npz'), pos=traj)
    obj_traj_path = os.path.join(data_path, 'obj')
    os.makedirs(obj_traj_path, exist_ok=True)
    for t, pos in enumerate(traj):
        create_obj_with_new_vertex_positions_the_hacky_way(
            pos, file_path=original_mesh_path, target_file_path=os.path.join(obj_traj_path, f'{t:05}.obj')
        )
        # trajectory_data = process_traj(traj, dt, k, delaunay, 
        #                                 subsample=subsample, num_samples=num_samples, 
        #                                 sim_data=False, norm_threshold=0.1) 
    print(f"Saved {len(traj)} obj files to {obj_traj_path}")
        
    return obj_traj_path


if __name__=='__main__':
    obj_type = 'TOWEL'
    mesh_idx = 0
    folder_idx = "{:05}".format(mesh_idx)
    data_paths = f'./sim_datasets/test_dataset_0415/{obj_type}/{folder_idx}' 
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
    original_mesh_path =mesh_paths[mesh_idx]
    
    for all_trajs in env_all_trajs:
        for data_path in all_trajs:          
            obj_traj_path = process_obj_traj(original_mesh_path, data_path, action_steps,load_keys, sim_data=True)
    
