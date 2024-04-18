import pathlib
import numpy as np
import h5py
import cv2


def get_meshes_paths(object_type, flat_mesh_dataset):
    DATA_DIR = pathlib.Path(__file__).parent.parent / "asset"  
    mesh_dir_relative_path = f"flat_meshes/{object_type}/{flat_mesh_dataset}"
    
    mesh_dir_path = DATA_DIR / mesh_dir_relative_path
    mesh_paths = [str(path) for path in mesh_dir_path.glob("*.obj")]
    mesh_paths.sort()
    
    return mesh_paths


def store_data_by_name(data_names, data, path):
    
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        # breakpoint()
        # hf.create_dataset(data_names[i], data= data[i])
        hf.create_dataset(data_names[i], data= data[data_names[i]])
    print(f"Data stored in {path}" )
    hf.close()


def store_nested_data(path, data):
    with h5py.File(path, 'w') as hf:
        def recurse_store(group, key, value):
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    recurse_store(group.create_group(key), sub_key, sub_value)
            elif isinstance(value, list):
                # Assuming lists in your structure are meant to be stored as datasets.
                # Convert value to numpy array first to ensure compatibility.
                # This might need adjustment based on the actual data types in the list.
                group.create_dataset(key, data=np.array(value))
            else:
                # Directly store the value if it's neither dict nor list.
                # This might need adjustment based on the actual data types you have.
                group.create_dataset(key, data=value)

        for key, value in data.items():
            recurse_store(hf, key, value)

    print(f"Data stored in {path}")
    
    
def sample_cloth_params(deformation_config):
    static_friction = np.random.uniform(0.3, 1.0)
    dynamic_friction = np.random.uniform(0.3, 1.0)
    particle_friction = np.random.uniform(0.3, 1.0)
    
    # drag is important to create some high frequency wrinkles
    drag = np.random.uniform(deformation_config.max_drag / 5, deformation_config.max_drag)    
    
    stretch_stiffness = np.random.uniform(0.5, deformation_config.max_stretch_stiffness)
    bend_stiffness = np.random.uniform(0.01, deformation_config.max_bending_stiffness)
    
    return static_friction, dynamic_friction, particle_friction, drag, stretch_stiffness, bend_stiffness