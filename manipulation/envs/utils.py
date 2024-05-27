import numpy as np
from pyflex_utils.se3 import SE3Container

def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))

    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])


def get_rotation_matrix(angle, axis):
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle)
    c = np.cos(angle)

    m = np.zeros((4, 4))

    m[0][0] = axis[0] * axis[0] + (1.0 - axis[0] * axis[0]) * c
    m[0][1] = axis[0] * axis[1] * (1.0 - c) - axis[2] * s
    m[0][2] = axis[0] * axis[2] * (1.0 - c) + axis[1] * s
    m[0][3] = 0.0

    m[1][0] = axis[0] * axis[1] * (1.0 - c) + axis[2] * s
    m[1][1] = axis[1] * axis[1] + (1.0 - axis[1] * axis[1]) * c
    m[1][2] = axis[1] * axis[2] * (1.0 - c) - axis[0] * s
    m[1][3] = 0.0

    m[2][0] = axis[0] * axis[2] * (1.0 - c) - axis[1] * s
    m[2][1] = axis[1] * axis[2] * (1.0 - c) + axis[0] * s
    m[2][2] = axis[2] * axis[2] + (1.0 - axis[2] * axis[2]) * c
    m[2][3] = 0.0

    m[3][0] = 0.0
    m[3][1] = 0.0
    m[3][2] = 0.0
    m[3][3] = 1.0

    return m

def get_matrix_world_to_camera(cam_pos=[-0.0, 0.82, 0.82], cam_angle=[0, -45 / 180. * np.pi, 0.]):
    cam_x, cam_y, cam_z = cam_pos[0], cam_pos[1], \
                          cam_pos[2]
    cam_x_angle, cam_y_angle, cam_z_angle = cam_angle[0], cam_angle[1], \
                                            cam_angle[2]

    # get rotation matrix: from world to camera
    matrix1 = get_rotation_matrix(- cam_x_angle, [0, 1, 0])
    matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [1, 0, 0])
    rotation_matrix = matrix2 @ matrix1

    # get translation matrix: from world to camera
    translation_matrix = np.zeros((4, 4))
    translation_matrix[0][0] = 1
    translation_matrix[1][1] = 1
    translation_matrix[2][2] = 1
    translation_matrix[3][3] = 1
    translation_matrix[0][3] = - cam_x
    translation_matrix[1][3] = - cam_y
    translation_matrix[2][3] = - cam_z

    return rotation_matrix @ translation_matrix

import h5py
import cv2

def store_data_by_name(data_names, data, path):
    
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        # breakpoint()
        hf.create_dataset(data_names[i], data= data[i])
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


def get_world_coords(rgb, depth, matrix_world_to_camera):
    height, width, _ = rgb.shape
    K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees

    # Apply back-projection: K_inv @ pixels * depth
    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    x = np.linspace(0, width - 1, width).astype(float)
    y = np.linspace(0, height - 1, height).astype(float)
    u, v = np.meshgrid(x, y)
    one = np.ones((height, width, 1))
    x = (u - u0) * depth / fx
    y = (v - v0) * depth / fy
    z = depth
    cam_coords = np.dstack([x, y, z, one])

    # matrix_world_to_camera = get_matrix_world_to_camera(
    #     env.camera_params[env.camera_name]['pos'], env.camera_params[env.camera_name]['angle'])

    # convert the camera coordinate back to the world coordinate using the rotation and translation matrix
    cam_coords = cam_coords.reshape((-1, 4)).transpose()  # 4 x (height x width)
    world_coords = np.linalg.inv(matrix_world_to_camera) @ cam_coords  # 4 x (height x width)
    world_coords = world_coords.transpose().reshape((height, width, 4))

    return world_coords



    
def compute_intrinsics(fov, image_size):
    image_size = float(image_size)
    focal_length = (image_size / 2)\
        / np.tan((np.pi * fov / 180) / 2)
    return np.array([[focal_length, 0, image_size / 2],
                     [0, focal_length, image_size / 2],
                     [0, 0, 1]])

def compute_extrinsics(camera_position, camera_angle):
    camera_position = np.array(camera_position)
    camera_angle = np.array(camera_angle)
    rotation_matrix = SE3Container.from_euler_angles_and_translation(camera_angle).rotation_matrix
    translation = -rotation_matrix @ camera_position
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rotation_matrix
    extrinsics[:3, 3] = translation
    return extrinsics


def pixel_to_3d_position(pixel, depth, camera_params, camera_name='camera_0'):
    """
    Convert a pixel position and depth value from camera_0 to a 3D position in the environment.
    
    Args:
    - pixel: A tuple or list with the (x, y) pixel coordinates.
    - depth: Depth value at the pixel position.
    - camera_params: Dictionary containing camera parameters for camera_0.
    
    Returns:
    - A numpy array with the 3D position in the environment.
    """
    x, y = pixel
    cam_params = camera_params[camera_name]
    
    # Compute intrinsic matrix
    intrinsic = intrinsic_from_fov(cam_params['cam_size'][0], cam_params['cam_size'][1], cam_params['cam_fov'])
    
    # Compute extrinsic matrix
    extrinsic = get_matrix_world_to_camera(cam_pos=cam_params['cam_position'], cam_angle=cam_params['cam_angle'])
    
    # Inverse intrinsic matrix
    intrinsic_inv = np.linalg.inv(intrinsic[:3, :3])
    
    # Convert pixel to normalized camera coordinates
    pixel_homogeneous = np.array([x, y, 1.0])
    normalized_camera_coords = depth * (intrinsic_inv @ pixel_homogeneous)
    
    # Convert normalized camera coordinates to world coordinates
    normalized_camera_coords_homogeneous = np.append(normalized_camera_coords, 1.0)
    world_coords = np.linalg.inv(extrinsic) @ normalized_camera_coords_homogeneous
    
    return world_coords[:3]


def project_3d_to_pixel(world_coords, camera_params):
    """
    Project a 3D position in the environment to a pixel position in the image.
    
    Args:
    - world_coords: A numpy array with the (x, y, z) 3D coordinates.
    - camera_params: Dictionary containing camera parameters for camera_0.
    
    Returns:
    - A tuple with the (x, y) pixel coordinates.
    """
    cam_params = camera_params['camera_0']
    
    # Compute intrinsic matrix
    intrinsic = intrinsic_from_fov(cam_params['cam_size'][0], cam_params['cam_size'][1], cam_params['cam_fov'])
    
    # Compute extrinsic matrix
    extrinsic = get_matrix_world_to_camera(cam_pos=cam_params['cam_position'], cam_angle=cam_params['cam_angle'])
    
    # Convert world coordinates to homogeneous coordinates
    world_coords_homogeneous = np.append(world_coords, 1.0)
    
    # Convert world coordinates to camera coordinates
    camera_coords = extrinsic @ world_coords_homogeneous
    
    # Project camera coordinates to pixel coordinates
    pixel_coords_homogeneous = intrinsic @ camera_coords[:3]
    
    # Normalize the pixel coordinates
    pixel_coords = pixel_coords_homogeneous[:2] / pixel_coords_homogeneous[2]
    
    return int(pixel_coords[0]), int(pixel_coords[1])




    
# def get_scene_config(deformation_config):
#         static_friction = np.random.uniform(0.3, 1.0)
#         dynamic_friction = np.random.uniform(0.3, 1.0)
#         particle_friction = np.random.uniform(0.3, 1.0)
#         # drag is important to create some high frequency wrinkles
#         drag = np.random.uniform(deformation_config.max_drag / 5, deformation_config.max_drag)
            
#         config = create_pyflex_cloth_scene_config(
#         static_friction=static_friction,
#         dynamic_friction=dynamic_friction,
#         particle_friction=particle_friction,
#         drag=drag,
#         particle_radius=0.01,  # keep radius close to particle rest distances in mesh to avoid weird behaviors
#         solid_rest_distance=0.01,  # mesh triangulation -> approx 1cm edges lengths
#         )
#         return config
