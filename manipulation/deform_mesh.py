import dataclasses
import json
import pathlib
import random
from types import SimpleNamespace

import numpy as np
from pyflex_utils.se3 import SE3Container
from pyflex_utils import (
    ClothParticleSystem,
    # ParticleGrasper,
    ParticleGrasperObserver,
    PyFlexStepWrapper,
    create_pyflex_cloth_scene_config,
    wait_until_scene_is_stable,
)
from pyflex_utils.utils import create_obj_with_new_vertex_positions_the_hacky_way, load_cloth_mesh_in_simulator
# from pyflex.synthetic_cloth_data import DATA_DIR
# from pyflex.synthetic_cloth_data.meshes.utils.projected_mesh_area import get_mesh_projected_xy_area
# from synthetic_cloth_data.utils import get_metadata_dict_for_dataset
import os

# make the path independent of the machine is run on
DATA_DIR = pathlib.Path(__file__).parent / "asset"  
# DATA_DIR = "/home/erik/Downloads/pyflex/custom"

import pyflex
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




@dataclasses.dataclass
class DeformationConfig:
    pass


@dataclasses.dataclass
class ARTFDeformationConfig(DeformationConfig):
    max_bending_stiffness: float = 0.025  # higher becomes unrealistic
    max_stretch_stiffness: float = 2.0
    max_drag: float = 0.00001  # higher -> cloth will start to fall down very sloooow
    max_fold_distance: float = 0.6  # should allow to fold the cloth in half

    max_orientation_angle: float = np.pi / 4  # higher will make the cloth more crumpled

    fold_probability: float = 0.6
    grasp_keypoint_vertex_probability: float = 0.5
    flip_probability: float = 0.4
    lift_probability: float = 0.0
    max_lift_height: float = 0.2


@dataclasses.dataclass
class ClothFunnelsDeformationConfig(DeformationConfig):
    max_bending_stiffness: float = 0.025  # higher becomes unrealistic
    max_stretch_stiffness: float = 2.0
    max_drag: float = 0.00001  # higher -> cloth will start to fall down very sloooow
    max_height: float = 0.5
    max_distance: float = 0.5

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
    #     env.camera_params[env.camera_names]['pos'], env.camera_params[env.camera_names]['angle'])

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

def deform_mesh(
    deformation_config: DeformationConfig, undeformed_mesh_path: str, target_mesh_path: str, gui: bool = False
):
    # create pyflex scene
    pyflex.init(not gui, gui, 480, 480, 0)
    output_dir = pathlib.Path(pathlib.Path(target_mesh_path).parent).parent
    output_dir_cam = os.path.join(output_dir, "cam_params")
    output_dir_img = os.path.join(output_dir, "images")
    os.makedirs(output_dir_cam, exist_ok=True)
    os.makedirs(output_dir_img, exist_ok=True)

    # https://en.wikipedia.org/wiki/Friction
    static_friction = np.random.uniform(0.3, 1.0)
    dynamic_friction = np.random.uniform(0.3, 1.0)
    particle_friction = np.random.uniform(0.3, 1.0)
    # drag is important to create some high frequency wrinkles
    drag = np.random.uniform(deformation_config.max_drag / 5, deformation_config.max_drag)
    # pyflex becomes unstable if radius is set to higher values (self-collisions)
    # and rest_distance seems to be most stable if it is close to the highest edge lengths in the mesh.
    config = create_pyflex_cloth_scene_config(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        particle_friction=particle_friction,
        drag=drag,
        particle_radius=0.01,  # keep radius close to particle rest distances in mesh to avoid weird behaviors
        solid_rest_distance=0.01,  # mesh triangulation -> approx 1cm edges lengths
    )
    pyflex.set_scene(0, config["scene_config"])
    pyflex.set_camera_params(config["camera_params"][config["camera_names"][0]])
    # pyflex.set_camera_params({ "render_type": ["cloth"], "cam_position": [1, 0.5, 0], "cam_angle": [np.pi / 2, -np.pi / 8, 0.0], "cam_size": [480, 480], "cam_fov": 80 / 180 * np.pi,})

    # breakpoint()
    camera_params = {}
    for i in range(len(config["camera_names"])):
        height, width = config["camera_params"][config["camera_names"][i]]["cam_size"][0], config["camera_params"][config["camera_names"][i]]["cam_size"][1]
        instrinsics = compute_intrinsics(180*config["camera_params"][config["camera_names"][i]]["cam_fov"]/np.pi, config["camera_params"][config["camera_names"][i]]["cam_size"][0])
        # intrinsics = intrinsic_from_fov(height, width, fov=180*config["camera_params"][config["cameras"]][config["camera_0"]]["cam_fov"]/np.pi)
                                        #  fov=90)
        matrix_world_to_camera = get_matrix_world_to_camera(cam_pos=config["camera_params"][config["camera_names"][i]]["cam_position"], cam_angle=config["camera_params"][config["camera_names"][i]]["cam_angle"])
                                                # cam_angle=[0, -45 / 180. * np.pi, 0.])
        # save camera params to json
        camera_params[config["camera_names"][i]] = {
                        "intrinsic": instrinsics.tolist(),
                        "extrinsic": matrix_world_to_camera.tolist(),
                        }
        

    with open(os.path.join(output_dir_cam, "camera_params.json"), "w") as f:
        json.dump(camera_params, f)

   
    # import the mesh

    # 0.5 is arbitrary but we don't want too much stretching
    stretch_stiffness = np.random.uniform(0.5, deformation_config.max_stretch_stiffness)
    bend_stiffness = np.random.uniform(0.01, deformation_config.max_bending_stiffness)

    cloth_vertices, _ = load_cloth_mesh_in_simulator(
        undeformed_mesh_path,
        cloth_stretch_stiffness=stretch_stiffness,
        cloth_bending_stiffness=bend_stiffness,
    )

    n_particles = len(cloth_vertices)
    pyflex_stepper = PyFlexStepWrapper()
    cloth_system = ClothParticleSystem(n_particles, pyflex_stepper=pyflex_stepper)

    # randomize masses for drag to have effect
    # (artifact of pyflex?)

    inverse_masses = cloth_system.get_masses()
    print(f'Inverse masses: {inverse_masses}')
    masses = 1.0 / inverse_masses
    masses += np.random.uniform(-np.max(masses) / 10, np.max(masses) / 10, size=masses.shape)
    inverse_masses = 1.0 / masses
    cloth_system.set_masses(inverse_masses)

    if isinstance(deformation_config, ARTFDeformationConfig):
        # separate the rotations, otherwise the y-rotation will be applied before the Z-rotation
        # which can increase the angle of the Z-rotation and thus make the cloth more crumpled
        rotation_matrix = SE3Container.from_euler_angles_and_translation(
            np.array(
                [
                    np.random.uniform(0, deformation_config.max_orientation_angle),
                    0,
                    np.random.uniform(0, deformation_config.max_orientation_angle),
                ]
            )
        ).rotation_matrix

        y_rotation_matrix = SE3Container.from_euler_angles_and_translation(
            np.array([0, np.random.uniform(0, 2 * np.pi), 0])
        ).rotation_matrix
        # breakpoint()
        cloth_system.set_positions(cloth_system.get_positions() @ rotation_matrix @ y_rotation_matrix)
        cloth_system.center_object()

        # drop mesh
        # tolerance empirically determined
        wait_until_scene_is_stable(pyflex_stepper=cloth_system.pyflex_stepper, max_steps=500, tolerance=0.05)
        # breakpoint()

        grasper = ParticleGrasperObserver(pyflex_stepper, cloth_system, camera_params=config)
        # grasper.camera_params = config["camera_params"]
        # rgb, depth = grasper.get_images(width, height)
        # breakpoint()
        # world_coord = get_world_coords(rgb, depth, matrix_world_to_camera)

        # fold towards a random point around the grasp point
        # if np.random.uniform() < deformation_config.fold_probability:

        if np.random.uniform() < deformation_config.grasp_keypoint_vertex_probability:
            # load keypoints from json file
            json_path = undeformed_mesh_path.replace(".obj", ".json")
            keypoints = json.load(open(json_path))["keypoint_vertices"]
            grasp_particle_idx = random.choice(list(keypoints.values()))
        else:
            grasp_particle_idx = np.random.randint(0, n_particles)

        grasper.grasp_particle(grasp_particle_idx)

        fold_distance = np.random.uniform(0.1, deformation_config.max_fold_distance)

        cloth_center = cloth_system.get_center_of_mass()
        vertex_position = cloth_system.get_positions()[grasp_particle_idx]
        center_direction = np.arctan2(cloth_center[2] - vertex_position[2], cloth_center[0] - vertex_position[0])

        # 70% of time wihtin pi/3 of the center direction. folds outside of the mesh are less interesting.
        fold_direction = np.random.normal(center_direction, np.pi / 6)

        fold_vector = np.array([np.cos(fold_direction), 0, np.sin(fold_direction)]) * fold_distance


        # don't fold all the way, as that makes the sim 'force it back to flat' due to the inifite weight of the grasped particle
        grasper.circular_fold_particle(fold_vector, np.pi * 0.9)
        # breakpoint()
        grasper.release_particle()

        
        # TODO: get all the intermediate observations

        # if np.random.uniform() < deformation_config.lift_probability:
        #     lift_particle_idx = np.random.randint(0, n_particles)
        #     grasper.grasp_particle(lift_particle_idx)
        #     grasper.lift_particle(np.random.uniform(0.05, deformation_config.max_lift_height))
        #     grasper.release_particle()

        # if np.random.uniform() < deformation_config.flip_probability:
        #     # lift, flip and drop again to have occluded folds
        #     logger.debug("flipping after folding")
        #     cloth_system.center_object()
        #     # rotate 180 degrees around x axis
        #     cloth_system.set_positions(
        #         cloth_system.get_positions()
        #         @ SE3Container.from_euler_angles_and_translation(np.array([np.pi, 0, 0])).rotation_matrix
        #     )
        #     # lift and drop
        #     cloth_system.set_positions(cloth_system.get_positions() + np.array([0, 0.5, 0]))
        #     wait_until_scene_is_stable(pyflex_stepper=cloth_system.pyflex_stepper, tolerance=0.05)

    elif isinstance(deformation_config, ClothFunnelsDeformationConfig):
        # drop with random theta orientation
        y_rotation_matrix = SE3Container.from_euler_angles_and_translation(
            np.array([0, np.random.uniform(0, 2 * np.pi), 0])
        ).rotation_matrix

        cloth_system.set_positions(cloth_system.get_positions() @ y_rotation_matrix)
        cloth_system.center_object()
        wait_until_scene_is_stable(pyflex_stepper=cloth_system.pyflex_stepper, max_steps=500, tolerance=0.05)
        # grasp random particle and move along vector
        particle_idx = np.random.randint(0, n_particles)
        distance = np.random.uniform(0, deformation_config.max_distance)
        height = np.random.uniform(0, deformation_config.max_height)
        angle = np.random.uniform(0, 2 * np.pi)
        offset = np.array([np.cos(angle) * distance, height, np.sin(angle) * distance])

        grasper = ParticleGrasperObserver(pyflex_stepper)
        grasper.grasp_particle(particle_idx)
        grasper.move_particle(grasper.get_particle_position() + offset)
        grasper.release_particle()

    else:
        raise NotImplementedError(f"deformation config {deformation_config} not implemented")
    wait_until_scene_is_stable(pyflex_stepper=cloth_system.pyflex_stepper, max_steps=200, tolerance=0.05, gripper=grasper)
    cloth_observations = grasper.cloth_observations
    # breakpoint()
    # store_data_by_name(["camera_0", "camera_1", "particles"], cloth_observations, os.path.join(output_dir_img, "cloth_observations.h5"))
    store_data_by_name(["camera_0_rgb", "camera_0_depth", "camera_1_rgb",  "camera_1_depth", "particles"], 
                       [cloth_observations["camera_0"]["rgb"], cloth_observations["camera_0"]["depth"], cloth_observations["camera_1"]["rgb"],  cloth_observations["camera_1"]["depth"], cloth_observations["particles"]],
                       os.path.join(output_dir_img, "cloth_observations.h5"))
    # store_nested_data( os.path.join(output_dir_img, "cloth_observations.h5"), cloth_observations)
    # cloth_system.center_object()

    print(f"Undefromed mesh path: {undeformed_mesh_path}")

    # TODO: save all the meshes
    # export mesh
    # breakpoint()
    for idx in range(len(cloth_observations["particles"])):
        target_mesh_path = os.path.join(pathlib.Path(target_mesh_path).parent, f"{idx:06d}.obj")
        create_obj_with_new_vertex_positions_the_hacky_way(
            cloth_system.get_positions(), undeformed_mesh_path, target_mesh_path
        )

    # cannot use this multiple times in the same process (segfault)
    # so start in new process, in which case there is no need to actually call the clean since all memory will be released anyways.
    # pyflex.clean()


def generate_deformed_mesh(
    deformation_config: DeformationConfig,
    mesh_dir_relative_path: str,
    output_dir_relative_path: str,
    id: int,
    debug: bool = False,
):

    np.random.seed(id)

    mesh_dir_path = DATA_DIR / mesh_dir_relative_path
    mesh_paths = [str(path) for path in mesh_dir_path.glob("*.obj")]
    mesh_path = np.random.choice(mesh_paths)

    filename = f"{id:06d}.obj"
    output_dir_relative_path = DATA_DIR / output_dir_relative_path
    output_dir_relative_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_relative_path / filename

    # generate deformed mesh
    deform_mesh(deformation_config, mesh_path, output_path, gui=debug)

    # # create json file
    flat_mesh_data = json.load(open(mesh_path.replace(".obj", ".json")))
    # write data to json file

    # mesh_area = get_mesh_projected_xy_area(output_path)
    data = {
        "keypoint_vertices": flat_mesh_data["keypoint_vertices"],
        # "area": mesh_area,
        "flat_mesh": {
            "relative_path": mesh_path.replace(f"{DATA_DIR}/", ""),
            "obj_md5_hash": flat_mesh_data["obj_md5_hash"],
            "area": flat_mesh_data["area"],  # duplication, but makes it easier to use later on..
        },
    }

    with open(str(output_path).replace(".obj", ".json"), "w") as f:
        json.dump(data, f)



if __name__ == "__main__":
    import tqdm

    # @hydra.main(config_path="configs", config_name="config_tshirt")
    # @hydra.main(config_path="configs", config_name="config_shorts")
    def generate_deformed_meshes(cfg):

        # write metadata
        data = {
            "num_samples": cfg.num_samples,
            "flat_mesh_dir": cfg.mesh_dir,
        }
        
        # breakpoint()
        cfg.output_dir = cfg.output_dir.replace("pyflex", "pyflex/custom")
        # data.update(get_metadata_dict_for_dataset())
        output_dir = DATA_DIR / pathlib.Path(cfg.output_dir)
        print(f"------------------- output dir: {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        # breakpoint()
        metadata_path = output_dir / "metadata.json"
        json.dump(data, open(metadata_path, "w"))
        print(f"Metadata written to {metadata_path}")

        deformation_config = ARTFDeformationConfig()

        for id in tqdm.trange(cfg.start_id, cfg.start_id + cfg.num_samples):

            # breakpoint()
            generate_deformed_mesh(deformation_config, cfg.mesh_dir, cfg.output_dir.replace('/dev',f'/{id:04}/dev'), id, debug=cfg.debug)
            # breakpoint()
            
    
    cfg = {
        "defaults":{ "deformation": "artf"},
        "mesh_dir": "flat_meshes/TOWEL/dev",
        "output_dir": "deformed_meshes/TOWEL/pyflex/dev",
        "num_samples": 1, #10,
        "start_id": 0,
        "debug": True,
        }
    cfg = SimpleNamespace(**cfg)
    
    # make it such that cfg.nu_samples can be used
    generate_deformed_meshes(cfg)
