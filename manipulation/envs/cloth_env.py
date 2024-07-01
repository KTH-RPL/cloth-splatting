import numpy as np
from gym.spaces import Box
import pyflex
from manipulation.envs.gym_env import GymEnv
from manipulation.utils.eval_utils import get_current_covered_area
from manipulation.action_space.action_space import Picker, PickerPickPlace, PickerQPG
from copy import deepcopy
from pyflex_utils.se3 import SE3Container
from pyflex_utils.utils import (
    create_pyflex_cloth_scene_config, 
    DeformationConfig, 
    load_cloth_mesh_in_simulator, 
    PyFlexStepWrapper, 
    ClothParticleSystem,
    wait_until_scene_is_stable,
    ParticleGrasperObserver,
    ParticleGrasper,
)
from manipulation.envs.camera_utils import get_world_coor_from_image, get_matrix_world_to_camera, intrinsic_from_fov
import pathlib
import json
from manipulation.envs.utils import (
    compute_intrinsics,
    # get_matrix_world_to_camera,
    pixel_to_3d_position,
    project_3d_to_pixel,
)
import random
import os 
import argparse
from manipulation.utils.trajectory_gen import (
    visualize_multiple_trajectories,
    visualize_trajectory_and_actions,
    generate_bezier_trajectory,
    compute_actions_from_trajectory
)

class ClothEnv(GymEnv):
    def __init__(self, 
                 num_steps_per_action=1,
                 action_mode="line", 
                 render_mode='cloth', 
                 picker_radius=0.05,  
                 picker_threshold=0.005,
                 particle_radius=0.00625, 
                 **kwargs):
        
        self.render_mode = render_mode
        self.action_mode = action_mode
        self.action_repeat = 1
        self.cloth_particle_radius = particle_radius
        self.num_steps_per_action = num_steps_per_action
        
        self.deformation_config = kwargs['deformation_config']
        self.stretch_stiffness = kwargs['scene_config']['stretch_stiffness']
        self.bend_stiffness = kwargs['scene_config']['bend_stiffness']
        
        self.undeformed_mesh_path = kwargs['target_mesh_path']
        self.camera_config = {"camera_params": kwargs['camera_params'],
                              "camera_names": kwargs['camera_names']}
        super().__init__(**kwargs)

        assert self.observation_mode in ['key_point', 'point_cloud', 'cam_rgb']
        
        assert self.action_mode in ['line', 'square', 'circular', 'picker']
        
        # TODO: integrate this part to complete the gym environment
        # self.action_tool = Picker(1, picker_radius=picker_radius, particle_radius=particle_radius, picker_threshold=picker_threshold,
        #                               picker_low=(-1, 0., -1), picker_high=(1.0, 0.5, 1))        
       
        # if self.action_mode == 'picker':
        #     self.action_space = 3    # 3 for the pick and 3 for the place
        #     self.picker_radius = picker_radius            
        # TODO: add our picker
        # assert action_mode in ['picker', 'pickerpickplace', 'sawyer', 'franka', 'picker_qpg']
        # self.observation_mode = observation_mode

        # num_picker = 1
        # if action_mode == 'picker':
            # self.action_tool = Picker(1, picker_radius=picker_radius, particle_radius=particle_radius, picker_threshold=picker_threshold,
            #                           picker_low=(-1, 0., -1), picker_high=(1.0, 0.5, 1))
        #     self.action_space = self.action_tool.action_space
        #     self.picker_radius = picker_radius
        # elif action_mode == 'pickerpickplace':
        #     self.action_tool = PickerPickPlace(num_picker=num_picker, particle_radius=particle_radius, env=self, picker_threshold=picker_threshold,
        #                                        picker_low=(-0.5, 0., -0.5), picker_high=(0.5, 0.3, 0.5))
        #     self.action_space = self.action_tool.action_space
        #     assert self.action_repeat == 1
        # elif action_mode == 'picker_qpg':
        #     cam_pos, cam_angle = self.get_camera_params()
        #     self.action_tool = PickerQPG((self.camera_height, self.camera_height), cam_pos, cam_angle, picker_threshold=picker_threshold,
        #                                  num_picker=num_picker, particle_radius=particle_radius, env=self,
        #                                  picker_low=(-0.3, 0., -0.3), picker_high=(0.3, 0.3, 0.3)
        #                                  )
        #     self.action_space = self.action_tool.action_space            
            
        # if self.observation_mode in ['key_point', 'point_cloud']:
        #     if self.observation_mode == 'key_point':
        #         obs_dim = len(self._get_key_point_idx()) * 3
        #     else:
        #         max_particles = 120 * 120
        #         obs_dim = max_particles * 3
        #         self.particle_obs_dim = obs_dim
        #     if action_mode.startswith('picker'):
        #         obs_dim += num_picker * 3
        #     else:
        #         raise NotImplementedError
        #     self.observation_space = Box(np.array([-np.inf] * obs_dim), np.array([np.inf] * obs_dim), dtype=np.float32)
        # elif self.observation_mode == 'cam_rgb':
        #     self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3),
        #                                  dtype=np.float32)
            
    def _step(self, action, velocity=0.05):
        self.action_tool.step(action, velocity)
        # for i in range(self.num_steps_per_action):
        #     pyflex.step()

    
    def save_cameras(self, output_dir_cam=None):
        if output_dir_cam is None:
            output_dir_cam = self.output_dir_cam
        camera_params = {}
        for i in range(len(self.camera_names)):
            height, width = self.camera_params[self.camera_names[i]]["cam_size"][0], self.camera_params[self.camera_names[i]]["cam_size"][1]
            instrinsics = compute_intrinsics(180*self.camera_params[self.camera_names[i]]["cam_fov"]/np.pi, self.camera_params[self.camera_names[i]]["cam_size"][0])

            matrix_world_to_camera = get_matrix_world_to_camera(cam_pos=self.camera_params[self.camera_names[i]]["cam_position"], cam_angle=self.camera_params[self.camera_names[i]]["cam_angle"])
            # save camera params to json
            camera_params[self.camera_names[i]] = {
                            "intrinsic": instrinsics.tolist(),
                            "extrinsic": matrix_world_to_camera.tolist(),
                            }
        

        with open(os.path.join(output_dir_cam, "camera_params.json"), "w") as f:
            json.dump(camera_params, f)
            
    def pixel_to_3d(self, pixel, depth, camera_name='camera_0'):
        cam_params = self.camera_params[camera_name]
        matrix_world_to_camera = get_matrix_world_to_camera(cam_pos=cam_params['cam_position'], cam_angle=cam_params['cam_angle'])
        
        fov = cam_params["cam_fov"]*180/np.pi
        world_coord = get_world_coor_from_image(pixel[0], pixel[1], depth.shape, matrix_world_to_camera, depth, fov=fov)
        return world_coord
        # return pixel_to_3d_position(pixel, depth, self.camera_params, camera_name=camera_name)
    
    
    def project_to_image(self, position, camera_name='camera_0'):
        cam_params = self.camera_params[camera_name]
        matrix_world_to_camera = get_matrix_world_to_camera(cam_pos=cam_params['cam_position'], cam_angle=cam_params['cam_angle'])
        
        height, width = cam_params["cam_size"][0], cam_params["cam_size"][1]
        
        position = position.reshape(-1, 3)
        world_coordinate = np.concatenate([position, np.ones((len(position), 1))], axis=1)  # n x 4
        camera_coordinate = matrix_world_to_camera @ world_coordinate.T  # 3 x n
        camera_coordinate = camera_coordinate.T  # n x 3
        fov = cam_params["cam_fov"]*180/np.pi
        K = intrinsic_from_fov(height, width, fov)  

        u0 = K[0, 2]
        v0 = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        x, y, depth = camera_coordinate[:, 0], camera_coordinate[:, 1], camera_coordinate[:, 2]
        u = (x * fx / depth + u0).astype("int")
        v = (y * fy / depth + v0).astype("int")

        return u, v
        # return project_3d_to_pixel(position, self.camera_params, camera_name=camera_name)
            
    def init_mesh(self, ):
        # TODO: save somewhere these parameters!

        cloth_vertices, _ = load_cloth_mesh_in_simulator(
            self.undeformed_mesh_path,
            cloth_stretch_stiffness=self.stretch_stiffness,
            cloth_bending_stiffness=self.bend_stiffness,
        )

        self.n_particles = len(cloth_vertices)
        
        pyflex_stepper = PyFlexStepWrapper()
        self.cloth_system = ClothParticleSystem(self.n_particles, pyflex_stepper=pyflex_stepper)
        
        inverse_masses = self.cloth_system.get_masses()
        print(f'inverse massses:{inverse_masses}')
        masses = 1.0 / inverse_masses
        masses += np.random.uniform(-np.max(masses) / 10, np.max(masses) / 10, size=masses.shape)
        inverse_masses = 1.0 / masses
        self.cloth_system.set_masses(inverse_masses)
        
        self.cloth_system.center_object()
        
    def _get_cloth_positions(self):
        return self.cloth_system.get_positions()
    
    def _get_cloth_velocities(self):
        return self.cloth_system.get_velocities()
        
    def apply_random_rotation(self,):
        rotation_matrix = SE3Container.from_euler_angles_and_translation(
            np.array(
                [
                    np.random.uniform(0, self.deformation_config.max_orientation_angle),
                    0,
                    np.random.uniform(0, self.deformation_config.max_orientation_angle),
                ]
            )
        ).rotation_matrix

        y_rotation_matrix = SE3Container.from_euler_angles_and_translation(
            np.array([0, np.random.uniform(0, 2 * np.pi), 0])
        ).rotation_matrix
        
        self.cloth_system.set_positions(self.cloth_system.get_positions() @ rotation_matrix @ y_rotation_matrix)
        self.cloth_system.center_object()
        
        wait_until_scene_is_stable(pyflex_stepper=self.cloth_system.pyflex_stepper, max_steps=500, tolerance=0.05)
        
        
    def get_random_pick(self, keypoint_idx=None):
        # TODO: decide how this is integrated
        if np.random.uniform() < self.deformation_config.grasp_keypoint_vertex_probability:
            if keypoint_idx is None:    
                grasp_particle_idx = random.choice(list(self.keypoints.values()))
            else:
                grasp_particle_idx = list(self.keypoints.values())[keypoint_idx]
        else:
            grasp_particle_idx = np.random.randint(0, self.n_particles)        
        self.grasp_keypoint = grasp_particle_idx
            
        return grasp_particle_idx
    
    def get_keypoint_pick(self, keypoint_idx):
        if keypoint_idx is None:    
            grasp_particle_idx = random.choice(list(self.keypoints.values()))
        else:
            grasp_particle_idx = list(self.keypoints.values())[keypoint_idx]
        return grasp_particle_idx
        
    
    def get_random_place(self, grasp_particle_idx=None, keypoint_idx=None):
        if grasp_particle_idx is None:
            grasp_particle_idx = self.get_random_pick()      
            
        place_vector_norm = 0
        grasp_position = self.cloth_system.get_positions()[grasp_particle_idx]
        # Select from keypoints
        if np.random.uniform() < self.deformation_config.place_keypoint_vertex_probability:
            while place_vector_norm < self.deformation_config.min_fold_distance:
                if keypoint_idx is None:
                    place_particle_idx = random.choice(list(self.keypoints.values()))
                else:
                    place_particle_idx = list(self.keypoints.values())[keypoint_idx]
                palce_position = self.cloth_system.get_positions()[place_particle_idx]
                fold_vector = palce_position - grasp_position
                place_vector_norm = np.linalg.norm(fold_vector)
        # select from cloth particles
        else:
            while place_vector_norm < self.deformation_config.min_fold_distance:
                place_particle_idx = np.random.randint(0, self.n_particles)   
                palce_position = self.cloth_system.get_positions()[place_particle_idx]
                fold_vector = palce_position - grasp_position
                place_vector_norm = np.linalg.norm(fold_vector)
            
        # fold_distance = np.random.uniform(0.1, self.deformation_config.max_fold_distance)
        # cloth_center = self.cloth_system.get_center_of_mass()
        # vertex_position = self.cloth_system.get_positions()[grasp_particle_idx]
        # center_direction = np.arctan2(cloth_center[2] - vertex_position[2], cloth_center[0] - vertex_position[0])
        
        # # 70% of time wihtin pi/3 of the center direction. folds outside of the mesh are less interesting.
        # fold_direction = np.random.normal(center_direction, np.pi / 6)

        # fold_vector = np.array([np.cos(fold_direction), 0, np.sin(fold_direction)]) * fold_distance
        return fold_vector
    
    def get_keypoint_place(self, grasp_particle_idx=None, keypoint_idx=None):
        if grasp_particle_idx is None:
            grasp_particle_idx = self.get_random_pick()    
        grasp_position = self.cloth_system.get_positions()[grasp_particle_idx]
        if keypoint_idx is None:
            place_particle_idx = random.choice(list(self.keypoints.values()))
        else:
            place_particle_idx = list(self.keypoints.values())[keypoint_idx]
        palce_position = self.cloth_system.get_positions()[place_particle_idx]
        fold_vector = palce_position - grasp_position
        
        return fold_vector
    
    def get_closest_point(self, point):
        # given a 3D point, we would like to get the closest point in the cloth
        cloth_points = self.cloth_system.get_positions()
        distances = np.linalg.norm(cloth_points - point, axis=1)
        closest_point_idx = np.argmin(distances)
        return closest_point_idx
    
    

    def set_scene(self, camera_name=None, state=None):
        if self.render_mode == 'particle':
            render_mode = 1
        elif self.render_mode == 'cloth':
            render_mode = 2
        elif self.render_mode == 'both':
            render_mode = 3            
        
        if camera_name is None:
            camera_name = self.camera_names[0]
        self.camera_name = camera_name
        print("Set camera name: ", self.camera_name)     
        self.save_cameras()   
        
        pyflex.set_scene(0, self.scene_config)
        pyflex.set_camera_params(self.camera_params[self.camera_name])        
        
        self.init_mesh()
        
        particles = self.cloth_system.get_positions()
        particle_grid_idx = np.array(list(range(particles.shape[0])))
        x_split =  particles.shape[0] // 2
        self.fold_group_a = particle_grid_idx[:x_split].flatten()
        self.fold_group_b = particle_grid_idx[x_split:].flatten()
        
        self.keypoints = json.load(open(self.undeformed_mesh_path.replace(".obj", ".json")))["keypoint_vertices"]   # ['keypoint_vertices', 'area', 'obj_md5_hash']
        
        self.action_tool = ParticleGrasperObserver(self.cloth_system.pyflex_stepper, self.cloth_system, camera_params=self.camera_config)
        if self.action_mode == 'circular':
            self.action_space = 4
            self.action_tool = ParticleGrasper(self.cloth_system.pyflex_stepper)
        
        if state is not None:
            self.set_state(state)
            
    def set_test_color(self, num_particles):
        """
        Assign random colors to group a and the same colors for each corresponding particle in group b
        :return:
        """
        colors = np.zeros((num_particles))
        rand_size = 30
        rand_colors = np.random.randint(0, 5, size=rand_size)
        rand_index = np.random.choice(range(len(self.fold_group_a)), rand_size)
        colors[self.fold_group_a[rand_index]] = rand_colors
        colors[self.fold_group_b[rand_index]] = rand_colors
        self.set_colors(colors)
        
    def set_colors(self, colors):
        pyflex.set_groups(colors)
            
    def reset(self, config=None, camera_name=None, state=None):
        if config is not None:
            self.scene_config = config
            
        self.set_scene(camera_name, state)
        # self.set_test_color(self.n_particles)
        
        wait_until_scene_is_stable(pyflex_stepper=self.cloth_system.pyflex_stepper, max_steps=500, tolerance=0.05)
        
        # self.apply_random_rotation()
        
        self.video_frames = []
        self.particle_num = pyflex.get_n_particles()
        self.prev_reward = 0.
        self.time_step = 0
        obs = self._get_obs()
        if self.recording:
            self.video_frames.append(self.render(mode='rgb_array'))
        return obs
    
    def step(self, action, record_continuous_video=False, img_size=None):
        """ If record_continuous_video is set to True, will record an image for each sub-step"""
        frames = []
        # print(f'Action repeat: {self.action_repeat}')
        for i in range(self.action_repeat):
            self._step(action)
            if record_continuous_video and i % 2 == 0:  # No need to record each step
                frames.append(self.get_image(img_size, img_size))
        obs = self._get_obs()
        reward = self.compute_reward(action, obs, set_prev_reward=True)
        info = self._get_info()

        if self.recording:
            self.video_frames.append(self.render(mode='rgb_array'))
        self.time_step += 1

        done = False
        if self.time_step >= self.horizon:
            done = True
        if record_continuous_video:
            info['flex_env_recorded_frames'] = frames
        return obs, reward, done, info
    
    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        # TODO: do be implemented
        return 0
    
    def compute_coverage(self):
        return get_current_covered_area(self.n_particles, self.cloth_particle_radius)
    
    def _get_obs_old(self):
        if self.observation_mode == 'cam_rgb':
            return self.get_image(self.camera_height, self.camera_width)
        if self.observation_mode == 'point_cloud':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
            pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
            pos[:len(particle_pos)] = particle_pos
        elif self.observation_mode == 'key_point':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
            keypoint_pos = particle_pos[self.keypoints, :3]
            pos = keypoint_pos
        # TODO: Integrate actions
        if self.action_mode in ['sphere', 'picker']:
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            pos = np.concatenate([pos.flatten(), shapes[:, 0:3].flatten()])
        return pos
    
    def _get_obs(self):
        cloth_observations = {}
        for cam in self.camera_names:
            self.camera_name = cam
            pyflex.set_camera_params(self.camera_params[cam])
            rgb, depth = self.get_images()
            # concat rgt and depth
            rgbd = np.concatenate([rgb, depth[:, :, None]], axis=2)
            cloth_observations[cam]= {'rgbd': rgbd}# {'rgb': [], 'depth': []}
        
        pyflex.set_camera_params(self.camera_params[self.default_cam])
        self.camera_name = self.default_cam
            
            
        cloth_observations['pos'] = self._get_cloth_positions()
        cloth_observations['vel'] = self._get_cloth_velocities()
        
        if self.action_tool.is_grasping():
            cloth_observations['gripper_pos'] = self.action_tool.get_particle_position()
        else: 
            cloth_observations['gripper_pos'] = np.ones(3)
            
        # todo: integrate the True flag
        cloth_observations['done'] = False
            
        return cloth_observations
            
        
    
    def _get_info(self):
        return {'to set': []}
    

    
            
            
            
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')

    args = parser.parse_args()
    
    # make the path independent of the machine is run on
    DATA_DIR = pathlib.Path(__file__).parent.parent / "asset"  
    mesh_dir_relative_path = "flat_meshes/TOWEL/dev"
    
    mesh_dir_path = DATA_DIR / mesh_dir_relative_path
    mesh_paths = [str(path) for path in mesh_dir_path.glob("*.obj")]
    target_mesh_path = np.random.choice(mesh_paths)
    
    deformation_config = DeformationConfig()
    
    static_friction = np.random.uniform(0.3, 1.0)
    dynamic_friction = np.random.uniform(0.3, 1.0)
    particle_friction = np.random.uniform(0.3, 1.0)
    # drag is important to create some high frequency wrinkles
    drag = np.random.uniform(deformation_config.max_drag / 5, deformation_config.max_drag)

    env_kwargs = create_pyflex_cloth_scene_config(
    cloth_type='TOWEL',
    env_id=0,
    traj_id=0,
    target_mesh_path=target_mesh_path,      # TODO: set
    action_mode = "circular", # ['line', 'square', 'circular']
    render= True,
    headless = True,
    recording=False,
    deformation_config  = deformation_config,
    dynamic_friction = dynamic_friction, # 0.75,
    particle_friction = particle_friction, #1.0,
    static_friction= static_friction, #0.0,
    drag = drag,
    particle_radius = 0.01,  # keep radius close to particle rest distances in mesh to avoid weird behaviors
    solid_rest_distance = 0.01, # mesh triangulation -> approx 1cm edges lengths
)
    # Generate and save the initial states for running this environment for the first time
    # env_kwargs['use_cached_states'] = False
    # env_kwargs['save_cached_states'] = False
    # env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless

    # if not env_kwargs['use_cached_states']:
    #     print('Waiting to generate environment variations. May take 1 minute for each variation...')
    # env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env = ClothEnv(num_steps_per_action=1, **env_kwargs)
    obs = env.reset()
    # plt.imshow(obs['camera_0']['rgbd'][:, :,:3]/255)
    
    # TODO: implement random sample
    # action = env.action_space.sample()
    
    
    ################## TEST WITH STEP
    
    pick_idx = env.get_random_pick()
    pick = env.cloth_system.get_positions()[pick_idx]
    place =  pick + env.get_random_place(pick_idx)      
    height = 0.2
    tilt  =   np.radians(30)
    velocity = 1
    
    # switch z and y in pick and place
    # pick[1], pick[2] = pick[2], pick[1]
    # place[1], place[2] = place[2], place[1]
    pick[[1, 2]] = pick[[2, 1]]
    place[[1, 2]] = place[[2, 1]]
    
    trajectory = generate_bezier_trajectory(pick, place, height, tilt, velocity, dt=0.01)
    trajectory[:, [1, 2]] = trajectory[:, [2, 1]]
    actions = compute_actions_from_trajectory(trajectory)
    
    visualize_trajectory_and_actions(trajectory, actions)
    env.action_tool.grasp_particle(pick_idx)
    
    for a in actions:
            _, _, _, info = env.step(np.asarray(a), record_continuous_video=True, img_size=args.img_size)
            # _, _, _, info = env.step(np.asarray([a[0], a[2], a[1]]), record_continuous_video=True, img_size=args.img_size)
            # _, _, _, info = env.step(np.asarray([0.0, -0.01, 0]), record_continuous_video=True, img_size=args.img_size)
    
    
    
    ################### TEST WITH THE GRASPER
    
    pick = env.get_random_pick()
    displacement = env.get_random_place(pick)
    lift_height =  deformation_config.max_lift_height
    env.action_tool.grasp_particle(pick)
    env.action_tool.squared_fold_particle(lift_height, displacement)
    env.action_tool.release_particle()
    wait_until_scene_is_stable(pyflex_stepper=env.cloth_system.pyflex_stepper, max_steps=500, tolerance=0.05)
    
    

    pick = env.get_random_pick()
    displacement = env.get_random_place(pick)
    
    
    angle = np.pi * 0.9
    

    # TODO: integrat the circular fold that goes inside as well
    env.action_tool.grasp_particle(pick)
    env.action_tool.circular_fold_particle(displacement, angle, velocity=0.05)
    env.action_tool.release_particle()
    wait_until_scene_is_stable(pyflex_stepper=env.cloth_system.pyflex_stepper, max_steps=500, tolerance=0.05)
    

    action = np.concatenate([env.cloth_system.get_positions()[pick], displacement])
    frames = [env.get_image(args.img_size, args.img_size)]
    for i in range(env.horizon):
        action = env.action_space.sample()
        # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
        # intermediate frames. Only use this option for visualization as it increases computation.
        _, _, _, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
        frames.extend(info['flex_env_recorded_frames'])
        if args.test_depth:
            show_depth()

    if args.save_video_dir is not None:
        save_name = osp.join(args.save_video_dir, args.env_name + '.gif')
        save_numpy_as_gif(np.array(frames), save_name)
        print('Video generated and save to {}'.format(save_name))

