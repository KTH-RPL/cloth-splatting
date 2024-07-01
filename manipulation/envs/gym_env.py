import os
import copy
from gym import error
import numpy as np
import gym
from softgym.utils.visualization import save_numpy_as_gif
import cv2
import os.path as osp
import pickle
import pyflex
import pathlib


class GymEnv(gym.Env):
    def __init__(self,
                 headless=False,
                 render=True,
                 target_mesh_path=None,
                 **kwargs):
        
        self.horizon = 1000
        self.env_id = kwargs['env_id']
        self.scene_config = kwargs['scene_config']
        
        self.observation_mode = kwargs['observation_mode']
        self.camera_params = kwargs["camera_params"]
        self.camera_names = kwargs["camera_names"]
        self.camera_name = kwargs["camera_name"]
        self.default_cam = kwargs["camera_name"]
        self.camera_width, self.camera_height = self.camera_params[self.camera_name]['cam_size'][0], self.camera_params[self.camera_name]['cam_size'][1] 

        pyflex.init(headless, render, self.camera_width, self.camera_height, 0)
        
        
        # create output dirs
        if target_mesh_path is not None:
            self.output_dir = pathlib.Path(pathlib.Path(target_mesh_path).parent).parent
            self.output_dir_cam = os.path.join(self.output_dir,f"{self.env_id:04}", "cam_params")
            self.output_dir_data = os.path.join(self.output_dir, f"{self.env_id:04}","data")
            os.makedirs(self.output_dir_cam, exist_ok=True)
            os.makedirs(self.output_dir_data, exist_ok=True)

        # TODO: add video path
        self.record_video, self.video_path, self.video_name = kwargs["recording"], None, None

        self.metadata = {'render.modes': ['human', 'rgb_array']}

        # if device_id == -1 and 'gpu_id' in os.environ:
        #     device_id = int(os.environ['gpu_id'])
        # self.device_id = device_id

        # self.horizon = horizon
        self.time_step = 0
        # self.action_repeat = action_repeat
        self.recording = False
        # self.prev_reward = None
        # self.deterministic = deterministic
        # self.use_cached_states = use_cached_states
        # self.save_cached_states = save_cached_states
        # self.current_config = self.get_default_config()
        # self.current_config_id = None
        # self.cached_configs, self.cached_init_states = None, None
        # self.num_variations = num_variations

        # self.dim_position = 4
        # self.dim_velocity = 3
        # self.dim_shape_state = 14
        # self.particle_num = 0
        # self.eval_flag = False

        # version 1 does not support robot, while version 2 does.
        pyflex_root = os.environ['PYFLEXROOT']
        self.version = 1
        
    def get_cached_configs_and_states(self, cached_states_path, num_variations):
        """
        If the path exists, load from it. Should be a list of (config, states)
        :param cached_states_path:
        :return:
        """
        print("Not implemented yet")
        
    def update_camera(self, camera_name, camera_param=None):
        """
        :param camera_name: The camera_name to switch to
        :param camera_param: None if only switching cameras. Otherwise, should be a dictionary
        :return:
        """
        if camera_param is not None:
            self.camera_params[camera_name] = camera_param
        else:
            camera_param = self.camera_params[camera_name]
        pyflex.set_camera_params(camera_param)
        
    def get_state(self):
        pos = pyflex.get_positions()
        vel = pyflex.get_velocities()
        shape_pos = pyflex.get_shape_states()
        phase = pyflex.get_phases()
        camera_params = copy.deepcopy(self.camera_params)
        return {'particle_pos': pos, 'particle_vel': vel, 'shape_pos': shape_pos, 'phase': phase, 'camera_params': camera_params,
                'config_id': self.current_config_id}
        
    def set_state(self, state_dict):
        pyflex.set_positions(state_dict['particle_pos'])
        pyflex.set_velocities(state_dict['particle_vel'])
        pyflex.set_shape_states(state_dict['shape_pos'])
        pyflex.set_phases(state_dict['phase'])
        self.update_camera(self.camera_name)
        
    def get_colors(self):
        '''
        Overload the group parameters as colors also
        '''
        groups = pyflex.get_groups()
        return groups
    
    def set_colors(self, colors):
        pyflex.set_groups(colors)
        
        
    def start_record(self):
        self.video_frames = []
        self.recording = True
        
        
    def end_record(self, video_path=None, **kwargs):
        if not self.recording:
            print('function end_record: Error! Not recording video')
        self.recording = False
        if video_path is not None:
            save_numpy_as_gif(np.array(self.video_frames), video_path, **kwargs)
        else:
            print("NO path provided to save video. Video not saved")
        del self.video_frames
        
    
    def reset(self, config=None, initial_state=None, config_id=None):
        if config is None:
            congif = self.scene_config
        # TODO: implement the stored config
        #     if config_id is None:
        #         if self.eval_flag:
        #             eval_beg = int(0.8 * len(self.cached_configs))
        #             config_id = np.random.randint(low=eval_beg, high=len(self.cached_configs)) if not self.deterministic else eval_beg
        #         else:
        #             train_high = int(0.8 * len(self.cached_configs))
        #             config_id = np.random.randint(low=0, high=max(train_high, 1)) if not self.deterministic else 0

        #     self.current_config = self.cached_configs[config_id]
        #     self.current_config_id = config_id
        #     self.set_scene(self.cached_configs[config_id], self.cached_init_states[config_id])
        # else:
            # self.current_config = config
        self.set_scene(config, initial_state)
        self.video_frames = []
        self.particle_num = pyflex.get_n_particles()
        self.prev_reward = 0.
        self.time_step = 0
        obs = self._reset()
        if self.recording:
            # TODO: not sure this is working
            self.video_frames.append(self.render(mode='rgb_array'))
        return obs
    
    def step(self, action, record_continuous_video=False, img_size=None):
        """ If record_continuous_video is set to True, will record an image for each sub-step"""
        frames = []
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
    
    
    def initialize_camera(self):
        """
        This function sets the postion and orientation of the camera
        camera_pos: np.ndarray (3x1). (x,y,z) coordinate of the camera
        camera_angle: np.ndarray (3x1). (x,y,z) angle of the camera (in degree).

        Note: to set camera, you need
        1) implement this function in your environement, set value of self.camera_pos and self.camera_angle.
        2) add the self.camera_pos and self.camera_angle to your scene parameters,
            and pass it when initializing your scene.
        3) implement the CenterCamera function in your scene.h file.
        Pls see a sample usage in pour_water.py and softgym_PourWater.h

        if you do not want to set the camera, you can just not implement CenterCamera in your scene.h file,
        and pass no camera params to your scene.
        """
        
    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            rgb, _, _ = pyflex.render(uv=False)
            width, height = self.camera_params[self.camera_name]['cam_size'][0], self.camera_params[self.camera_name]['cam_size'][1]

            # Need to reverse the height dimension
            rgb = np.flip(rgb.reshape([height, width, 4]), 0)[:, :, :3].astype(np.uint8)
        return rgb

    def get_image(self, width=720, height=720):
        """ use pyflex.render to get a rendered image. """
        rgb, _, _ = pyflex.render(uv=False)
        rgb = rgb.astype(np.uint8)
        # if width != rgb.shape[0] or height != rgb.shape[1]:
        #     rgb = cv2.resize(rgb, (width, height))
        return rgb
    
    def get_images(self, width=720, height=720):
        rgb, depth, _ = pyflex.render(uv=False)
        im_width, im_height = self.camera_params[self.camera_name]['cam_size'][0], self.camera_params[self.camera_name]['cam_size'][1]

        # Need to reverse the height dimension
        rgb = np.flip(rgb.reshape([im_height, im_width, 4]), 0)[:, :, :3].astype(np.uint8)
        depth = np.flip(depth.reshape([im_height, im_width]), 0)
        # if (width != rgb.shape[0] or height != rgb.shape[1]) and \
        #         (width is not None and height is not None):
        #     rgb = cv2.resize(rgb, (width, height))
        #     depth = cv2.resize(depth, (width, height))
        return rgb, depth
    
    def set_scene(self, config, state=None):
        """ Set up the flex scene """
        raise NotImplementedError

    def get_default_config(self):
        """ Generate the default config of the environment scenes"""
        raise NotImplementedError

    def generate_env_variation(self, num_variations, **kwargs):
        """
        Generate a list of configs and states
        :return:
        """
        raise NotImplementedError

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        """ set_prev_reward is used for calculate delta rewards"""
        raise NotImplementedError

    def _get_obs(self):
        raise NotImplementedError

    def _get_info(self):
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError

    def _step(self, action):
        raise NotImplementedError

    def _seed(self):
        pass
