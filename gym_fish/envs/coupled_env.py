from typing import Any, Dict, Tuple
import abc
import gym
import os
import cv2
import json
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from .entities.coupled_sim import coupled_sim
from .lib import pyflare as fl
#
from gym_fish.envs.visualization.renderer import  renderer
from gym_fish.envs.visualization.camera import camera

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

def get_full_path(asset_path):
    if asset_path.startswith("/"):
        full_path = asset_path
    else:
        full_path = os.path.join(os.path.dirname(__file__), asset_path)
    if not os.path.exists(full_path):
        raise IOError("File %s does not exist" % full_path)
    return full_path


class coupled_env(gym.Env):
    # can be human, depth array and rgb_array
    render_mode = 'rgb_array'
    def __init__(self,data_folder:str,env_config:str,gpuId:int) -> None:
        super().__init__()
        fl.SetGPUID(gpuId)
        if data_folder=="":
            data_folder = os.getcwd()+"/data/"
        if data_folder.startswith("/"):
            data_folder = os.path.abspath(data_folder)
        else:
            data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), data_folder))
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        self.data_folder = data_folder + '/'
        print("visulization data save folder",self.data_folder)
        #set-up camera
        self.scene_camera = self._get_scene_camera()
        self.gl_renderer = renderer(self.scene_camera)
        self.config_file = env_config
        self.simulator = coupled_sim(config_file=self.config_file)
        self.simulator.solver.SetSaveFolderPath(self.data_folder)
        self.seed()
        self.action_space = self._get_action_space()
        self.observation_space = convert_observation_to_space(self.reset())
    def render_from_camera(self,cam,mode='rgb_array'):
        self.gl_renderer.camera = cam
        color,depth = self.gl_renderer.render()
        self.gl_renderer.camera = self.scene_camera
        if mode=='human':
            return color
        elif mode=='rgb_array':
            return np.array(color)
        elif mode=='depth_array':
            return np.array(depth)

    def render(self, **kwargs):
        color,depth = self.gl_renderer.render()
        if self.render_mode=='human':
            return color
        elif self.render_mode=='rgb_array':
            return np.array(color)
        elif self.render_mode=='depth_array':
            return np.array(depth)
    def save_at_framerate(self,save_solid:bool=True,save_fluid:bool=True,save_coupled_data:bool=False):
        if self.simulator.iter_count%self.simulator.iters_at_framerate==0:
            frame_num=(self.simulator.iter_count / self.simulator.iters_at_framerate)
            self.simulator.save(suffix=str("%05d" % frame_num),save_solid=save_solid,save_fluid=save_fluid,save_coupled_data=save_coupled_data)
    def export_video(self,video_name):
        out = cv2.VideoWriter(video_name+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, self.gl_renderer.camera.window_size)
        for i in range(len(self.render_frames)):
            out.write(np.asarray(self.render_frames[i]))
        out.release()
    def render_at_framerate(self):
        if self.simulator.iter_count%self.simulator.iters_at_framerate==0:
            color,_ = self.gl_renderer.render()
            self.render_frames.append(color)
    def resetDynamics(self):
        self.simulator = coupled_sim(config_file=self.config_file)
        self.gl_renderer.meshes.clear()
        for entity in self.simulator.solid_solver.GetAllEntities().values():
            self.gl_renderer.add_mesh(entity)
    def step(self, action) :
        self._step(action)
        obs = self._get_obs()
        done= self._get_done()
        reward,info = self._get_reward(obs,action)
        return obs,reward,done,info
    @abc.abstractmethod
    def _get_scene_camera(self)->camera:
        pass
    @abc.abstractmethod
    def _get_action_space(self)->spaces.Box:
        pass
    def close(self) -> None:
        del self.simulator
    @abc.abstractmethod
    def _update_state(self)->None:
        pass
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    @abc.abstractmethod
    def _step(self, action)->None:
        pass
    @abc.abstractmethod
    def _get_obs(self)->np.array:
        self._update_state()
        pass
    @abc.abstractmethod
    def _get_reward(self,cur_obs,cur_action):
        pass
    @abc.abstractmethod
    def _get_done(self)->bool:
        pass
    @abc.abstractmethod
    def reset(self) ->np.array:
        self.render_frames =[]
        self.resetDynamics()
        pass




