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
from .entities.fluid_solver import fluid_solver
from .entities.rigid_solver import rigid_solver
from .py_util import flare_util as fl_util
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
def decode_env_json(env_json:str):
    env_path = get_full_path(env_json)
    env_path_folder =os.path.dirname(env_path)
    with open(env_path) as f:
        j = json.load(f)
        rigid_json_name = j["rigid_json"]
        fluid_json_name = j["fluid_json"]
        rigid_json = os.path.abspath(os.path.join(env_path_folder,rigid_json_name))
        fluid_json = os.path.abspath(os.path.join(env_path_folder,fluid_json_name))
        # return rigid_json,fluid_json
        cam =camera()
        cam.__dict__ = j["camera"]
        gl_renderer = renderer(camera=cam)
        return rigid_json,fluid_json,gl_renderer

class coupled_env(gym.Env):
    # can be human, depth array and rgb_array
    render_mode = 'rgb_array'
    def __init__(self,data_folder:str,env_json:str,gpuId:int,couple_mode:fl.COUPLE_MODE =fl.COUPLE_MODE.TWO_WAY,empirical_force_amplifier=1000) -> None:
        super().__init__()
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
        #rigid_json,fluid_json = decode_env_json(env_json=env_json)
        rigid_json,fluid_json,gl_renderer = decode_env_json(env_json=env_json)
        self.gl_renderer= gl_renderer
        self.scene_camera = self.gl_renderer.camera
        # here init dynamics ,action_space and observation space
        self.fluid_json =fluid_json
        self.rigid_json = rigid_json
        self.gpuId =  gpuId
        self.couple_mode  = couple_mode
        self.empirical_force_amplifier= empirical_force_amplifier
            
        self.seed()
        _obs = self.reset()
        self.action_space = self._get_action_space()
        self.observation_space = convert_observation_to_space(_obs)
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

    def render(self):
        color,depth = self.gl_renderer.render()
        if self.render_mode=='human':
            return color
        elif self.render_mode=='rgb_array':
            return np.array(color)
        elif self.render_mode=='depth_array':
            return np.array(depth)
    def save_at_framerate(self,save_objects:bool=True,save_fluid:bool=True):
        if self.simulator.iter_count%self.simulator.iters_at_framerate==0:
            frame_num=(self.simulator.iter_count / self.simulator.iters_at_framerate)
            self.simulator.save(save_objects, save_fluid,suffix=str("%05d" % frame_num))
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
        fluid_param =fl_util.fluid_param()
        fluid_param.from_json(self.fluid_json)
        rigids_data = fl_util.rigid_data(gpuId=self.gpuId)
        rigids_data.from_json(self.rigid_json)
        rigid_solv = rigid_solver(rigids_data)
        fluid_solv = fluid_solver(fluid_param = fluid_param,gpuId=self.gpuId,couple_mode=self.couple_mode)
        self.simulator = coupled_sim(fluid_solv,rigid_solv)
        self.simulator.fluid_solver.set_savefolder(self.data_folder)
        
        self.gl_renderer.meshes.clear()
        for i in range(self.simulator.rigid_solver.agent_num):
            self.gl_renderer.add_mesh(self.simulator.rigid_solver.get_agent(i))
            self.simulator.rigid_solver.get_agent(i).set_empirical_force_amplifier(self.empirical_force_amplifier)
    
    def _get_action_space(self):
        low = self.simulator.rigid_solver.get_action_lower_limits()
        high= self.simulator.rigid_solver.get_action_upper_limits()
        return spaces.Box(low = low,high=high,shape=low.shape)
    def close(self) -> None:
        del self.simulator
    @abc.abstractmethod
    def _update_state(self)->None:
        pass
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    def step(self, action) :
        self._step(action)
        obs = self._get_obs()
        done= self._get_done()
        reward,info = self._get_reward(obs,action)
        return obs,reward,done,info
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




