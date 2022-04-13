from sys import path
from typing import Any, Dict, Tuple

from gym import spaces

from .coupled_env import coupled_env
from .lib import pyflare as fl
from .py_util import np_util as np_util 
import numpy as np
import os
import math
import json

from .visualization.camera import camera


class FishEnvSchooling(coupled_env):
    def __init__(self, 
                control_dt=0.2,
                wa=0.5,
                wd = 1.0,
                max_time = 10,
                done_dist=0.2,
                data_folder = "",
                env_json :str = '../assets/jsons/env_file/env_school.json',
                gpuId: int=0) -> None:
        self.wd = wd
        self.wa = wa
        self.done_dist = done_dist
        self.control_dt=control_dt
        self.max_time = max_time
        self.save=False
        # use parent's init function to init default env data, like action space and observation space, also init dynamics
        super().__init__(data_folder,env_json, gpuId)
        
        self.local_bb_half = np.array([3,1,2])/2

    def _get_scene_camera(self) -> camera:
        return camera(0.03,300,fov=45,window_size=[800,600],center=[-1.96045,3.45622,-3.37217],target=[1.5,0,0])

    def _get_action_space(self) -> spaces.Box:
        entity = self.simulator.solid_solver.GetEntity("chasefish")
        bcu = entity.bcu
        low = entity.GetForceLowerLimits()
        high = entity.GetForceUpperLimits()
        if bcu.enabled:
            low = np.concatenate([low,[-bcu.max_delta_change]])
            high = np.concatenate([high,[bcu.max_delta_change]])
        return spaces.Box(low,high,shape=low.shape)

    def _step(self, action) -> None:
        t = 0
        while t<self.control_dt:
            targetAngle = (3.1415/6)*math.sin(6.28*2*self.simulator.time)
            act = (targetAngle-self.free_robot.GetJoint("spine").GetPositions()[0]-self.free_robot.GetBaseLinkFwd()[2])/self.simulator.dt
            self.free_robot.setCommands(np.ones(self.free_robot_action_dim)*act)
            self.chase_robot.SetCommands(action[:-1])
            if self.chase_robot.bcu.enabled:
                self.chase_robot.bcu.Change(action[-1])
            self.simulator.Step()
            t = t+self.simulator.dt
            self.render_at_framerate()
            self.save_at_framerate(self.save,self.save)
            if not np.isfinite(self._get_obs()).all():
                break
                
    def _get_reward(self, cur_obs, cur_action) :
        dist_reward = self.wd*np.exp(-3* (self.walk_target_dist**2))
#         dist_reward = self.wd*np.exp(-3* (np.abs((self.body_xyz-self.target_xyz)[0])**2))
        
        cur_action =self.normalize_action(cur_action)
        action_reward = -np.sum(np.abs(cur_action)**0.5)*self.wa
        
        total_reward = dist_reward+action_reward
        
        info = {'dist_reward':dist_reward,"action_reward":action_reward}
        return total_reward,info
    
    def outOfLocoBB(self):
        rela_dist = np.abs(self.body_xyz -self.target_xyz)
        return (rela_dist>self.local_bb_half).any()
    
    def _get_done(self) -> bool:
        done = False 
        done = done or self.simulator.time>self.max_time 
        done = done or self.outOfLocoBB()
        done = done or self.walk_target_dist<self.done_dist 
        done = done or self.walk_target_dist>1.2
        done = done or self.collided 
        self.collided = False
        return  done 
    
    def normalize_action(self,action):
        action_space_mean = (self.action_space.low+self.action_space.high)/2
        action_space_std = (self.action_space.high-self.action_space.low)/2
        return np.clip((action-action_space_mean)/action_space_std,-1,1)
    
    def _get_obs(self) -> np.array:
        self._update_state()
        #in local coordinate
        dp_local = np.dot(self.world_to_local,np.transpose(self.target_xyz-self.body_xyz))
        vel_local = np.dot(self.world_to_local,np.transpose(self.vel))
        joint_pos = self.chase_robot.GetPositions()
        joint_vel = self.chase_robot.GetVelocities()
        obs = np.concatenate(
            ([float(self.collided)],
                dp_local,
                vel_local,
                joint_pos/0.52,
                joint_vel/10,
                
        ),axis=0)
        if np.isfinite(obs).all():
            self.last_obs = obs
        return self.last_obs
    
    def _update_state(self):
        self.body_xyz =  self.chase_robot.GetCOM()
        self.target_xyz = self.free_robot.GetCOM()
        self.vel  =  self.chase_robot.GetCOMLinearVelocity()
        # update local matrix
        x_axis = self.chase_robot.GetBaseLinkFwd()
        y_axis = self.chase_robot.GetBaseLinkUp()
        z_axis = self.chase_robot.GetBaseLinkRight()
        self.world_to_local = np.linalg.inv(np.array([x_axis,y_axis,z_axis]).transpose())
        self.walk_target_dist = np.linalg.norm(self.body_xyz-self.target_xyz)
        self.collided = (self.collided or self.free_robot.beCollided())
        self.free_robot_trajectory_points.append(self.target_xyz)
        self.chase_robot_trajectory_points.append(self.body_xyz)

    def _reset_robot(self):
        self.free_robot = self.simulator.solid_solver.GetEntity("freefish")
        self.chase_robot = self.simulator.solid_solver.GetEntity("chasefish")
        self.free_robot_action_dim = self.free_robot.GetControlDOFs()
        
        
    def _reset_task(self):
        pass


    def reset(self) -> Any:
        super().reset()
        self.collided  = False
        self._reset_robot()
        self._reset_task()
        self.free_robot_trajectory_points=[]
        self.chase_robot_trajectory_points=[]
        self._update_state()
        self.last_obs = self._get_obs()
        return self._get_obs()

    def plot3d(self, title=None, fig_name=None, elev=45, azim=45):
        import matplotlib.pyplot as plt 
        path_points = np.array(self.free_robot_trajectory_points+self.chase_robot_trajectory_points)
        ax=plt.figure().add_subplot(111, projection = '3d')
        X = path_points[:,0]
        Y = path_points[:,1]
        Z = path_points[:,2]
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_z - max_range, mid_z + max_range)
        ax.set_zlim(mid_y - max_range, mid_y + max_range)
        
        if self.free_robot_trajectory_points!=None:
            ax.scatter3D(xs=[x[0] for x in self.free_robot_trajectory_points],
                zs=[x[1] for x in self.free_robot_trajectory_points],
                ys=[x[2] for x in self.free_robot_trajectory_points],
                c=[[0,i/len(self.free_robot_trajectory_points),0] for i in range(len(self.free_robot_trajectory_points))])
        if self.chase_robot_trajectory_points!=None:
            ax.scatter3D(xs=[x[0] for x in self.chase_robot_trajectory_points],
                zs=[x[1] for x in self.chase_robot_trajectory_points],
                ys=[x[2] for x in self.chase_robot_trajectory_points],
                c=[[0,0,i/len(self.chase_robot_trajectory_points)] for i in range(len(self.chase_robot_trajectory_points))])
        ax.view_init(elev=elev,azim=azim)#改变绘制图像的视�?即相机的位置,azim沿着z轴旋转，elev沿着y�?        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        if title!=None:
            ax.set_title(title)
        if fig_name!=None:
            plt.savefig(fig_name)
        plt.show()