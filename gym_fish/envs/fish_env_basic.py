from sys import path
from typing import Any, Dict, Tuple
from .coupled_env import coupled_env
from .lib import pyflare as fl
from .py_util import np_util as np_util 
import numpy as np
import os
import math
import json
from pyquaternion import Quaternion

class FishEnvBasic(coupled_env):
    def __init__(self, 
                control_dt=0.2,
                wp= np.array([0.0,1.0]),
                wr= 0.0,
                wa=0.5,
                max_time = 10,
                done_dist=0.1,
                radius = 1,
                # theta should be in the range of [0,180]
                theta = np.array([90,90]),
                # phi should be in the range of [0,360]
                phi = np.array([45,45]),
                data_folder = "",
                env_json :str = '../assets/env_file/env_basic.json',
                gpuId: int=0,
                couple_mode: fl.COUPLE_MODE = fl.COUPLE_MODE.TWO_WAY,
                empirical_force_amplifier =1600,
                 is3D=False,
                 use_com=True
                ) -> None:
        self.use_com = use_com
        self.wr = wr
        self.wp = wp
        self.wa = wa
        self.done_dist = done_dist
        self.theta = theta/180.0*math.pi
        self.phi = phi/180.0*math.pi
        self.control_dt=control_dt
        self.max_time = max_time
        self.radius = radius
        self.save=False
        self.is3D = is3D
        # use parent's init function to init default env data, like action space and observation space, also init dynamics
        super().__init__(data_folder,env_json, gpuId, couple_mode=couple_mode,empirical_force_amplifier=empirical_force_amplifier)


    def _step(self, action) -> None:
        t = 0
        save_fluid = (self.save and self.couple_mode!= fl.COUPLE_MODE.EMPIRICAL)
        while t<self.control_dt:
            self.simulator.iter(action)
            t = t+self.simulator.dt
            self.render_at_framerate()
            if self.save:
                self.save_at_framerate(True,save_fluid)
            if self._simulator_fail():
                break
    def _get_reward(self, cur_obs, cur_action) :
        dist_potential_old = self.dist_potential
        self.dist_potential = self.calc__dist_potential()
        dist_reward = self.wp[0]*np.exp(-3* (self.walk_target_dist**2))+self.wp[1]*float(self.dist_potential - dist_potential_old)
        
        
        cur_action =self.normalize_action(cur_action)
        action_reward = -np.sum(np.abs(cur_action)**0.5)*self.wa
        ori_reward =  - self.rotation_distance*self.wr
        total_reward = dist_reward+action_reward+ori_reward
        
        if (not self._simulator_fail()):
            self.last_reward = total_reward
        info = {'dist_reward':dist_reward,"action_reward":action_reward,"ori_reward":ori_reward}
        return self.last_reward,info

    def _get_done(self) -> bool:
        done = False 
        done = done or self.simulator.time>self.max_time 
        done = done or np.linalg.norm(self.body_xyz-self.goal_pos)<self.done_dist
        done = done or self._simulator_fail()
        return  done 
    def _simulator_fail(self)->bool:
        return not (np.isfinite(self.body_xyz).all() and (np.abs(self.body_xyz)<10).all())
    def normalize_action(self,action):
        action_space_mean = (self.action_space.low+self.action_space.high)/2
        action_space_std = (self.action_space.high-self.action_space.low)/2
        return np.clip((action-action_space_mean)/action_space_std,-1,1)
    def _get_obs(self) -> np.array:
        self._update_state()
        self.trajectory_points.append(self.body_xyz)
        agent = self.simulator.rigid_solver.get_agent(0)
        
        if not self.is3D:
            if agent.has_buoyancy:
                scalar_obs  = np.array([self.angle_to_target,agent.bcu.bladder_volume])
            else:
                scalar_obs= np.array([self.angle_to_target])
            obs = np.concatenate(
                (
                    scalar_obs,
                    self.dp_local,
                    self.vel_local,
                    agent.positions/0.52,
                    agent.velocities/10,
            ),axis=0)
        else:
            if agent.has_buoyancy:
                scalar_obs  = np.array([self.angle_to_target,agent.bcu.bladder_volume,self.rotation_distance])
            else:
                scalar_obs= np.array([self.angle_to_target,self.rotation_distance])
            obs = np.concatenate(
                (
                    scalar_obs,
                    self.dp_local,
                    self.quaternion_diff,
                    self.vel_local,
                    agent.positions/0.52,
                    agent.velocities/10,
            ),axis=0)
        
        if (not self._simulator_fail()):
            self.last_obs = obs
        return self.last_obs

    def _update_state(self):
        agent = self.simulator.rigid_solver.get_agent(0)
        if self.use_com:
            self.body_xyz =  agent.com
            vel  =  agent.linear_vel
        else:
            self.body_xyz = agent.base_link.position
            vel  =  agent.base_link.linear_vel
        # update local matrix
        x_axis = agent.fwd_axis
        y_axis = agent.up_axis
        z_axis = agent.right_axis
        self.world_to_local = np.linalg.inv(np.array([x_axis,y_axis,z_axis]).transpose())
        self.rpy = np.arccos(np.array([x_axis[0],y_axis[1],z_axis[2]]))
        self.walk_target_dist = np.linalg.norm(self.body_xyz-self.goal_pos)
        self.angle_to_target = np.arccos(np.dot(x_axis, (self.goal_pos-self.body_xyz)/self.walk_target_dist ))
        if np.dot((self.goal_pos-self.body_xyz)/self.walk_target_dist,agent.right_axis)<0:
            self.angle_to_target = -self.angle_to_target
        #in local coordinate
        self.dp_local = np.dot(self.world_to_local,np.transpose(self.goal_pos-self.body_xyz))
        self.vel_local = np.dot(self.world_to_local,np.transpose(vel))
        
        quaternion_now  =  Quaternion(matrix=np.array([x_axis,y_axis,z_axis]).transpose())
        # normalize
        self.rotation_distance = Quaternion.absolute_distance(quaternion_now,self.quaternion_desired)
        self.quaternion_diff = (quaternion_now.inverse*self.quaternion_desired).elements


    def calc__dist_potential(self):
        return -self.walk_target_dist /self.control_dt* 4

    def set_task(self,theta,phi,dist):
        agent = self.simulator.rigid_solver.get_agent(0)
        self.init_pos = agent.com
        goal_dir = np.array([math.sin(theta)*math.cos(phi),math.sin(theta)*math.sin(phi),math.cos(theta)])
            
        fwd_vec =goal_dir
        fwd_vec= fwd_vec /(np.linalg.norm(fwd_vec)+1e-8)
        up_vec = np.array([0,1,0])
        right_vec = np.cross(fwd_vec,up_vec)
        right_vec = right_vec/(np.linalg.norm(right_vec)+1e-8)
        up_vec = np.cross(right_vec,fwd_vec)
        up_vec = up_vec /(np.linalg.norm(up_vec)+1e-8)
        r = np.array([fwd_vec,up_vec,right_vec]).transpose()
        self.quaternion_desired  =  Quaternion(matrix=r)
    
        self.goal_pos = self.init_pos+self.radius*goal_dir
        has_sol,start_pts = np_util.generate_traj(self.init_pos,self.goal_pos,dist,visualize=False)
        self.path_start = start_pts[np.random.choice(start_pts.shape[0]),:]
        self.path_start =np.array([self.path_start[0],self.init_pos[1],self.path_start[1]])
        self.path_dir = self.goal_pos-self.path_start
        self.path_dir = self.path_dir/np.linalg.norm(self.path_dir)
        self.path_start = self.goal_pos-self.path_dir*self.radius
    
    def _reset_task(self):
        agent = self.simulator.rigid_solver.get_agent(0)
        agent.bcu.reset(randomize=False)
        theta = self.np_random.uniform(self.theta[0],self.theta[1])
        phi = self.np_random.uniform(self.phi[0],self.phi[1])
        
        dist = 0.01
        self.set_task(theta,phi,dist)


    def reset(self) -> Any:
        super().reset()
        self._reset_task()
        self.trajectory_points=[]
        self._update_state()
        self.dist_potential = self.calc__dist_potential()
        self.last_obs = self._get_obs()
        return self.last_obs

    def plot3d(self, title=None, fig_name=None, elev=45, azim=45):
        import matplotlib.pyplot as plt  
        path_points = np.array([
            self.path_start * (1.0 - t) + self.goal_pos * t for t in np.arange(0.0, 1.0, 1.0 / 100)
    
        ])
        trajectory_points = self.trajectory_points
        ax = plt.figure().add_subplot(111, projection='3d')
        X = path_points[:, 0]
        Y = path_points[:, 1]
        Z = path_points[:, 2]
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
        mid_x = (X.max() + X.min()) * 0.5
        mid_y = (Y.max() + Y.min()) * 0.5
        mid_z = (Z.max() + Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_z - max_range, mid_z + max_range)
        ax.set_zlim(mid_y - max_range, mid_y + max_range)
        ax.scatter3D(xs=X, zs=Y, ys=Z, c='g')
        if trajectory_points != None:
            ax.scatter3D(xs=[x[0] for x in trajectory_points],
                         zs=[x[1] for x in trajectory_points],
                         ys=[x[2] for x in trajectory_points],
                         c=[[0, 0, i / len(trajectory_points)] for i in range(len(trajectory_points))])
        ax.view_init(elev=elev, azim=azim)  # 改变绘制图像的视�?即相机的位置,azim沿着z轴旋转，elev沿着y�?        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        if title != None:
            ax.set_title(title)
        if fig_name != None:
            plt.savefig(fig_name)
        plt.show()