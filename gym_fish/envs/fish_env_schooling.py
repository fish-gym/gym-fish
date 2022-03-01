from sys import path
from typing import Any, Dict, Tuple
from .coupled_env import coupled_env
from .lib import pyflare as fl
from .py_util import np_util as np_util 
import numpy as np
import os
import math
import json

class FishEnvSchooling(coupled_env):
    def __init__(self, 
                control_dt=0.2,
                wa=0.5,
                wd = 1.0,
                max_time = 10,
                done_dist=0.2,
                data_folder = "",
                env_json :str = '../assets/env_file/env_school.json',
                gpuId: int=0,
                couple_mode: fl.COUPLE_MODE = fl.COUPLE_MODE.TWO_WAY,empirical_force_amplifier=1600) -> None:
        self.wd = wd
        self.wa = wa
        self.done_dist = done_dist
        self.control_dt=control_dt
        self.max_time = max_time
        self.save=False
        # use parent's init function to init default env data, like action space and observation space, also init dynamics
        super().__init__(data_folder,env_json, gpuId, couple_mode=couple_mode,empirical_force_amplifier=empirical_force_amplifier)
        
        self.local_bb_half = np.array([3,1,2])/2


    def _step(self, action) -> None:
        t = 0
        save_fluid = (self.save and self.couple_mode!= fl.COUPLE_MODE.EMPIRICAL)
        while t<self.control_dt:
            targetAngle = (3.1415/6)*math.sin(6.28*2*self.simulator.time)
#             act = (targetAngle-self.free_robot.joints["spine02"].positions[0])/self.simulator.dt-self.free_robot.fwd_axis[2]*2 
#             act = (targetAngle-self.free_robot.joints["spine"].positions[0])/self.simulator.dt   
            act = (targetAngle-self.free_robot.joints["spine"].positions[0]-self.free_robot.fwd_axis[2])/self.simulator.dt 
            self.free_robot._dynamics.setCommands(np.ones(self.free_robot_action_dim)*act)
            self.simulator.iter(action)
            t = t+self.simulator.dt
            self.render_at_framerate()
            if self.save:
                self.save_at_framerate(True,save_fluid)
#                 self.save_at_framerate(True,False)
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
        joint_pos = self.chase_robot.positions
        joint_vel = self.chase_robot.velocities
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
        self.body_xyz =  self.chase_robot.com
        self.target_xyz = self.free_robot.com
        self.vel  =  self.chase_robot.linear_vel
        # update local matrix
        x_axis = self.chase_robot.fwd_axis
        y_axis = self.chase_robot.up_axis
        z_axis = self.chase_robot.right_axis
        self.world_to_local = np.linalg.inv(np.array([x_axis,y_axis,z_axis]).transpose())
        self.walk_target_dist = np.linalg.norm(self.body_xyz-self.target_xyz)
        self.collided = (self.collided or self.free_robot.collided)
        self.free_robot_trajectory_points.append(self.target_xyz)
        self.chase_robot_trajectory_points.append(self.body_xyz)

    def _reset_robot(self):
        self.free_robot = self.simulator.rigid_solver.get_agent(0)
        self.chase_robot = self.simulator.rigid_solver.get_agent(1)
        self.free_robot_action_dim = self.free_robot._dynamics.getNumDofs()
        if self.free_robot.has_buoyancy:
            self.free_robot_action_dim =self.free_robot_action_dim+1
        frame = self.free_robot.base_link.body_frame
        self.chase_robot.set_ref_frame(frame)
        
        
    def _reset_task(self):
        self.simulator.rigid_solver.get_agent(0).bcu.reset(randomize=False)
        self.simulator.rigid_solver.get_agent(1).bcu.reset(randomize=False)


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