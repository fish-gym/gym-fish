from numpy.core.arrayprint import _void_scalar_repr
from numpy.lib.function_base import select
from gym_fish.envs.py_util import flare_util
from gym_fish.envs.lib import pyflare as fl
import numpy as np
from .agent_basics import *
from gym_fish.envs.visualization.camera import camera
from pyquaternion import Quaternion


class fluid_sensors:
    def __init__(self,sensors:fl.Markers) -> None:
        self.pos = sensors.getMarkersPos()
        self.normal = sensors.getMarkersNormal()
        self.velocity = sensors.getMarkersVel()
        self.pressure = sensors.getMarkersPressure()


class buoyancy_control_unit:
    def __init__(self,bladder_volume_init:float=1,bladder_volume_min:float=0.1,bladder_volume_max:float=1.3,control_min:float=0.02,control_max:float=0.1) -> None:
        self.bladder_volume_init = bladder_volume_init        
        self.bladder_volume = bladder_volume_init
        self.bladder_volume_min = bladder_volume_min
        self.bladder_volume_max = bladder_volume_max
        self.control_min = control_min
        self.control_max = control_max
    def change(self,delta:float)->None:
        bladder_volume  =  self.bladder_volume+np.clip(delta,self.control_min,self.control_max)
        bladder_volume = np.clip(bladder_volume,self.bladder_volume_min,self.bladder_volume_max)
        self.bladder_volume = bladder_volume
    def reset(self,randomize = False)->None:
        if not randomize:
            self.bladder_volume = self.bladder_volume_init
        else:
            self.bladder_volume = np.random.uniform(low = self.bladder_volume_min,high = self.bladder_volume_max)
        
    @property
    def diff_from_init(self)->float:
        return self.bladder_volume_init-self.bladder_volume

class underwater_agent:
    def __init__(self,skeleton_data:flare_util.skeleton_data) -> None:
        self._dynamics = skeleton_data.dynamics
        param = skeleton_data.param
        self.body_density = param.density
        self.joints = {j.getName():agent_joint(j) for j in self._dynamics.getJoints()}
        self.links = {l.getName():agent_link(l) for l in self._dynamics.getLinks() if l.getName()!='World' and l.getName()!='simple_frame'}
        self.controllable = param.controllable
        self.has_buoyancy = param.has_buoyancy
        self._setup_bcu(param.bladder_volume_min,param.bladder_volume_max,param.bladder_volume_control_min,param.bladder_volume_control_max)
        # the cameras are aligned with the baselink by default
        self.cameras=[]
        # retain data for later update camera
        self.base_link_init_pos = self.base_link.position
        self.base_link_init_rotation = Quaternion(matrix=np.array([self.fwd_axis,self.up_axis,self.right_axis]).transpose())
        self.cameras_init_pos_to_baselink = []
        self.cameras_init_up_axis = []
        self.cameras_init_fwd_axis = []
        for cam in param.cameras:
            _cam = camera()
            _cam.__dict__ = cam
            self.cameras.append(_cam)
            self.cameras_init_pos_to_baselink.append(np.array(_cam.center)-self.base_link.position)
            self.cameras_init_up_axis.append(np.array(_cam.up))
            self.cameras_init_fwd_axis.append(np.array(_cam.target)-np.array(_cam.center))
    def update_cameras(self):
        t = self.base_link.position-self.base_link_init_pos 
        r = self.base_link_init_rotation.inverse*Quaternion(matrix=np.array([self.fwd_axis,self.up_axis,self.right_axis]).transpose())
        for i in range(0,len(self.cameras)):
            camera_up =   r.rotate(self.cameras_init_up_axis[i])
            camera_fwd =   r.rotate(self.cameras_init_fwd_axis[i])
            camera_center = r.rotate(self.cameras_init_pos_to_baselink[i])+t
            camera_target = camera_center+camera_fwd
            self.cameras[i].center = camera_center
            self.cameras[i].up = camera_up
            self.cameras[i].target = camera_target
            
            
    def _setup_bcu(self,bladder_volume_min:float,bladder_volume_max:float,control_min:float,control_max:float):
        self.bcu = buoyancy_control_unit(1,bladder_volume_min,bladder_volume_max,control_min,control_max)
        water_den = 1000
        bcu_unit_bladder_volume_init = self.mass*(1-water_den/self.body_density)/(water_den)
        self.bcu_amplify_ratio = bcu_unit_bladder_volume_init
        # will control in a normalized scheme, while 
    @property
    def sensors(self)->fluid_sensors:
        return  fluid_sensors( self._dynamics.getMarkers())
    @property
    def buoyancy_force(self)->np.array:
        return self.bcu.diff_from_init*1000*(-9.81)*self.bcu_amplify_ratio
    @property
    def mass(self):
        return self._dynamics.getMass()
    @property
    def ctrl_dofs(self):
        if not self.controllable:
            return 0;
        elif self.has_buoyancy:
            return self._dynamics.getNumDofs()+1
        else:
            return self._dynamics.getNumDofs()
    @property
    def collided(self):
        return self._dynamics.beCollided
    @property
    def com(self):
        return self._dynamics.getCOM()
    @property
    def base_link(self):
        return self.links[self.base_link_name];
    @property
    def base_link_name(self):
        return self._dynamics.baseLinkName;
    @property
    def fwd_axis(self):
        return self._dynamics.getBaseLinkFwd();
    @property
    def right_axis(self):
        return self._dynamics.getBaseLinkRight();
    @property
    def up_axis(self):
        return self._dynamics.getBaseLinkUp();
    @property
    def linear_vel(self):
        return self._dynamics.getCOMLinearVelocity()       
    @property
    def angular_vel(self):
        return self._dynamics.getCOMAngularVelocity()     
    @property
    def linear_accel(self):
        return self._dynamics.getCOMLinearAcceleration()
    @property
    def angular_accel(self):
        return self._dynamics.getCOMAngularAcceleration()
    @property
    def positions(self,include_root:bool=False):
        return self._dynamics.getPositions(include_root)
    @property
    def velocities(self,include_root:bool=False):
        return self._dynamics.getVelocities(include_root)
    @property
    def accelerations(self,include_root:bool=False):
        return self._dynamics.accelerations(include_root)    
    @property
    def action_upper_limits(self):
        if not self.controllable:
            return np.array([]);
        elif self.has_buoyancy:
            return np.append(self._dynamics.getForceUpperLimits(),self.bcu.control_max)
        else:
            return self._dynamics.getForceUpperLimits()
    @property
    def action_lower_limits(self):
        if not self.controllable:
            return np.array([]);
        elif self.has_buoyancy:
            return np.append(self._dynamics.getForceLowerLimits(),self.bcu.control_min)
        else:
            return self._dynamics.getForceLowerLimits()
    @property
    def empirical_force_amplifier(self):
        return self._dynamics.empirical_force_amplifier
    def set_empirical_force_amplifier(self,k:float):
        self._dynamics.empirical_force_amplifier = k
    def set_commands(self, commands:np.array):
        if not self.controllable:
            return
        if self.has_buoyancy:
            self._dynamics.setCommands(commands[0:-1])
            self.bcu.change(commands[-1])
            self.apply_buoyancy_force()
        else:
            self._dynamics.setCommands(commands)
        self.last_commands = commands
    # This will make the velocity which is to be used in coupling behavior becomes relative to this frame,
    # this is highly important for make all agents have a common ref frame when they undergoes the same fluid domain
    def set_ref_frame(self,frame:fl.skFrame):
        self._dynamics.setRefFrame(frame)
    def apply_buoyancy_force(self):
        self.base_link.apply_force(np.array([0,self.buoyancy_force,0]))
#         a = self.buoyancy_force/self.mass
#         for l in self.links.values():
#             l.apply_force(np.array([0,l.mass*a,0]))