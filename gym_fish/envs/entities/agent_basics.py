from numpy.core.arrayprint import _void_scalar_repr
from numpy.lib.function_base import select
from gym_fish.envs.lib import pyflare as fl
import numpy as np

class agent_joint:
    def __init__(self,joint:fl.SkeletonJoint) -> None:
        self._joint  = joint

    @property
    def dof(self):
        return self._joint.getNumDofs()
    @property
    def jnt_type(self):
        return self._joint.getJointType()
    @property
    def name(self):
        return self._joint.getName()
    @property
    def velocities(self):
        return self._joint.getVelocities()
    @property
    def positions(self):
        return self._joint.getPositions()
    @property
    def accelerations(self):
        return self._joint.getAccelerations()
    @property
    def force_lower_limits(self):
        return self._joint.getForceLowerLimits()
    @property
    def force_upper_limits(self):
        return self._joint.getForceUpperLimits()
    @property
    def position_lower_limits(self):
        return self._joint.getPositionLowerLimits()
    @property
    def position_upper_limits(self):
        return self._joint.getPositionUpperLimits()
    @property
    def velocity_lower_limits(self):
        return self._joint.getVelocityLowerLimits()
    @property
    def velocity_upper_limits(self):
        return self._joint.getVelocityUpperLimits()
    def setForce(self,dof_idx:int,force:float):
        self._joint.setForce(dof_idx,force)
    def setVelocity(self,dof_idx:int,velocity:float):
        self._joint.setVelocity(dof_idx,velocity)
    def setPosition(self,dof_idx:int,position:float):
        self._joint.setPosition(dof_idx,position)
    def setCommand(self,dof_idx:int,cmd:float):
        self._joint.setCommand(dof_idx,cmd)
    def setPositions(self,positions:np.array):
        self._joint.setPositions(positions)
    def setVelocities(self,vels:np.array):
        self._joint.setPositions(vels)
    def setAccelerations(self,vels:np.array):
        self._joint.setPositions(vels)
    


class agent_link:
    def __init__(self,link:fl.SkeletonLink) -> None:
        self._link = link
    @property
    def mass(self):
        return self._link.getMass()
    @property
    def name(self):
        return self._link.getName()
    @property
    def angular_vel(self):
        return self._link.getAngularVelocity()
    @property
    def linear_vel(self):
        return self._link.getLinearVelocity()
    @property
    def linear_accel(self):
        return self._link.getLinearAcceleration()
    @property
    def angular_accel(self):
        return self._link.getAngularAcceleration()
    @property
    def position(self):
        return self._link.getPosition()
    @property
    def rotation(self):
        return self._link.getRotation()
    @property
    def body_frame(self):
        return self._link.getFrame()
    # this should be a np.array(fx,fy,fz)
    def apply_force(self,force:np.array):
        self._link.applyForce(force)
        



