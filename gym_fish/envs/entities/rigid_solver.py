
from ..py_util import flare_util
from .underwater_agent import *

import numpy as np

class rigid_solver:
    def __init__(self,rigid_data:flare_util.rigid_data) -> None:
        self._rigid_data = rigid_data
        self._rigid_world =  rigid_data.rigidWorld
        self._agents = [underwater_agent(skeleton_data=sk) for sk in rigid_data.skeletons]
    def get_agent(self,i):
        return self._agents[i]
    def get_action_upper_limits(self)->np.array:
        if self.agent_num>1:
            return np.concatenate([ a.action_upper_limits for a in self._agents])
        else:
            return self._agents[0].action_upper_limits
    def get_action_lower_limits(self)->np.array:
        if self.agent_num>1:
            return np.concatenate([ a.action_lower_limits for a in self._agents])
        else:
            return self._agents[0].action_lower_limits
    def set_commands(self,commands:np.array)->None:
        cmd_offset= 0
        for agent in self._agents:
            agent.set_commands(commands[cmd_offset:cmd_offset+agent.ctrl_dofs])
            cmd_offset = cmd_offset+agent.ctrl_dofs
    @property 
    def gravity(self):
        return self._rigid_data.gravity
    @property
    def agent_num(self):
        return len(self._agents)
    @property
    def dt(self):
        return self._rigid_world.getTimestep()
    @property
    def time(self):
        return self._rigid_world.time