
from gym_fish.envs.py_util import flare_util
from gym_fish.envs.lib import pyflare as fl
from .rigid_solver import rigid_solver
import numpy as np
import os
import math

class fluid_solver:
    def __init__(self,fluid_param:flare_util.fluid_param,gpuId:int=0,couple_mode:fl.COUPLE_MODE =fl.COUPLE_MODE.TWO_WAY ) -> None:
        self._simulator = fl.make_simulator(fluid_param.data,gpuId)
        self.couple_mode = couple_mode
        self.ok = False
    def attach(self,_rigid_solver:rigid_solver):
        self._simulator.attachWorld(_rigid_solver._rigid_world)
        self._simulator.commitInit()
        # self._simulator.log()
        self.ok = True
    @property
    def iter_count(self):
        return self._simulator.getIterNum()
    @property
    def iters_at_framerate(self,framerate:int=30):
        return self._simulator.getIterPerSave(framerate)
    def iter(self):
        if self.ok==False:
            print("fluid solver is not ok to run")
            return
        self._simulator.step(self.couple_mode)
    def set_savefolder(self,folder_path:str='./data'):
        self._simulator.mainDataFolderPath = str(folder_path)
        self.dataPath = {}
        self.dataPath["fluid"] = str(self._simulator.mainDataFolderPath + self._simulator.fluidFolderName + '/')
        self.dataPath["objects"] = str(self._simulator.mainDataFolderPath + self._simulator.objectsFolderName + '/')
        self.dataPath["trajectory"] = str(self._simulator.mainDataFolderPath + 'Trajectory/')
        if not os.path.exists(self._simulator.mainDataFolderPath):
            os.makedirs(self._simulator.mainDataFolderPath)
        for p in self.dataPath.values():
            if not os.path.exists(p):
                os.makedirs(p)
    def save(self,save_fluid=False, save_objects=True,suffix:str="0000"):
        if save_fluid:
            fluid_name = "fluid"+suffix
            self._simulator.saveFluidData(fluid_name)
        if save_objects:
            objects_name = "object"+suffix
            self._simulator.saveObjectsData(objects_name)


    

