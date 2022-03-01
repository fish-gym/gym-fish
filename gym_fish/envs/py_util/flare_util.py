from typing import List
from ..lib import pyflare as fl
from . import json_util
from . import trajectory_util
import os
#############################################################################################
################################ START  WRITE JSON WRAPPER #################################
#############################################################################################
class fluid_param(json_util.json_support):
    normal_attrs = ['x0', 'y0', 'z0', 'width', 'height', 'depth', 'N', 'l0p', 'u0k', 'u0p', 'rou0p', 'visp','pml_width','slip_ratio']
    enum_attrs = ['setup_mode']

    def __init__(self):
        super().__init__()
        self.data = fl.make_simParams()

    def from_dict(self, d: dict,filefolder:str=""):
        for attr in self.normal_attrs :
            if hasattr(self.data, attr) and (attr in d.keys()):
                setattr(self.data, attr, d[attr])
        if hasattr(self.data,'setup_mode')and ('setup_mode' in d.keys()):
            self.data.setup_mode =fl.SETUP_MODE(d['setup_mode'])

    def to_dict(self):
        d = {}
        for attr in self.normal_attrs:
            if hasattr(self.data, attr):
                d[attr] = getattr(self.data, attr)
        for attr in self.enum_attrs:
            if hasattr(self.data, attr):
                d[attr] = int(getattr(self.data, attr))
        return d


class path_param(json_util.json_support):
    def __init__(self, source_file: str=None):
        super().__init__()
        self.source_file = source_file
        if source_file!=None:
            self.points = trajectory_util.trajectoryPoints_file(self.source_file)
            self.path_sample_num=100
        else:
            self.points=[]
            self.path_sample_num=0


    def setPoints(self, points: List[fl.trajectoryPoint3d]):
        self.source_file = None
        self.points = points

    def to_dict(self) -> dict:
        d = {'path_sample_num': self.path_sample_num}
        if self.source_file != None:
            d['source_file'] = self.source_file
        else:
            d['points'] = [[x.data[0], x.data[1], x.data[2]] for x in self.points]
        return d

    def from_dict(self, d: dict,filefolder:str=""):
        if 'path_sample_num' in d.keys():
            self.path_sample_num = d['path_sample_num']
        if 'source_file' in d.keys():
            path_skeletonFile = os.path.abspath(os.path.join(filefolder,d['source_file']))
            self.source_file = str(path_skeletonFile)
            self.points = trajectory_util.trajectoryPoints_file(self.source_file)
        else:
            for p in d['points']:
                point = fl.make_tpPoint()
                point.data = p
                self.points.append(point)


class path_data(json_util.json_support):

    def __init__(self, path_setting: path_param=None):
        super().__init__()
        self.trajectory = fl.make_trajectory()
        self.path_setting = path_setting
        if self.path_setting!=None:
            self.setPoints(self.path_setting.points, self.path_setting.path_sample_num)

    def setPoints(self, points, sample_num):
        self.points = points
        self.trajectory.setPoints(self.points)
        self.trajectory.fit()
        self.trajectory.sample(sample_num)

    def to_dict(self) -> dict:
        return self.path_setting.to_dict()

    def from_dict(self, d: dict,filefolder:str=""):
        self.path_setting = path_param()
        self.path_setting.from_dict(d)
        self.setPoints(self.path_setting.points, self.path_setting.path_sample_num)


class skeleton_param(json_util.json_support):
    def __init__(self, skeleton_file: str="", sample_num: int=5000,density:float=1028, offset_pos: List[float] = [0, 0, 0],
                 offset_rotation: List[float] = [0, 0, 0]):
        super().__init__()
        self.skeleton_file =skeleton_file
        self.sample_num = sample_num

        self.controllable = True
        self.has_buoyancy = True
        self.density = density
        self.bladder_volume_min = 0
        self.bladder_volume_max= 1
        self.bladder_volume_control_min = 0
        self.bladder_volume_control_max = 0.1
        self.offset_pos = offset_pos
        self.offset_rotation = offset_rotation
        self.offset_scale = [1, 1, 1]
        self.cameras = []

    def to_dict(self) -> dict:	
        return self.__dict__

    def from_dict(self, d: dict,filefolder:str=""):
        self.__dict__ = d
        skeleton_file_path = os.path.abspath(os.path.join(filefolder,self.skeleton_file))
        self.skeleton_file = skeleton_file_path
        # not allow scaling though this interface exists
        self.offset_scale = [1, 1, 1]
        if not hasattr(self, 'cameras'):
            self.cameras = []
            
        


class skeleton_data(json_util.json_support):

    def __init__(self, param: skeleton_param=None,gpuId:int = 0):
        super().__init__()
        self.param = param
        self.skeleton=None
        self.dynamics =None


        self.gpuId = gpuId
        if param!=None:
            self.init_from_setting()
    def init_from_setting(self):
        self.skeleton = fl.skeletonFromJson(self.param.skeleton_file,self.gpuId)
        self.dynamics = fl.make_skDynamics(self.skeleton,
                                            self.param.sample_num,
                                            self.gpuId,
                                           self.param.offset_pos,
                                            self.param.offset_rotation,
                                            self.param.offset_scale
                                            )

    def to_dict(self) ->dict:
        if self.param!=None:
            return self.param.to_dict()
        else:
            return {}
    def from_dict(self,d:dict,filefolder:str=""):
        self.param = skeleton_param()
        self.param.from_dict(d,filefolder)
        self.init_from_setting()



class rigid_data(json_util.json_support):
    def __init__(self, gravity=None, skeletons=None,gpuId:int=0):
        super().__init__()
        self.gpuId = gpuId
        if skeletons is None:
            skeletons = []
        self.skeletons = skeletons
        if gravity is None:
            gravity = [0, 0, 0]
        self.gravity = gravity
        self.rigidWorld = fl.make_skWorld(self.gravity)
        for skeleton in self.skeletons:
            self.rigidWorld.addSkeleton(skeleton.dynamics)
    def to_dict(self) ->dict:
        d = {}
        d['skeletons'] = [self.skeletons[i].to_dict() for i in range(len(self.skeletons))]
        d['gravity'] = self.gravity
        return d
    def from_dict(self,d:dict,filefolder:str=""):
        if  'gravity' in d.keys():
            self.gravity = d['gravity']
        else:
            self.gravity=[0,0,0]
        self.rigidWorld.reset()
        self.rigidWorld.setGravity(self.gravity)
        if 'skeletons' not in d.keys():
            return
        self.skeletons.clear()
        for skeleton_dict in d['skeletons']:
            sk  = skeleton_data(None,self.gpuId)
            sk.from_dict(skeleton_dict,filefolder)
            self.rigidWorld.addSkeleton(sk.dynamics)
            self.skeletons.append(sk)



