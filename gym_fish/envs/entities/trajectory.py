
from typing import Any, List
from gym_fish.envs.py_util import flare_util
import numpy as np
class pose:
    def __init__(self,position:np.array = np.array([0,0,0]),orientation:np.array = np.array([0,0,0]) ) -> None:
        self.position = position
        # specific representation to be determined
        self.orientation = orientation

class trajectory:
    def __init__(self,points:List[List[float]]) -> None:
        self._traj = fl.make_trajectory()
        self._points = points
        _ps = []
        for p in self._points:
            point = fl.make_tpPoint()
            point.data = p
            _ps.append(point)
        self.trajectory.setPoints(_ps)
        self.trajectory.fit()
        self.trajectory.sample(300)
    def __init__(self,path_data:flare_util.path_data) -> None:
        self._traj = path_data.trajectory
        self._points = [p.data for p in path_data.points]
    @property
    def points(self):
        return self._points
    def get_dist(self,x:float,y:float,z:float)->float:
        t  = self.parameterize(x,y,z)
        p_on_path = self._traj.getPose(t)
        pos = p_on_path.getPosition()
        return np.linalg.norm(pos-np.array([x,y,z]))
    def get_pose(self,t:float)->pose:
        ori = np.array(self._traj.getPose(t+0.03).getPosition())-np.array(self._traj.getPose(t-0.03).getPosition())
        ori = ori/(np.linalg.norm(ori)+1e-8)
        return pose(np.array(self._traj.getPose(t).getPosition()),ori)
    def get_ref_pose(self,x:float,y:float,z:float) -> pose:
        t  = self.parameterize(x,y,z)
        return self.get_pose(t)
    def get_curvature(self,x:float,y:float,z:float)->np.array:
        return self._traj.getCurvature(np.array([x,y,z]))

    def parameterize(self,x:float,y:float,z:float) -> float:
        return self._traj.getReferencePose(np.array([x,y,z]))

    