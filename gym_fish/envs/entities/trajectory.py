
from typing import Any, List
import numpy as np
class pose:
    def __init__(self,position:np.array = np.array([0,0,0]),orientation:np.array = np.array([0,0,0]) ) -> None:
        self.position = position
        # specific representation to be determined
        self.orientation = orientation

class trajectory:
    def __init__(self,config_file:str) -> None:
        self._traj = fl.LoadTrajectoryFromJsonFile(config_file)
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

    