from typing import Tuple
import numpy as np
from pyrr import Matrix44, Quaternion, Vector3, vector
class camera:
    def __init__(self,
    z_near=0.1,
    z_far=1000,fov=60,
    center = [0,0,0],
    up = [0,1,0],
    target = [0,0,0],
    window_size:Tuple[float]=(1920,1080)
    ) -> None:
        self.z_near = z_near
        self.z_far = z_far
        self.fov = fov
        self.window_size = window_size
        self.center = center
        self.up = up
        self.target = target
        
    def build_look_at(self):
        self.mat_lookat = Matrix44.look_at(
            self.center,
            self.target,
            self.up)
    def build_projection(self):
        self.mat_projection = Matrix44.perspective_projection(
            self.fov,
            self.window_size[0]/self.window_size[1],
            self.z_near,
            self.z_far)
    @property
    def viewProejction(self):
        self.build_look_at()
        self.build_projection()
        return self.mat_projection * self.mat_lookat

