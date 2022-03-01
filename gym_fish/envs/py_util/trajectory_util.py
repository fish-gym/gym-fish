import math
import numpy as np
from ..lib import pyflare as fl
from pathlib import Path
def trajectoryPoints_circle(center,radiusx,radiusz,point_num=100,angle=360,inverse = False):

    angleStep = math.radians(angle)/point_num
    points = []
    for i in range(point_num):
        p = fl.make_tpPoint()
        if inverse == False:
            p.data = [radiusx * math.cos(angleStep * i + math.pi) + center[0], center[1],
                      radiusz * math.sin(angleStep * i + math.pi) + center[2]]
        else:
            p.data = [radiusx * math.cos(-angleStep * i + math.pi) + center[0], center[1],
                      radiusz * math.sin(-angleStep * i + math.pi) + center[2]]
        points.append(p)
    return points
def trajectoryPoints_line(start,end,point_num=100):
    
    points = []
    step = (end-start)/point_num
    for i in range(point_num):
        p = fl.make_tpPoint()
        p.data = start+i*step
        points.append(p)
    return points
    
def trajectoryPoints_file(file_path):
    points = []
    print(Path(file_path),Path(file_path).exists())
    with Path(file_path).open() as f:
        for line in f.readlines():
            p  =fl.make_tpPoint()
            p.data =eval(line)
            points.append(p)
    return points


# trajectory = fl.poseTrajectory()
# points = trajectoryPoints_circle([2, 2, 3], 1, 1.2, 300, angle=180,inverse=True)
# trajectory.setPoints(points)
# trajectory.fit()
# trajectory.sample(100)
#
# fl.VTKWriter.writeTrajectory(trajectory,"../data/vis_data/Trajectory/trajectory_test.vtk")
# pass
