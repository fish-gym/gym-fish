from gym_fish.envs.lib import pyflare as fl

class coupled_sim:
    def __init__(self,config_file:str) -> None:
        self.solver = fl.GetSolverFromJsonFilePath(config_file)
        self.solid_solver=  self.solver.GetSolidSolver()
        self.iter_count=0
    @property
    def dt(self):
        return self.solver.GetTimeStep()
    @property
    def time(self):
        return self.solver.GetTime()
    @property
    def iter_count(self):
        return self.iter_count
    @property
    def iters_at_framerate(self,framerate:int=30):
        return self.solver.IterNumForFrameRate(framerate)
    def save(self,suffix:str="0000",save_solid:bool=False,save_fluid:bool=False,save_coupled_data:bool =False):
        self.solver.Save(suffix,save_coupled_data,save_fluid,save_solid)
    def step(self):
        self.solver.Step()

    