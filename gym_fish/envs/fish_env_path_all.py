import numpy as np
from .lib import pyflare as fl
from .fish_env_path_basic import FishEnvPathBasic

class KoiPathEnv(FishEnvPathBasic):
    def __init__(self, 
                control_dt=0.2,
                 wc = np.array([1.0,0.5]),
                 wp = np.array([0.0,1.0]),
                 wa = 0.5,
                max_time = 10,
                done_dist=0.1,
                radius = 1,
                 theta = np.array([-180,180]),
                 phi = np.array([0,0]),
                dist_distri_param =np.array([0,0.5]),
                data_folder = "",
                env_json :str = '../assets/env_file/env_path_koi.json',
                gpuId: int=0,
                couple_mode: fl.COUPLE_MODE = fl.COUPLE_MODE.TWO_WAY,
                empirical_force_amplifier =1600,
                 use_com=True,
                 no_closeness_obs = False,
                ) -> None:
        super().__init__(control_dt=control_dt,wc = wc,wp = wp,wa = wa,max_time = max_time,done_dist=done_dist,radius = radius,theta = theta,phi = phi,dist_distri_param =dist_distri_param,data_folder = data_folder,env_json = env_json,gpuId=gpuId,couple_mode=couple_mode,empirical_force_amplifier =empirical_force_amplifier,use_com=use_com,no_closeness_obs =no_closeness_obs)

class FlatfishPathEnv(FishEnvPathBasic):
    def __init__(self, 
                control_dt=0.2,
                 wc = np.array([1.0,0.5]),
                 wp = np.array([0.0,1.0]),
                 wa = 0.5*4/5,
                max_time = 10,
                done_dist=0.1,
                radius = 1,
                 theta = np.array([-180,180]),
                 phi = np.array([0,0]),
                dist_distri_param =np.array([0,0.5]),
                data_folder = "",
                env_json :str = '../assets/env_file/env_path_flatfish.json',
                gpuId: int=0,
                couple_mode: fl.COUPLE_MODE = fl.COUPLE_MODE.TWO_WAY,
                empirical_force_amplifier =1600,
                 use_com=True,
                 no_closeness_obs = False,
                ) -> None:
        super().__init__(control_dt=control_dt,wc = wc,wp = wp,wa = wa,max_time = max_time,done_dist=done_dist,radius = radius,theta = theta,phi = phi,dist_distri_param =dist_distri_param,data_folder = data_folder,env_json = env_json,gpuId=gpuId,couple_mode=couple_mode,empirical_force_amplifier =empirical_force_amplifier,use_com=use_com,no_closeness_obs =no_closeness_obs)
        
class EelPathEnv(FishEnvPathBasic):
    def __init__(self, 
                control_dt=0.2,
                 wc = np.array([1.0,0.5]),
                 wp = np.array([0.0,1.0]),
                 wa = 0.5*4/7,
                max_time = 10,
                done_dist=0.1,
                radius = 1,
                 theta = np.array([0,360]),
                 phi = np.array([0,0]),
                dist_distri_param =np.array([0,0.0]),
                data_folder = "",
                env_json :str = '../assets/env_file/env_path_eel.json',
                gpuId: int=0,
                couple_mode: fl.COUPLE_MODE = fl.COUPLE_MODE.TWO_WAY,
                empirical_force_amplifier =1600,
                 use_com=False,
                 no_closeness_obs = True,
                ) -> None:
        super().__init__(control_dt=control_dt,wc = wc,wp = wp,wa = wa,max_time = max_time,done_dist=done_dist,radius = radius,theta = theta,phi = phi,dist_distri_param =dist_distri_param,data_folder = data_folder,env_json = env_json,gpuId=gpuId,couple_mode=couple_mode,empirical_force_amplifier =empirical_force_amplifier,use_com=use_com,no_closeness_obs =no_closeness_obs)