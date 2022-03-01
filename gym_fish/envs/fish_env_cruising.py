
import numpy as np
from .lib import pyflare as fl
from .fish_env_basic import FishEnvBasic

class KoiCruisingEnv(FishEnvBasic):
    def __init__(self, 
                control_dt=0.2,
                wp= np.array([0.0,1.0]),
                wr= 0.0,
                wa=0.5,
                max_time = 10,
                done_dist=0.15,
                radius = 2,
                theta = np.array([90,90]),
                phi = np.array([0,0]),
                data_folder = "",
                env_json :str = '../assets/env_file/env_cruising_koi.json',
                gpuId: int=0,
                couple_mode: fl.COUPLE_MODE = fl.COUPLE_MODE.TWO_WAY,
                empirical_force_amplifier =1600,
                 is3D= False,
                 use_com=True
                ) -> None:
        super().__init__(control_dt,wp, wr,wa,max_time,done_dist,radius,theta,phi,data_folder,env_json,gpuId,couple_mode,empirical_force_amplifier,is3D,use_com)

# class KoiFinnedCruisingEnv(FishEnvBasic):
#     def __init__(self, 
#                 control_dt=0.2,
#                 wp= np.array([0.0,1.0]),
#                 wr= 0.0,
#                 wa=0.5,
#                 max_time = 10,
#                 done_dist=0.15,
#                 radius = 2,
#                 theta = np.array([90,90]),
#                 phi = np.array([0,0]),
#                 data_folder = "",
#                 env_json :str = '../assets/env_file/env_cruising_koifinned.json',
#                 gpuId: int=0,
#                 couple_mode: fl.COUPLE_MODE = fl.COUPLE_MODE.TWO_WAY,
#                 empirical_force_amplifier =1600,
#                  use_com=True
#                 ) -> None:
#         super().__init__(control_dt,wp, wr,wa,max_time,done_dist,radius,theta,phi,data_folder,env_json,gpuId,couple_mode,empirical_force_amplifier,use_com)        

        
class FlatfishCruisingEnv(FishEnvBasic):
    def __init__(self, 
                control_dt=0.2,
                wp= np.array([0.0,1.0]),
                wr= 0.0,
                wa=0.5*4/5,
                max_time = 10,
                done_dist=0.15,
                radius = 2,
                theta = np.array([90,90]),
                phi = np.array([0,0]),
                data_folder = "",
                env_json :str = '../assets/env_file/env_cruising_flatfish.json',
                gpuId: int=0,
                couple_mode: fl.COUPLE_MODE = fl.COUPLE_MODE.TWO_WAY,
                empirical_force_amplifier =1600,
                 is3D= False,
                 use_com=True
                ) -> None:
        super().__init__(control_dt,wp, wr,wa,max_time,done_dist,radius,theta,phi,data_folder,env_json,gpuId,couple_mode,empirical_force_amplifier,is3D,use_com)

        

class EelCruisingEnv(FishEnvBasic):
    def __init__(self, 
                control_dt=0.2,
                wp= np.array([0.0,1.0]),
                wr= 0.0,
                wa=0.5*4/7,
                max_time = 10,
                done_dist=0.15,
                radius = 2,
                theta = np.array([90,90]),
                phi = np.array([0,0]),
                data_folder = "",
                env_json :str = '../assets/env_file/env_cruising_eel.json',
                gpuId: int=0,
                couple_mode: fl.COUPLE_MODE = fl.COUPLE_MODE.TWO_WAY,
                empirical_force_amplifier =1600,
                 is3D= False,
                 use_com=True
                ) -> None:
        super().__init__(control_dt,wp, wr,wa,max_time,done_dist,radius,theta,phi,data_folder,env_json,gpuId,couple_mode,empirical_force_amplifier,is3D,use_com)
        
        

class KoiCruising3dEnv(FishEnvBasic):
    def __init__(self, 
                control_dt=0.08,
#                 control_dt=0.2,
                wp= np.array([0.0,1.0]),
                wr= 3.0,
                wa= 0.5,
#                 wa= 0,
                max_time = 10,
                done_dist=0.15,
                radius = 2,
                theta = np.array([90,90]),
                phi = np.array([45,45]),
                data_folder = "",
                env_json :str = '../assets/env_file/env_cruising_koi3d.json',
                gpuId: int=0,
                couple_mode: fl.COUPLE_MODE = fl.COUPLE_MODE.TWO_WAY,
                empirical_force_amplifier =1600,
                 is3D= True,
                 use_com=True
                ) -> None:
        super().__init__(control_dt,wp, wr,wa,max_time,done_dist,radius,theta,phi,data_folder,env_json,gpuId,couple_mode,empirical_force_amplifier,is3D,use_com)