Create a New Fish Enviroment
=============================

Intall in editable mode
-----------------------
When creating new environments, it is helpful to install Fish Gym in editable mode.

.. code-block:: shell
    
    cd gym-fish
    pip install -e .

Template
--------
Here is the basic template for creating a new environment in Fish Gym.

Create environment python scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We would create a new environment file: ``gym-fish/gym_fish/envs/new_task.py``

.. code-block:: python

    from .coupled_env import coupled_env
    from .lib import pyflare as fl 
    import numpy as np
    import os
    class FishEnvNewTask(coupled_env):
        def __init__(self, 
                    env_json :str = '../assets/env_file/env_newtask.json',
                    gpuId: int=0,) -> None:
            # use parent's init function to init default env data, like action space and observation space, also init dynamics
            super().__init__("",env_json, gpuId)

            def _step(self, action)->None:
                pass
            def _get_obs(self)->np.array:
                self._update_state()
                pass
            def _get_reward(self,cur_obs,cur_action):
                pass
            def _get_done(self)->bool:
                pass
            def reset(self) ->np.array:
                super().reset()
                pass
            def _update_state(self)->None:
                pass

Please note the ``env_json`` (i.e. ``../assets/env_file/env_newtask.json``) does not actually exist,  we will craft it later.

``_step()`` is called each time the robot executes a new action.

``_get_obs()`` is called when we want the current observations available to the robot.

``_get_reward()`` is called when we want the current reward and debug info of the robot.

``_get_done()`` is called to judge the terminal condition.

``reset()`` resets the environment and is called prior to the beginning of each simulation rollout.


We will then add the following line to the ``gym-fish/gym_fish/envs/__init__.py``

.. code-block:: python

    from gym_fish.envs.new_task import FishEnvNewTask

And the following lines to ``gym-fish/gym_fish/__init__.py``

.. code-block:: python

    register(
        id='fish-newtask-v0',
        entry_point='gym_fish.envs:FishEnvNewTask',
    )


Create configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We would create a new environment file: ``gym-fish/gym_fish/assets/env_file/env_newtask.json``

.. code-block:: json

    {
        "rigid_json": "../rigid/rigids_newtask.json",
        "fluid_json": "../fluid/fluid_newtask.json",
        "camera":{
            "window_size":[800,600],
            "z_near":0.1,
            "z_far":1000,
            "fov":60,
            "center":[0.5,1,2.0],
            "target":[0.5,0,0],
            "up":[0,1,0]
        }
    }

For simulation, we specify the fluid config file  and the rigid config file  for coupled simulation.

For scene rendering, we specify the camera setting.

Then we create a new fluid configuration file : ``gym-fish/gym_fish/assets/fluid/fluid_newtask.json``

.. code-block:: json

    {
        "x0": 0.0,
        "y0": 0.0,
        "z0": 0.0,
        "width": 2.0,
        "height": 2.0,
        "depth": 2.0,
        "N": 50.0,
        "l0p": 1.0,
        "u0k": 0.1,
        "u0p": 0.5,
        "rou0p": 1000,
        "visp": 0.0001,
        "slip_ratio":1.0,
        "setup_mode": 0,
        "pml_width":0
    }

``(x0,y0,z0)`` and ``(width,height,depth)`` tells the center and size of the fluid region, in meters.

It's recommended to keep other parameters as default. 

To increase the simulation resolution of fluid region (not the size), you can increase ``N`` or decrease ``l0p``, as ``dx = l0p/N``.

To step the world with a finer timestep, you can increase ``u0p``, which is propotional to ``dt``.

``rou0p`` and ``visp`` are density and viscosity for the fluid, respectively.



Then we create a new fluid configuration file : ``gym-fish/gym_fish/assets/rigid/rigids_newtask.json``


.. code-block:: json

    {
        "skeletons": [
            {
                "skeleton_file": "../agents/koi_no_fin.json",
                "controllable": true,
                "sample_num": 10000,
                "has_buoyancy": false,
                "density":1080,
                "bladder_volume_min":0,
                "bladder_volume_max":2,
                "bladder_volume_control_min":-0.01,
                "bladder_volume_control_max":0.01,
                "offset_pos": [
                    0,
                    0,
                    0
                ],
                "offset_rotation": [
                    0,
                    0,
                    0
                ]
            }
        ],
        "gravity": [
            0,
            0,
            0
        ]
    }


We recommend to just reuse files like ``rigids_koi_no_fin.json`` for koi in a 2d plane and ``rigids_koi_no_fin_3d_buoyancy.json`` for task in 3d.
When considering multiple agents, ``rigids_school`` might be a good reference.




