Train New Policies
=====================
Training a new control policy can be accomplished using any control learning library that works with the OpenAI Gym interface.

Fish Gym comes with built-in functions for easily training and evaluating reinforcement learning (RL) policies using stable-baselines3.


Koi Pose Control 
-------------------------------
Here we train a policy for a robot koi who tries control its pose in order to make a U-turn, i.e. turning back to tail.

.. code:: shell
    
    python3 train.py --env fish-pose-control-v0 --gpu-id 0 --n-timesteps 50000 --eval-freq 2000 --eval-episodes 1


The trained model will be saved into the folder ``models`` and the result video will be saved into the folder ``videos``

Similarly, the ``fish-pose-control-v0`` argument can be replaced environment id mentioned in section ``Enviroments``.
