Run Pretrained Policies
=============================

We provide pretrained control policies for each task and robot.

These controllers are trained using Soft-Actor-Critic (SAC) implemented in PyTorch by stable-baselines3.

We train each policy for a total of 2000 simulation rollouts, each of which contains 50 time steps. All policies are trained on a machine with
an NVidia TitanXP GPU, and the training process usually starts to converge after 6 hours (1000 episodes), and have a
smooth convergence within 10 hours.

Koi Cruising on a plane
-------------------------------
Here we evaluate a pretrained policy for a robot koi who tries to swim to reach a given target location that is a distant away from the robot.
In this case, the target is located at the forward direction of the fish.

.. code:: shell
    
    python3 enjoy.py --env koi-cruising-v0 --gpu-id 0

The result will be saved into the folder ``result\videos\koi-cruising-v0_final.mp4``

Similarly, the ``koi-cruising-v0`` argument can be replaced environment id mentioned in section ``Enviroments``.

