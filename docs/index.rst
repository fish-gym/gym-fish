Fish Gym
=====================================

Fish Gym is a physics-based simulation framework for physical articulated underwater agent 
interaction with fluid. This is the first physics-based environment that support coupled 
interation between agents and fluid in semi-realtime. Fish Gym is integrated into the OpenAI Gym 
interface, enabling the use of existing reinforcement learning and control algorithms to control 
underwater agents to accomplish specific underwater exploration task.



.. warning:: Note this environment is now tested on Ubuntu18.04 and later, not tested on Mac, definitely not work on Windows due to dependency issues.

Github repository: https://github.com/dongfangliu/gym-fish

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   User Guide/Installation
   User Guide/Environments
   User Guide/Getting Started
   User Guide/Run Pretrained Policies
   User Guide/Train New Policies
   User Guide/Create a New Fish Enviroment


.. toctree::
   :maxdepth: 2
   :caption: API

   API/agent_basics
   API/underwater_agents
   API/envs