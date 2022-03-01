# Fish Gym
Fish Gym is a physics-based simulation framework for physical articulated underwater agent interaction with fluid.
This is the first physics-based environment that support coupled interation between agents and fluid in semi-realtime.
Fish Gym is integrated into the OpenAI Gym interface, enabling the use of existing reinforcement learning and control algorithms to control underwater agents to accomplish specific underwater exploration task.

https://user-images.githubusercontent.com/20988615/156165618-d0d034d8-2e48-4f2f-bc63-514da431b918.mp4

[Documentation](https://gym-fish.readthedocs.io/) 
[Docker](https://gym-fish.readthedocs.io/en/latest/User%20Guide/Installation.html#using-docker-imanges)

# Installation
## Use Docker Images
https://gym-fish.readthedocs.io/en/latest/User%20Guide/Installation.html#using-docker-imanges

## Maunal Installation

The framework depends on [DART](https://github.com/dartsim/dart) and [CUDA](https://developer.nvidia.com/cuda-toolkit). The gym is based on python 3.6.9.

### 1. DART Installation

It is recommended to install DART using Ubuntu packages with Personal Package Archives
```
sudo apt-add-repository ppa:dartsim/ppa
sudo apt-get update # not necessary since Bionic
sudo apt-get install libdart6-dev
```
Please refer to [Official DART Installation Guide](http://dartsim.github.io/install_dart_on_ubuntu.html) for detailed information.

### 2. CUDA Installation
Cuda 10.2 is recommended to run the framework, which can be downloaded from the [NVIDIA website](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)

Please also remember to [disable nouveau](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile-nouveau) if you want to also install the driver. Refer to [Official CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) more detailed information

### 3. Conda Installation
Install either [Anaconda](https://www.anaconda.com/products/individual#Anaconda%20Installers) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or use your favorite python virtual enviroment. The framework is based on python 3.6.9.

### 4. Python Environment Setup
```shell
conda create -n gym_fish python=3.6.9
conda activate gym_fish
conda install -c conda-forge moderngl
```

### 5. Install Fish Gym
```shell
git clone https://github.com/fish-gym/gym-fish.git
cd gym-fish
pip3 install -e .
```


###6. (for headless machines) Virtual Display
If you are using a headless machine (server/cluster). Please either connect to a remote display or running a local virtual display. Otherwise you might meet an *XOpenDisplay* error when you render.

## Getting Started
### Run the environment
Now that we will try to run a basic task environment: point to point navigation.
```shell
conda activate gym_fish
python3
```
```python
import gym
import gym_fish

# Our environment runs on GPU to accelerate simulations,ensure a cuda-supported GPU exists on your machine
gpuId = 0
env = gym.make('koi-cruising-v0',gpuId =gpuId)
action = env.action_space.sample()
obs,reward,done,info = env.step(action)
```

### Render the scene
Then we can see the scene in two modes : `human` and `rgbarray`.
`human` is suitable for machines that with a display.
`rgbarray` is for headless machines.

#### For machines with a display
```python
env.render_mode='human'
env.render()
```

#### For headless machines(server/cluster)
Run a virtual display. for more details, check [here](https://moderngl.readthedocs.io/en/latest/techniques/headless_ubuntu_18_server.html)

Run follwing commands in your shell:
```shell
export DISPLAY=:99.0
Xvfb :99 -screen 0 640x480x24 &
```

Render and outputs a numpy array
```
python
# This outputs a numpy array which can be saved as image
arr = env.render()
# Save use Pillow
from PIL import Image
image = Image.fromarray(arr)
# Then a scene output image can be viewed in
image.save('output.png')
```

## Run trained policies

### Koi Cruising on a plane

Here we evaluate a pretrained policy for a robot koi who tries to swim to reach a given target location that is a distant away from the robot.

In this case, the target is located at the forward direction of the fish.

```shell
 python3 enjoy.py --env koi-cruising-v0 --gpu-id 0
```

The result will be saved into the folder ``result\videos\koi-cruising-v0_final.mp4``

## Train your own policy

### Koi Pose control

Here we train a policy for a robot koi who tries control its pose in order to make a U-turn, i.e. turning back to tail.

```python
 python3 train.py --env fish-pose-control-v0 --gpu-id 0 --n-timesteps 50000 --eval-freq 2000 --eval-episodes 1
```

The trained model will be saved into the folder ``models`` and the result video will be saved into the folder ``videos``

Similarly, the ``fish-pose-control-v0`` argument can be replaced environment id mentioned in section ``Enviroments``.
