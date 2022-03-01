import gym
import gym_fish

# Our environment runs on GPU to accelerate simulations, ensure a cuda-supported GPU exists on your machine
gpuId = 0
env = gym.make('koi-cruising-v0', gpuId=gpuId)
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
env.render_mode = "rgb_array"
# This outputs a numpy array which can be saved as image
arr = env.render()

# Save use Pillow
from PIL import Image
image = Image.fromarray(arr)

# Then a scene output image can be viewed in 
image.save('output.png')

print("\nEnv Test Passed.")