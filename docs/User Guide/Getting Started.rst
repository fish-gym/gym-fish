Getting Started
===============

Run the environment
-------------------

Now that we will try to run a basic task environment: koi fish cruising.

.. code:: shell

    conda activate gym_fish
    python3

.. code:: python

    import gym
    import gym_fish
    # Our environment runs on GPU to accelerate simulations,ensure a cuda-supported GPU exists on your machine
    gpuId = 0
    env = gym.make('koi-cruising-v0',gpuId =gpuId)
    action = env.action_space.sample()
    obs,reward,done,info = env.step(action)

Render the scene
----------------

Then we can see the scene in two modes : ``human`` , ``rgb_array`` and  ``depth_array``.
``human`` is suitable for machines that with a display. ``rgb_array`` stores captured ``RGBA`` image as numpy array.
``rgb_array`` stores captured depth image as  numpy array.

For machines with a display
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    env.render_mode='human'
    # Output a Image
    image  = env.render()
    # visualize it 
    image.show()

For headless machines (server/cluster)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run a virtual display. for more details,check
`here <https://moderngl.readthedocs.io/en/latest/techniques/headless_ubuntu_18_server.html>`__
Run follwing commands in your shell:

.. code:: shell

    export DISPLAY=:99.0
    Xvfb :99 -screen 0 640x480x24 &

Render and outputs a numpy array

.. code:: python

    # This outputs a numpy array which can be saved as image
    env.render_mode='rgb_array'
    arr = env.render()
    # Save use Pillow
    from PIL import Image
    image = Image.fromarray(arr)
    # Then a scene output image can be viewed in 
    image.save('output.png')
