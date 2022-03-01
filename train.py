import argparse
import difflib
import os
import time
import gym
import gym_fish
from stable_baselines3 import SAC
def enjoy_path_env(env,model):
    trajectory_file  = os.path.dirname(gym_fish.__file__)+ '/assets/trajectory/path_69.json'
    from gym_fish.envs.py_util import flare_util
    from gym_fish.envs.entities.trajectory import trajectory
    ### test generalization to sequence goals
    path_data = flare_util.path_data()
    path_data.from_json(trajectory_file)
    path_traj = trajectory(path_data)
    env.reset()
    t = 0.001
    dt = 0.05
    env.goal_pos = path_traj.get_pose(t+dt).position
    env.path_dir = (env.goal_pos-env.body_xyz)/np.linalg.norm((env.goal_pos-env.body_xyz))
    env.max_time = 30
    while path_traj.parameterize(env.body_xyz[0],env.body_xyz[1],env.body_xyz[2])<0.98:
        obs = env._get_obs()
        action,_ = model.predict(obs, deterministic=True)
        env.step(action)
        env.render()
        # set next target, if close to current target 
        if np.linalg.norm(env.body_xyz-env.goal_pos)<0.25:
            t = t+dt        
            env.goal_pos = path_traj.get_pose(t+dt).position
            env.path_dir = (env.goal_pos-env.body_xyz)/np.linalg.norm((env.goal_pos-env.body_xyz))
            
def enjoy_other_env(env,model):
    obs = env.reset()
    done =  False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render() 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="koi-cruising-v0", help="environment ID")
    parser.add_argument("--gpu-id", help="Override Default GPU device", default=0, type=int)
    parser.add_argument("-n", "--n-timesteps", help="Overwrite the number of timesteps", default=50000, type=int)
    parser.add_argument("-save-folder", "--save-folder", help="Data save dir (for network models and render data)", default="", type=str)
    parser.add_argument("-tb", "--tensorboard-log", help="Tensorboard log dir", default="", type=str)
    parser.add_argument("--log-interval", help="Override log interval (default: -1, no change)", default=-1, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--eval-freq",
        help="Evaluate the agent every n steps (if negative, no evaluation). ",
        default=2000,
        type=int,
    )
    parser.add_argument("--eval-episodes", help="Number of episodes to use for evaluation", default=1, type=int)

    args = parser.parse_args()
    env_id = args.env
    registered_envs = set(gym.envs.registry.env_specs.keys())  # pytype: disable=module-attr

    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(f"{env_id} not found in gym registry, you maybe meant {closest_match}?")
    # build up data folder
    
    experiment_folder_name = '/'+env_id+'_train_'+time.strftime('%Y-%m-%d %H:%M/',time.localtime(time.time()))+'/'
    save_folder = args.save_folder
    if save_folder=="":
        save_folder = os.getcwd()+"/result/"
    if save_folder.startswith("/"):
        save_folder = os.path.abspath(save_folder)
    else:
        save_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), save_folder))
    
    
    tb_folder = args.tensorboard_log
    if tb_folder=="":
        tb_folder = os.getcwd()+"/result/"
    if save_folder.startswith("/"):
        tb_folder = os.path.abspath(save_folder)
    else:
        tb_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), tb_folder))
        
    save_folder= save_folder+experiment_folder_name
    tb_folder = tb_folder+experiment_folder_name
    network_folder = save_folder+"/models/"
    data_folder= save_folder+"/render_data/"
    video_folder= save_folder+"/videos/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    print("Creating a environment {} with gpuId {}....".format(env_id,args.gpu_id))
    env = gym.make(env_id,gpuId=args.gpu_id,data_folder = data_folder)
    print("done!")
    print("Creating a new policy for environment {} ....".format(env_id))
    model = SAC("MlpPolicy", env, verbose=args.verbose,tensorboard_log=tb_folder)
    print("done!")
    print("Start Learning...")
    # start learning
    if args.eval_freq>0:
        # create env for evaluation
        env_eval = gym.make(env_id,gpuId=args.gpu_id,data_folder = data_folder)
        cur_steps = 0
        while cur_steps<args.n_timesteps:
            learn_steps = min(args.eval_freq,args.n_timesteps-cur_steps)
            model.learn(total_timesteps=learn_steps, log_interval=args.log_interval,reset_num_timesteps=False)
            cur_steps = cur_steps+learn_steps
            
            model.save(network_folder+env_id+"_",str(cur_steps))
            # start evaluate
            eval_episode = 0
            eval_reward = 0
            while eval_episode<args.eval_episodes:
                obs = env_eval.reset()
                done =  False
                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env_eval.step(action)
                    eval_reward+=reward
                eval_episode = eval_episode+1
            env_eval.export_video(video_folder+env_id+"_eval_"+str(cur_steps))
            print("Evalute for {} episodes, mean reward is {}".format(eval_episode,eval_reward/eval_episode))
    else:
        model.learn(total_timesteps=args.n_timesteps, log_interval=args.log_interval)
    # save model
    model_path = network_folder+env_id+"train"
    model.save(model_path)
    print("Learning process completed, the trained model is saved to ".format(model_path))
    # enjoy with the learned model
    
    print("Running the env with trained model...")
    if 'path' in env_id:
        enjoy_path_env(env,model)
    else:
        enjoy_other_env(env,model)
    env.export_video(video_folder+env_id+"_train")
    print("The result video is saved to {}".format(video_folder+env_id+"_train"))
