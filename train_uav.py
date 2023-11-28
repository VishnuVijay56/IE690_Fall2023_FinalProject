import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import sys
sys.path.append('../IE690_Fall2023_FinalProject')

from uav_gym_env import UAVStallEnv
from sim_cmds import SimCmds

# multiprocess environment
n_cpu = 4
# env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])
# env = gym.make('CartPole-v1', render_mode="rgb_array")
# model.save("ppo2_cartpole")
# del model # remove to demonstrate saving and loading
# model = PPO.load("ppo2_cartpole")

# Simulation Parameters
sim_opt = SimCmds()
sim_opt.view_sim = False
sim_opt.sim_real_time = False
sim_opt.display_graphs = False
sim_opt.use_kf = False
sim_opt.wind_gust = False

env = UAVStallEnv(sim_opt)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1_000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")