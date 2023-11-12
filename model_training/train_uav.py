"""
This code is a general framework for training a model using PPO, currently it trains using the MLPpolicy on cartpole-v1

Errors I have:
1. Cannot use SubProcEnv to simulate the multiple environments at a time, these lines are commented out.
2. The trained agent does not render for me

Future Work:
- 
- etc.
"""

import gymnasium as gym

# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# multiprocess environment
# n_cpu = 4
# env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])

model = PPO("MlpPolicy", "CartPole-v1", verbose=1)
env = model.get_env()
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

# Enjoy trained agent
obs = env.reset()

# check_env(env, warn=True, skip_render_check=False)
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()