import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from uav_gym_env import UAVStallEnv
from sim_cmds import SimCmds

# Simulation Parameters
sim_opt = SimCmds()
sim_opt.view_sim = False
sim_opt.sim_real_time = False
sim_opt.display_graphs = False
sim_opt.use_kf = False
sim_opt.wind_gust = False

# Instantiate Environment
env = UAVStallEnv(sim_opt)

# Train Agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# Evaluate Trained Agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1_000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

print(vec_env.evaluate_model())