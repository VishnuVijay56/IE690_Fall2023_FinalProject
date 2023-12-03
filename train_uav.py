import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from uav_gym_env import UAVStallEnv
from sample_states import Sampler
from sim_cmds import SimCmds
from evaluate_env import model_evaluator

# Simulation Parameters
sim_opt = SimCmds()
sim_opt.view_sim = False
sim_opt.sim_real_time = False
sim_opt.display_graphs = False
sim_opt.use_kf = False
sim_opt.wind_gust = False

# Sampler
sampler = Sampler()

# Instantiate Environment
env = UAVStallEnv(sim_opt, sampler)

# Train Agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# Evaluate Trained Agent
vec_env = model.get_env()
obs, info = vec_env.reset()
target = info

# Initialize Evaluator
evaluator = model_evaluator(obs, target)

for i in range(1_000):
    print(i)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    evaluator.update(action, obs)

something = evaluator.evaluate()

print(env.evaluate_model())