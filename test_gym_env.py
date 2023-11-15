from uav_gym_env import UAVStallEnv
from stable_baselines3.common.env_checker import check_env
from sim_cmds import SimCmds

# Simulation Parameters
sim_opt = SimCmds()
sim_opt.view_sim = False
sim_opt.sim_real_time = False
sim_opt.display_graphs = False
sim_opt.use_kf = False
sim_opt.wind_gust = False

env = UAVStallEnv(sim_opt)

check_env(env, warn=True, skip_render_check=True)