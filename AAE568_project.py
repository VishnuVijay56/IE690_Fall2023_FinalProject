"""
AAE568_project.py: implementing project
    - Author: Vishnu Vijay AND ME
    - Created: 4/23/23
"""

# Imports
from run_sim import run_uav_sim
from sim_cmds import SimCmds 
from saturate_cmds import SaturateCmds


# Simulation Parameters
sim_opt = SimCmds()
sim_opt.view_sim = False
sim_opt.sim_real_time = False
sim_opt.display_graphs = True
sim_opt.use_kf = False
sim_opt.wind_gust = False

# from helper import write_discrete_SS
# write_discrete_SS()

# Time Span
t_span = (0, 40)


# Sim
sum = 0
num_runs = 1
for i in range(num_runs):
    print("----------------------------------")
    print("Simulation number: ", i)
    metric = run_uav_sim(t_span, sim_opt)
    sum += metric

print(sum / num_runs)