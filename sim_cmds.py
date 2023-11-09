"""
sim_cmds.py: command/message type for simulation
    - Author: Vishnu Vijay
    - Created: 4/25/23
"""

import numpy as np

class SimCmds:
    def __init__(self):
        self.view_sim = False
        self.sim_real_time = False
        self.display_graphs = False
        self.fullscreen = False
        self.use_kf = False
        self.wind_gust = False
        self.no_wind = True
        self.t_span = (0, 20)
        self.Ts = 0.01
        self.steady_state_wind = np.array([[0., 0., 0.]]).T