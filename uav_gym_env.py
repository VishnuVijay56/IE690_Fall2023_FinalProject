"""
uav_gym_env.py: AI Gym
    - Author: Vishnu Vijay
    - Created: 4/23/23
"""

# Library Imports
import numpy as np
import gym

# User-Defined Imports : non-message
from mav import MAV
from mav_dynamics import MAV_Dynamics
from wind_simulation import WindSimulation

# User-Defined Imports : message
from sim_cmds import SimCmds
from delta_state import Delta_State


# Custom Gym Environment for use with custom
# UAV SITL Simulator
class UAVStallEnv(gym.Env):
    # Initialization of UAV Stall Environment
    # Set global variables for use in the environment
    def __init__(self, sim_options : SimCmds):
        super(UAVStallEnv, self).__init__()

        # Options
        self.sim_options = sim_options

        # Create instance of MAV_Dynamics
        self.Ts = sim_options.Ts
        self.curr_time = 0
        self.mav_dynamics = MAV_Dynamics(time_step=self.Ts) # time step in seconds
        self.mav_state = self.mav_dynamics.mav_state
        # self.mav_delta = self.mav_dynamics.delta

        # Create instance of MAV object using MAV_State object
        self.mav_model = MAV(self.mav_state, sim_options.fullscreen, sim_options.view_sim)

        # Create instance of wind simulation
        self.wind_sim = WindSimulation(self.Ts, sim_options.steady_state_wind, sim_options.wind_gust)


    # Reset environment to initial state
    # Returns: observation of environment corresponding to initial state
    def reset(self):
        return None
    

    # Move agent based on passed action
    # Returns: Observation (states), 
    #          Reward (negative current cost?),
    #          Done (whether episode has terminated),
    #          Info (anything else about the environment) 
    # Action: (Elevator, Aileron, Rudder, Throttle)
    def step(self, action):

        # Wind
        wind_steady_gust = np.zeros((6,1))
        if(not self.sim_options.no_wind):
            wind_steady_gust = self.wind_sim.update()
        
        # Create Commands
        mav_delta = Delta_State(action[0], action[1], action[2], action[3])

        # Dynamics
        self.mav_dynamics.iterate(mav_delta, wind_steady_gust)
        self.mav_state = self.mav_dynamics.mav_state

        # Update MAV mesh for viewing
        self.mav_model.set_mav_state(self.mav_state)
        self.mav_model.update_mav_state()
        if(self.mav_model.view_sim):
            self.mav_model.update_render()
        
        # Update Time
        self.curr_time += self.Ts

        return None


