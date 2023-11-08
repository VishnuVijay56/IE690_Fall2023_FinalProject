"""
uav_gym_env.py: AI Gym
    - Author: Vishnu Vijay
    - Created: 4/23/23
"""

# Library Imports
import numpy as np
import gym

# Custom Gym Environment for use with custom
# UAV SITL Simulator
class UAVStallEnv(gym.Env):
    # Initialization of UAV Stall Environment
    # Set global variables for use in the environment
    def __init__(self):
        super(UAVStallEnv, self).__init__()


    # Reset environment to initial state
    # Returns: observation of environment corresponding to initial state
    def reset(self):
        return None
    

    # Move agent based on passed action
    # Returns: Observation (states), 
    #          Reward (negative current cost?),
    #          Done (whether episode has terminated),
    #          Info (anything else about the environment) 
    def step(self, action):
        return None


