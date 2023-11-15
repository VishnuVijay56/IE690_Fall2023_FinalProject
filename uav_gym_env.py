"""
uav_gym_env.py: AI Gym
    - Author: Vishnu Vijay
    - Created: 4/23/23
"""

# Library Imports
import numpy as np
import gymnasium as gym

# User-Defined Imports : non-message
from mav import MAV
from mav_dynamics import MAV_Dynamics
from wind_simulation import WindSimulation
from saturate_cmds import saturate

# User-Defined Imports : message
from sim_cmds import SimCmds
from mav_state import MAV_State
from delta_state import Delta_State


# Custom Gym Environment for use with custom
# UAV SITL Simulator
class UAVStallEnv(gym.Env):
    # Initialization of UAV Stall Environment
    # Set global variables for use in the environment
    def __init__(self, sim_options : SimCmds):
        super(UAVStallEnv, self).__init__()

        # 12-D Observation Space
        # (North, East, Alt, u, v, w, Phi, Theta, Psi, P, Q, R)
        observation_low = -np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.pi, np.pi, np.pi, np.inf, np.inf, np.inf]).reshape((12, 1)) #TODO: Assign Values
        observation_high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.pi, np.pi, np.pi, np.inf, np.inf, np.inf]).reshape((12, 1)) #TODO: Assign Values
        self.observation_space = gym.spaces.Box(low = observation_low,
                                                high = observation_high,
                                                dtype = np.float32)

        # 4-D Action Space
        # E, A, R, T
        action_low = np.array([-1, -1, -1, 0]).reshape((4, 1))
        action_high = np.array([1, 1, 1, 1]).reshape((4, 1))
        self.action_space = gym.spaces.Box(low = action_low,
                                           high = action_high, 
                                           dtype = np.float32)
        running_avg_size = 5
        self.actions_E = np.zeros((running_avg_size))
        self.actions_A = np.zeros((running_avg_size))
        self.actions_R = np.zeros((running_avg_size))
        self.actions_T = np.zeros((running_avg_size))

        # Define Target Set
        #TODO: Make sure these are defined properly
        self.target_low = np.array([0, 0, 0, 15, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]).reshape((12, 1))
        self.target_high = np.array([0, 0, 0, 30, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).reshape((12, 1))
        self.target_mean = (self.target_low + self.target_high) / 2

        # Options
        self.sim_options = sim_options

        # Create instance of MAV_Dynamics
        self.Ts = sim_options.Ts
        self.curr_time = 0
        self.end_time = sim_options.t_span[1]
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

        self.random_init_state()
        return self.mav_state.get_12D_state()
    

    def random_init_state(self):
        phi = 300 * np.random.rand(1) - 150
        theta = 90 * np.random.rand(1) - 45
        psi = 120 * np.random.rand(1) - 60
        p = 120 * np.random.rand(1) - 60
        q = 120 * np.random.rand(1) - 60
        r = 120 * np.random.rand(1) - 60
        alpha = 52 * np.random.rand(1) - 26
        beta = 52 * np.random.rand(1) - 26
        Va = 18 * np.random.rand(1) + 12
        
        new_state = MAV_State(0, phi, theta, psi, p, q, r, Va)
        new_state.alpha = alpha
        new_state.beta = beta

        self.mav_dynamics.mav_state = new_state
        self.mav_state = new_state



    # Move agent based on passed action
    # Returns: Observation (states), 
    #          Reward (negative current cost?),
    #          Done (whether episode has terminated),
    #          Info (anything else about the environment) 
    # Action: (Elevator, Aileron, Rudder, Throttle)
    def step(self, action):
        # Intializations
        is_done = False
        failure_flag = 0

        # Initial state (before action)
        curr_state = self.mav_dynamics.mav_state

        # Add action to queue
        self.actions_E = np.concatenate(np.delete(self.actions_E, 0), np.array(action[0]))
        self.actions_A = np.concatenate(np.delete(self.actions_A, 0), np.array(action[1]))
        self.actions_R = np.concatenate(np.delete(self.actions_R, 0), np.array(action[2]))
        self.actions_T = np.concatenate(np.delete(self.actions_T, 0), np.array(action[3]))

        # Wind
        wind_steady_gust = np.zeros((6,1))
        if(not self.sim_options.no_wind):
            wind_steady_gust = self.wind_sim.update()
        
        # Create Commands
        mav_delta = Delta_State(action[0] * np.radians(30), 
                                action[1] * np.radians(30), 
                                action[2] * np.radians(30), 
                                action[3])

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

        # Assign Reward to State, Action pair
        reward = self.cost_function(curr_state, action)

        # Is episode done?
        if (self.curr_time > self.end_time): # Time done
            is_done = True
        elif (any(abs(self.mav_state.get_12D_state()) > 1e8)): # Approaching Numerical error
            is_done = True
            #reward -= 100
            failure_flag = 1
        # elif () # Reached target

        return self.mav_state.get_12D_state(), reward, is_done, failure_flag


    def cost_function(self, state : MAV_State, action : np.array):
        r_phi = saturate(abs(state.phi - self.target_mean[6]) / 3.3, 0, 0.3)

        r_theta = saturate(abs(state.theta - self.target_mean[7]) / 2.25, 0, 0.3)

        desired_Va = np.sqrt((self.target_mean[3])**2 + (self.target_mean[4])**2 (self.target_mean[5])**2)
        r_Va = saturate(abs(state.Va - desired_Va), 0, 0.3)

        tot_comm_cost = self.command_cost(self.actions_E) + self.command_cost(self.actions_A) + \
                        self.command_cost(self.actions_R) + self.command_cost(self.actions_T)

        r_delta = saturate(tot_comm_cost / 80, 0, 0.1)

        total_reward = -(r_phi + r_theta + r_Va + r_delta)

        return total_reward


    def command_cost(self, set_of_actions : np.array):
        sum_diff = 0
        for i in range(len(set_of_actions) - 1):
            diff = abs(set_of_actions[i] - set_of_actions[i + 1])
            sum_diff += diff
        
        return sum_diff