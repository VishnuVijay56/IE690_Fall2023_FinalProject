"""
uav_gym_env.py: AI Gym
    - Author: Vishnu Vijay
    - Created: 9/23/23
"""

# Library Imports
import numpy as np
import gymnasium as gym

# User-Defined Imports : non-message
from mav import MAV
from mav_dynamics import MAV_Dynamics
from wind_simulation import WindSimulation
from saturate_cmds import saturate
from sample_states import Sampler
from helper import EulerToQuaternion

# User-Defined Imports : message
from sim_cmds import SimCmds
from mav_state import MAV_State
from delta_state import Delta_State


# Custom Gym Environment for use with custom
# UAV SITL Simulator
class UAVStallEnv(gym.Env):
    # Initialization of UAV Stall Environment
    # Set global variables for use in the environment
    def __init__(self, sim_options : SimCmds, sampler : Sampler):
        super(UAVStallEnv, self).__init__()

        # 12-D Observation Space
        # (North, East, Alt, u, v, w, Phi, Theta, Psi, P, Q, R)
        self.state_dim = 12
        max_val = np.inf
        observation_low = -np.array([max_val, max_val, max_val, max_val, max_val, max_val, np.pi, np.pi, np.pi, max_val, max_val, max_val]).flatten() #TODO: Assign Values
        observation_high = np.array([max_val, max_val, max_val, max_val, max_val, max_val, np.pi, np.pi, np.pi, max_val, max_val, max_val]).flatten() #TODO: Assign Values
        self.observation_space = gym.spaces.Box(low = observation_low,
                                                high = observation_high,
                                                dtype = np.float32)

        # 4-D Action Space
        # E, A, R, T
        self.action_dim = 4
        action_low = np.array([-1, -1, -1, -1]).flatten()
        action_high = np.array([1, 1, 1, 1]).flatten()
        self.action_space = gym.spaces.Box(low = action_low,
                                           high = action_high, 
                                           dtype = np.float32)
        running_avg_size = 5
        self.actions_E = np.zeros((running_avg_size))
        self.actions_A = np.zeros((running_avg_size))
        self.actions_R = np.zeros((running_avg_size))
        self.actions_T = np.zeros((running_avg_size))

        # Define Target Set
        self.target_state = MAV_State().get_12D_state()

        # Options
        self.sim_options = sim_options
        self.sampler = sampler

        # Create instance of MAV_Dynamics
        self.Ts = sim_options.Ts
        self.curr_time = 0
        self.idx = 0
        self.end_time = sim_options.t_span[1]
        self.max_steps = round(self.end_time / self.Ts)
        self.mav_dynamics = MAV_Dynamics(time_step=self.Ts) # time step in seconds
        self.mav_state = self.mav_dynamics.mav_state
        # self.mav_delta = self.mav_dynamics.delta

        # Create instance of MAV object using MAV_State object
        self.mav_model = MAV(self.mav_state, sim_options.fullscreen, sim_options.view_sim)

        # Create instance of wind simulation
        self.wind_sim = WindSimulation(self.Ts, sim_options.steady_state_wind, sim_options.wind_gust)

        # TODO: Call the reset function to randomize initial state
        self.reset()
    

    # Reset environment to a random initial state
    # Argument: Seed
    # Returns: observation of environment corresponding to initial state
    def reset(self, seed=None):
        # print("RESETTING")

        np.random.seed(seed) # Seed initial conditions

        # Get random initial and target state
        new_state = self.sampler.random_init_state()
        self.mav_state = new_state
        new_target = self.sampler.random_target_state()

        # Reset MAV dynamical state
        x_euler = new_state.get_12D_state().flatten()
        angles = x_euler[6:9]
        quat = EulerToQuaternion(angles[0], angles[1], angles[2]).flatten()
        x_quat = np.hstack((x_euler[0:6], quat, x_euler[9:]))
        self.mav_dynamics.internal_state = x_quat.reshape((x_quat.size,1))

        # Reset target state
        self.target_state = new_target.get_12D_state().astype(np.float32).flatten()

        # Set return vars
        obs = self.mav_state.get_12D_state().astype(np.float32).flatten()
        info = {"target_state":self.target_state} # TODO: Change to output useful info

        # Zero the episode history, set initial state
        self.state_history = np.zeros((self.state_dim, self.max_steps+2))
        self.action_history = np.zeros((self.action_dim, self.max_steps+1))
        self.time = np.zeros((self.max_steps+1))
        self.state_history[:, 0] = obs

        return (obs, info)


    # Move agent based on passed action
    # Argument: Action (list of (E, A, R, T))
    # Returns: Observation (states), 
    #          Reward (negative current cost?),
    #          Done (whether episode has terminated),
    #          Info (anything else about the environment) 
    def step(self, action):
        # print("  ---> IM STEPPIN HERE!")

        # Intializations
        is_done = False
        is_truncated = False
        failure_flag = 0

        # Initial state (before action)
        curr_state = self.mav_dynamics.mav_state

        # Add action to queue
        self.actions_E = np.hstack((np.delete(self.actions_E, 0), np.array(action[0])))
        self.actions_A = np.hstack((np.delete(self.actions_A, 0), np.array(action[1])))
        self.actions_R = np.hstack((np.delete(self.actions_R, 0), np.array(action[2])))
        self.actions_T = np.hstack((np.delete(self.actions_T, 0), np.array(action[3])))

        # Wind
        wind_steady_gust = np.zeros((6,1))
        if(not self.sim_options.no_wind):
            wind_steady_gust = self.wind_sim.update()
        
        # Create Commands
        mav_delta = Delta_State(action[0] * np.radians(30), 
                                action[1] * np.radians(30), 
                                action[2] * np.radians(30), 
                                (action[3] + 1) * 0.5)

        # Dynamics
        self.mav_dynamics.iterate(mav_delta, wind_steady_gust)
        self.mav_state = self.mav_dynamics.mav_state

        # Update MAV mesh for viewing
        self.mav_model.set_mav_state(self.mav_state)
        self.mav_model.update_mav_state()
        if(self.mav_model.view_sim):
            self.mav_model.update_render()

        # # Store States and Actions
        # self.state_history[:, self.idx+1] = self.mav_state.get_12D_state().flatten()
        # self.action_history[:, self.idx] = action
        # self.time[self.idx] = self.curr_time

        # Update Time
        self.curr_time += self.Ts
        self.idx += 1

        # Assign Reward to State, Action pair
        reward = self.cost_function(curr_state, action)

        # Variables for evaluating success
        self.start_of_succ = 0
        is_succ = False
        if (not self.state_in_bounds(self.mav_state.get_12D_state().flatten())):
            # reset the index if curr state not in target bounds
            self.start_of_succ = self.idx
        if (self.idx - self.start_of_succ) > 100:
            # if curr state remains in target bounds for 100 time steps
            is_succ = True

        # Is episode done?
        if (self.curr_time > self.end_time): # Time done
            is_done = True
        elif (any(abs(self.mav_state.get_12D_state()) > 1e8)): # Approaching Numerical error
            is_done = True
            #reward -= 100
            failure_flag = 1
        elif (is_succ): # Reached target
            reward += 10
            is_done = True

        return self.mav_state.get_12D_state().flatten(), reward, is_done, is_truncated, {"Flags":failure_flag}


    # Assigns a reward to a state action pair
    # Functions taken from Bohn, 2019
    # Argument: 12D State of the UAV,
    #           Action to be taken by the UAV
    # Return: Sum of negative cost of roll, pitch, velocity, and actuator command
    def cost_function(self, state : MAV_State, action : np.array):
        d2r =np.pi/180

        # r_phi = saturate(abs(state.phi[0] - self.target_state[6])/, 0, 3)
        r_phi = self.calculate_reward(abs(state.phi[0] - self.target_state[6]), window=60*d2r, magnitude=0.4)

        # r_theta = saturate(abs(state.theta[0] - self.target_state[7])/25, 0, 2.5)
        r_theta = self.calculate_reward(abs(state.theta[0] - self.target_state[7]), window=60*d2r, magnitude=0.225)

        desired_Va = np.sqrt((self.target_state[3])**2 + (self.target_state[4])**2 + (self.target_state[5])**2)
        # r_Va = saturate(abs(state.Va[0] - desired_Va)/25, 0, 2)
        r_Va = self.calculate_reward(abs(state.Va[0] - desired_Va), window=8, magnitude=0.225)

        tot_comm_cost = self.command_cost(self.actions_E) + self.command_cost(self.actions_A) + \
                        self.command_cost(self.actions_R) + self.command_cost(self.actions_T)

        # r_delta = saturate(tot_comm_cost / 80, 0, 1)
        r_delta = self.calculate_reward(tot_comm_cost, window=6, magnitude=0.1)

        # Penalize Rates
        rates = abs(np.linalg.norm(np.array((state.p, state.q, state.r))))
        # r_rates = saturate(abs(rates) / 80, 0, 1.5)
        r_rates = self.calculate_reward(rates, window=30*d2r, magnitude=0.5)

        total_reward = -(r_phi + r_theta + r_Va + r_delta + r_rates)

        return total_reward
    
    def calculate_reward(self, value, window, magnitude):
        return saturate(value/(window/magnitude), 0, magnitude)



    # Computes sum of the absolute differences between consecutive elements of a set
    # Argument: A list of past 6 actions taken by UAV for a specific actuator
    # Return: Sum of differences
    def command_cost(self, set_of_actions : np.array):
        sum_diff = 0
        for i in range(len(set_of_actions) - 1):
            diff = abs(set_of_actions[i] - set_of_actions[i + 1])
            sum_diff += diff
        
        return sum_diff


    # Determines if the state is within the target bounds
    # Argument: State (12-D vector)
    # Return: The state is within target bounds, Boolean
    def state_in_bounds(self, state : np.array):
        curr_roll = state[6]
        curr_pitch = state[7]
        curr_Va = np.sqrt(state[3]**2 + state[4]**2 + state[5]**2)

        target_roll = self.target_state[6]
        target_pitch = self.target_state[7]
        target_Va = np.sqrt(self.target_state[3]**2 + self.target_state[4]**2 + self.target_state[5]**2)

        met_target = True
        if abs(curr_roll - target_roll) > np.deg2rad(5):
            met_target = False
        if abs(curr_pitch - target_pitch) > np.deg2rad(5):
            met_target = False
        if abs(curr_Va - target_Va) > (2):
            met_target = False

        return met_target


    # Evaluates whether or not the agent succeeds and meets target state
    # Argument: None
    # Return: Whether or not controller succeeds, Boolean
    def eval_success(self):

        start_of_succ = 0
        for i in range(self.idx):
            this_state = self.state_history[:, i].flatten()
            # reset the index if curr state not in target bounds
            if (not self.state_in_bounds(this_state)):
                start_of_succ = i
            # if curr state remains in target bounds for 100 time steps
            if (i - start_of_succ) > 100:
                return True 

        return False # didn't reach target goal
    