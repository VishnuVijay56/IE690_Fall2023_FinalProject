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
        new_target = self.sampler.random_target_state()

        # Reset MAV dynamical state
        self.mav_dynamics.mav_state = new_state
        self.mav_state = new_state

        # Reset target state
        self.target_state = new_target.get_12D_state().astype(np.float32).flatten()

        # Set return vars
        obs = self.mav_state.get_12D_state().astype(np.float32).flatten()
        info = {"target_state":self.target_state} # TODO: Change to output useful info

        # Zero the episode history, set initial state
        self.state_history = np.zeros((self.state_dim, self.max_steps))
        self.action_history = np.zeros((self.action_dim, self.max_steps))
        self.time = np.zeros((self.max_steps))
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

        # Store States and Actions
        self.state_history[:, self.idx] = self.mav_state.get_12D_state().flatten()
        self.action_history[:, self.idx] = action
        self.time[self.idx] = self.curr_time

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

        return self.mav_state.get_12D_state().flatten(), reward, is_done, is_truncated, {"Flags":failure_flag}


    # Assigns a reward to a state action pair
    # Functions taken from Bohn, 2019
    # Argument: 12D State of the UAV,
    #           Action to be taken by the UAV
    # Return: Sum of negative cost of roll, pitch, velocity, and actuator command
    def cost_function(self, state : MAV_State, action : np.array):
        r_phi = saturate(abs(state.phi[0] - self.target_state[6]) / 3.3, 0, 0.3)

        r_theta = saturate(abs(state.theta[0] - self.target_state[7]) / 2.25, 0, 0.3)

        desired_Va = np.sqrt((self.target_state[3])**2 + (self.target_state[4])**2 + (self.target_state[5])**2)
        r_Va = saturate(abs(state.Va[0] - desired_Va), 0, 0.3)

        tot_comm_cost = self.command_cost(self.actions_E) + self.command_cost(self.actions_A) + \
                        self.command_cost(self.actions_R) + self.command_cost(self.actions_T)

        r_delta = saturate(tot_comm_cost / 80, 0, 0.1)

        total_reward = -(r_phi + r_theta + r_Va + r_delta)

        return total_reward


    # Computes sum of the absolute differences between consecutive elements of a set
    # Argument: A list of past 6 actions taken by UAV for a specific actuator
    # Return: Sum of differences
    def command_cost(self, set_of_actions : np.array):
        sum_diff = 0
        for i in range(len(set_of_actions) - 1):
            diff = abs(set_of_actions[i] - set_of_actions[i + 1])
            sum_diff += diff
        
        return sum_diff


    # Finds the values of the evaluation criteria:
    # -> Success/ Failure
    # -> Rise Time
    # -> Settling Time
    # -> Percent Overshoot
    # -> Control Variation
    # Argument: None
    # Return: Tuple of evaluated criteria
    def evaluate_model(self):
        # Evaluate our criteria
        evaluated_success = self.eval_success()
        evaluated_rise_time = self.eval_rise_time()
        evaluated_settling_time = self.eval_settling_time()
        evaluated_overshoot = self.eval_overshoot()
        evaluated_control_variation = self.eval_control_variation()

        # Return
        return (evaluated_success, evaluated_rise_time, evaluated_settling_time, 
                evaluated_overshoot, evaluated_control_variation)


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
        for i in range(self.max_steps):
            this_state = self.state_history[:, i].flatten()
            # reset the index if curr state not in target bounds
            if (not self.state_in_bounds(this_state)):
                start_of_succ = i
            # if curr state remains in target bounds for 100 time steps
            if (i - start_of_succ) > 100:
                return True 

        return False # didn't reach target goal
    

    # Evaluates the rise time of the agent
    # Argument: None
    # Return: Rise time, float
    # TODO: Brian
    def eval_rise_time(self):
        # Find 10% and 90% bounds
        velocity = np.norm(self.state_history[:, 3:6], axis=1)
        roll = self.state_history[:, 6]
        pitch = self.state_history[:, 7]
        state_history = np.array([velocity, roll, pitch])
        target_state = np.array((np.norm(target_state[3:6], axis=1), target_state[6], target_state[7]))
        len_arg = len(target_state)

        # Function for finding zero crossings
        crossings = lambda a: [np.where(np.diff(np.sign(a), axis=0)[:, i])[0] for i in range(len_arg)]
        init_state = state_history[:, 0]
        lower_bound  = (target_state - init_state)*0.1 + init_state
        upper_bound = (target_state - init_state)*0.9 + init_state

        # Find crossings of lower and upper bounds
        lb_crossings = crossings(state_history.T - lower_bound)
        ub_crossings = crossings(state_history.T - upper_bound)

        # Find t's at crossings of all states
        lb_times = np.zeros(len_arg)
        ub_times = np.zeros(len_arg)
        times = np.zeros(len_arg)

        for i in range(len_arg):
            try: # Check to see if bounds were crossed, if never crossed then rise time is infinity
                lb_times[i] = self.time[lb_crossings[i][0]]
                ub_times[i] = self.time[ub_crossings[i][0]]
                times[i] = ub_times[i] - lb_times[i]
            except:
                times[i] = np.inf

        return times
    

    # Evaluates the settling time of the agent
    # Argument: None
    # Return: Settling time, float
    def eval_settling_time(self):

        for i in range(self.max_steps):
            index = self.max_steps - i - 1
            this_state = self.state_history[index].flatten()
            if (not self.state_in_bounds(this_state)):
                return (index + 1) * self.Ts
                
        return 0
    

    # Evaluates the percent overshoot of the agent
    # Argument: None
    # Return: Percent overshoot, float
    # TODO: Brian
    def eval_overshoot(self, eps=1e-1):
        velocity = np.norm(self.state_history[:, 3:6], axis=1)
        roll = self.state_history[:, 6]
        pitch = self.state_history[:, 7]
        state_history = np.array([velocity, roll, pitch])
        target_state = np.array((np.norm(self.target_state[3:6], axis=1), self.target_state[6], self.target_state[7]))
        len_arg = len(target_state)

        init_state = state_history[:, 0]
        t_i = target_state - init_state  # Target - Initial
        direction = np.sign(t_i)
        overshoot_index = np.argmax(direction * state_history.T - target_state, axis=0)

        overshoot = np.zeros(len_arg)
        for i in range(len_arg):
            overshoot_val = state_history[i, overshoot_index[i]] - target_state[i] # Find the distance from the overshoot to the target state

            if t_i[i] <= eps: # If target state is basically initial state, then the overshoot percent is value of the overshoot
                overshoot[i] = overshoot_val * 100
            else:
                overshoot[i] = overshoot_val/t_i[i] * 100 # Finds the ratio of the overshoot value and the distance to the target from the initial state

        return overshoot
    
    
    # Evaluates the control variation of the agent
    # Argument: None
    # Return: Control variation, float
    # TODO: Vishnu
    def eval_control_variation(self):
        control_variation = np.zeros((self.action_dim))
        for (i, a) in enumerate(self.action_history):
            control_variation[i] = self.command_cost(a)

        return np.mean(control_variation)