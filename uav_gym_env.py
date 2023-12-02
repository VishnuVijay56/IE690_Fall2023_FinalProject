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
        #TODO: Make sure these are defined properly
        # self.target_low = np.array([0, 0, 0, 15, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]).flatten()
        # self.target_high = np.array([0, 0, 0, 30, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).flatten()
        # self.target_mean = (self.target_low + self.target_high) / 2
        self.target_state = MAV_State().get_12D_state()

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

        # Create State and Action History Matrix
        self.state_history = np.zeros((12, round(self.end_time / self.Ts)))
        self.action_history = np.zeros((4, round(self.end_time / self.Ts)))

        # TODO: Call the reset function to randomize initial state
        self.reset()

    # Reset environment to a random initial state
    # Argument: Seed
    # Returns: observation of environment corresponding to initial state
    def reset(self, seed=None):
        # print("RESETTING")

        np.random.seed(seed) # Seed initial conditions

        # Get random initial and target state
        new_state = self.random_init_state()
        new_target = self.random_target_state()

        # Reset MAV dynamical state
        self.mav_dynamics.mav_state = new_state
        self.mav_state = new_state

        # Reset target state
        self.target_state = new_target.get_12D_state().astype(np.float32).flatten()

        # Set return vars
        obs = self.mav_state.get_12D_state().astype(np.float32).flatten()
        info = {} # TODO: Change to output useful info

        # Zero the episode history, set initial state
        self.state_history = np.zeros((12, round(self.end_time / self.Ts)))
        self.action_history = np.zeros((4, round(self.end_time / self.Ts)))
        self.time = np.zeros((round(self.end_time / self.Ts)))
        self.state_history[:, 0] = obs

        return (obs, info)
    

    # Generates a random initial state according to initial conditions
    # from Bohn, 2019 paper
    # Argument: None
    # Returns: a new MAV_State() message
    def random_init_state(self):
        phi = np.deg2rad(300 * np.random.rand() - 150)
        theta = np.deg2rad(90 * np.random.rand() - 45)
        psi = np.deg2rad(120 * np.random.rand() - 60)
        
        p = np.deg2rad(120 * np.random.rand() - 60)
        q = np.deg2rad(120 * np.random.rand() - 60)
        r = np.deg2rad(120 * np.random.rand() - 60)
        
        alpha = np.deg2rad(52 * np.random.rand() - 26)
        beta = np.deg2rad(52 * np.random.rand() - 26)
        Va = 18 * np.random.rand() + 12
        
        new_state = MAV_State(0, phi, theta, psi, p, q, r, Va)
        new_state.alpha = alpha
        new_state.beta = beta

        return new_state


    # Generates a random target state according to initial conditions
    # from Bohn, 2019 paper
    # Argument: None
    # Returns: a new MAV_State() message
    def random_target_state(self):
        phi = np.deg2rad(300 * np.random.rand() - 150)
        theta = np.deg2rad(90 * np.random.rand() - 45)
        Va = 18 * np.random.rand() + 12

        new_target = MAV_State()
        new_target.phi = phi
        new_target.theta = theta
        new_target.Va = Va

        return new_target



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
        r_phi = saturate(abs(state.phi[0] - self.target_mean[6]) / 3.3, 0, 0.3)

        r_theta = saturate(abs(state.theta[0] - self.target_mean[7]) / 2.25, 0, 0.3)

        desired_Va = np.sqrt((self.target_mean[3])**2 + (self.target_mean[4])**2 + (self.target_mean[5])**2)
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


    # Evaluates whether or not the agent succeeds and meets target state
    # Argument: None
    # Return: Whether or not controller succeeds, Boolean
    # TODO: Vishnu
    def eval_success(self):
        return None
    

    # Evaluates the rise time of the agent
    # Argument: None
    # Return: Rise time, float
    # TODO: Brian
    def eval_rise_time(self):
        # Function for finding zero crossings
        crossings = lambda a: [np.where(np.diff(np.sign(a), axis=0)[:, i])[0] for i in range(self.state_dim)]

        # Find 10% and 90% bounds
        init_state = self.state_history[:, 0]
        lower_bound  = (self.target_state - init_state)*0.1 + init_state
        upper_bound = (self.target_state - init_state)*0.9 + init_state

        # Find crossings of lower and upper bounds
        lb_crossings = crossings(self.state_history.T - lower_bound)
        ub_crossings = crossings(self.state_history.T - upper_bound)

        # Find t's at crossings of all states
        lb_times = np.zeros(self.state_dim)
        ub_times = np.zeros(self.state_dim)
        times = np.zeros(self.state_dim)

        for i in range(self.state_dim):
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
    # TODO: Vishnu
    def eval_settling_time(self):
        return None
    

    # Evaluates the percent overshoot of the agent
    # Argument: None
    # Return: Percent overshoot, float
    # TODO: Brian
    def eval_overshoot(self, eps=1e-1):
        init_state = self.state_history[:, 0]
        t_i = self.target_state - init_state  # Target - Initial
        direction = np.sign(t_i)
        overshoot_index = np.argmax(direction * self.state_history.T - self.target_state, axis=0)

        overshoot = np.zeros(self.state_dim)
        for i in range(self.state_dim):
            overshoot_val = self.state_history[i, overshoot_index[i]] - self.target_state[i] # Find the distance from the overshoot to the target state

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