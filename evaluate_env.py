import numpy as np

class model_evaluator:
    def __init__(self, initial_state, target_state):
        # Arguments
        self.initial_state = initial_state
        self.target_state = target_state

        # Definitions
        self.action_dim = 4
        self.state_dim = 12
        self.max_steps = 2000
        self.idx = 0
        self.Ts = 0.01
        self.time = np.arange(0, 20.001, 0.01)

        # Zero the episode history, set initial state
        self.state_history = np.zeros((12, self.max_steps+1))
        self.action_history = np.zeros((4, self.max_steps))
        self.time = np.zeros((self.max_steps))
        self.state_history[:, 0] = self.initial_state

    def update(self, action, obs):
        self.action_history[:, self.idx] = action
        self.state_history[:, self.idx+1] = obs

        self.idx = self.idx+1

    # Finds the values of the evaluation criteria:
    # -> Success/ Failure
    # -> Rise Time
    # -> Settling Time
    # -> Percent Overshoot
    # -> Control Variation
    # Argument: None
    # Return: Tuple of evaluated criteria
    def evaluate(self):
        # Evaluate our criteria
        evaluated_success = self.eval_success()
        evaluated_rise_time = self.eval_rise_time()
        evaluated_settling_time = self.eval_settling_time()
        evaluated_overshoot = self.eval_overshoot()
        evaluated_control_variation = self.eval_control_variation()

        # Return
        return (evaluated_success, evaluated_rise_time, evaluated_settling_time, 
                evaluated_overshoot, evaluated_control_variation)

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
        velocity = np.linalg.norm(self.state_history[3:6, :], axis=0)
        roll = self.state_history[6, :]
        pitch = self.state_history[7, :]
        state_history = np.array([velocity, roll, pitch])
        target_state = np.array((np.linalg.norm(self.target_state[3:6]), self.target_state[6], self.target_state[7]))
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
            this_state = self.state_history[:, index].flatten()
            if (not self.state_in_bounds(this_state)):
                return (index + 1) * self.Ts
                
        return 0


    # Evaluates the percent overshoot of the agent
    # Argument: None
    # Return: Percent overshoot, float
    # TODO: Brian
    def eval_overshoot(self, eps=1e-1):
        velocity = np.linalg.norm(self.state_history[3:6, :], axis=0)
        roll = self.state_history[6, :]
        pitch = self.state_history[7, :]
        state_history = np.array([velocity, roll, pitch])
        target_state = np.array((np.linalg.norm(self.target_state[3:6]), self.target_state[6], self.target_state[7]))
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