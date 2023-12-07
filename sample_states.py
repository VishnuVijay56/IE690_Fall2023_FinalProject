"""
sample_states.py: Module to generate random init and target states
    - Author: Vishnu Vijay
    - Created: 12/01/23
"""

import numpy as np

from mav_state import MAV_State

class Sampler:

    def __init__(self) -> None:
        self.curriculum_level = 1

        self.state_dim = 12
        self.action_dim = 4

        self.initial_state = np.zeros(self.state_dim)
        self.target_state = np.zeros(self.state_dim)

    
    # Set the curriculum level of the agent (changes difficulty of the tasks we are teaching)
    # Argument: New level
    # Return: Change in level
    def set_curriculum_level(self, new_lvl):
        old_lvl = self.curriculum_level
        self.curriculum_level = new_lvl

        return (new_lvl - old_lvl)


    # Generates a random initial state according to initial conditions
    # from Bohn, 2019 paper
    # Argument: None
    # Returns: a new MAV_State() message
    def random_init_state(self):
        if self.curriculum_level == 3:
            phi = np.deg2rad(300 * np.random.rand() - 150)
            theta = np.deg2rad(90 * np.random.rand() - 45)
            psi = np.deg2rad(120 * np.random.rand() - 60)
            
            p = np.deg2rad(120 * np.random.rand() - 60)
            q = np.deg2rad(120 * np.random.rand() - 60)
            r = np.deg2rad(120 * np.random.rand() - 60)
            
            alpha = np.deg2rad(52 * np.random.rand() - 26)
            beta = np.deg2rad(52 * np.random.rand() - 26)
            Va = 18 * np.random.rand() + 15

        elif self.curriculum_level == 2:
            phi = np.deg2rad(150 * np.random.rand() - 75)
            theta = np.deg2rad(50 * np.random.rand() - 25)
            psi = np.deg2rad(60 * np.random.rand() - 30)
            
            p = np.deg2rad(60 * np.random.rand() - 30)
            q = np.deg2rad(60 * np.random.rand() - 30)
            r = np.deg2rad(60 * np.random.rand() - 30)
            
            alpha = np.deg2rad(26 * np.random.rand() - 13)
            beta = np.deg2rad(26 * np.random.rand() - 13)
            Va = 12 * np.random.rand() + 18

        elif self.curriculum_level == 1:
            phi = np.deg2rad(20 * np.random.rand() - 10)
            theta = np.deg2rad(18 * np.random.rand() - 9)
            psi = np.deg2rad(30 * np.random.rand() - 15)
            
            p = np.deg2rad(30 * np.random.rand() - 15)
            q = np.deg2rad(30 * np.random.rand() - 15)
            r = np.deg2rad(30 * np.random.rand() - 15)
            
            alpha = np.deg2rad(14 * np.random.rand() - 7)
            beta = np.deg2rad(14 * np.random.rand() - 7)
            Va = 6 * np.random.rand() + 21
        
        else:
            phi = np.deg2rad(2 * np.random.rand() - 1)
            theta = np.deg2rad(2 * np.random.rand() - 1)
            psi = np.deg2rad(2 * np.random.rand() - 1)
            
            p = np.deg2rad(2 * np.random.rand() - 1)
            q = np.deg2rad(2 * np.random.rand() - 1)
            r = np.deg2rad(2 * np.random.rand() - 1)
            
            alpha = np.deg2rad(2 * np.random.rand() - 1)
            beta = np.deg2rad(2 * np.random.rand() - 1)
            Va = 2 * np.random.rand() + 24

        new_state = MAV_State(0, phi, theta, psi, p, q, r, Va)
        new_state.alpha = alpha
        new_state.beta = beta

        self.initial_state = new_state.get_12D_state()

        return new_state


    # Generates a random target state according to initial conditions
    # from Bohn, 2019 paper
    # Argument: None
    # Returns: a new MAV_State() message
    def random_target_state(self):
        if self.curriculum_level == 3:
            phi = np.deg2rad(120 * np.random.rand() - 60)
            theta = np.deg2rad(60 * np.random.rand() - 30)
            Va = 18 * np.random.rand() + 12

        elif self.curriculum_level == 2:
            phi = np.deg2rad(120 * np.random.rand() - 60)
            theta = np.deg2rad(50 * np.random.rand() - 25)
            Va = 12 * np.random.rand() + 15

        elif self.curriculum_level == 1: 
            phi = np.deg2rad(40 * np.random.rand() - 20)
            theta = np.deg2rad(24 * np.random.rand() - 12)
            Va = 6 * np.random.rand() + 18
        
        else:
            phi = np.deg2rad(2 * np.random.rand() - 1)
            theta = np.deg2rad(2 * np.random.rand() - 1)
            Va = 2 * np.random.rand() + 24

        new_target = MAV_State()
        new_target.phi = phi
        new_target.theta = theta
        new_target.Va = Va

        self.target_state = new_target.get_12D_state()

        return new_target
    