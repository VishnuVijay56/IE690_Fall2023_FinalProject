import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from uav_gym_env import UAVStallEnv
from sample_states import Sampler
from sim_cmds import SimCmds
from evaluate_env import model_evaluator
from autopilot_LQR import Autopilot

## Initializations & Definitions
# Simulation Parameters
sim_opt = SimCmds()
sim_opt.view_sim = False
sim_opt.sim_real_time = False
sim_opt.display_graphs = False
sim_opt.use_kf = False
sim_opt.wind_gust = False

# Instantiate Sampler
sampler = Sampler()

# Instantiate Environment
myEnv_id = 'UAV_testbed/UAV_Environment_v0' # It is best practice to have a space name and version number.
gym.envs.registration.register(
    id=myEnv_id,
    entry_point=UAVStallEnv,
    max_episode_steps=2000, # Customize to your needs.
    reward_threshold=500 # Customize to your needs.
)
vec_env = make_vec_env(myEnv_id, n_envs=4, seed=0, env_kwargs={"sim_options":sim_opt, "sampler":sampler})


# ## Train Agent
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=2_000)
model.save("./models/PPO_AttitudeController_" + str(model._total_timesteps) + "timesteps")

del model

model = PPO.load("./models/PPO_AttitudeController")


## Evaluate Trained Agent
# Reset Environment
eval_env = UAVStallEnv(sim_opt, sampler)
initial_state = eval_env.reset()[0] # Grab initial value for observation
target_state = sampler.target_state
obs = initial_state

# Initialize controller & evaluator
LQR_controller = Autopilot(ts_control=0.01)
evaluator = model_evaluator(initial_state=initial_state, target_state=target_state)

for i in range(2_000):
    # RL model
    action, _states = model.predict(obs, deterministic=True)

    # LQR controller
    # action = LQR_controller.normalized_update(target_state=target_state, state=obs)

    # Step Simulator
    obs, reward, done, info, flags = eval_env.step(action)

    # Update Evaluator
    evaluator.update(action, obs)

## Outputs 
# Controller Metrics
np.set_printoptions(precision=4)
evaluation = evaluator.evaluate()
print("\n ----------------------------------------------- \n")
print("MODEL EVALUATIONS:\n")
print("Was the run successful:", evaluation[0])
print("The rise times for velocities, pitch, and roll:", evaluation[1])
print("The settling times for velocities, pitch, and roll:", evaluation[2])
print("The % overshoot for velocities, pitch, and roll:", evaluation[3])
print("The mean of the control variation:", evaluation[4])

print("\n ----------------------------------------------- \n")
print("INITIALIZATIONS:\n")
print("(North, East, Alt, u, v, w, Phi, Theta, Psi, P, Q, R)")
print("The initial state is:", initial_state)
print("The target state is:", target_state.flatten())

# Plot the run!
evaluator.plot_run()