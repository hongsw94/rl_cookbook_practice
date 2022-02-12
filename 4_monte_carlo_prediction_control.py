# Monte-carlo prediction and control

import numpy as np
from envs.gridworldv2 import GridworldV2Env
from value_function_utils import visualize_grid_action_values, visualize_grid_state_values

def mc_predict(env, max_episodes):
    pass 

def epsilon_greedy_policy(action_logits, epsilon=0.2):
    pass 

def mc_control(env, max_episodes):
    pass 


if __name__ == "__main__":
    max_episodes = 4000
    env = GridworldV2Env(step_cost=-0.1, max_ep_length=30)
    print(f"===Monte Carlo Prediction===")
    mc_predict(env, max_episodes=max_episodes)
    print(f"===Monte Carlo control===")
    mc_control(env, max_episodes=max_episodes)