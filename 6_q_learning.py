
import numpy as np
import random 

from value_function_utils import visualize_grid_action_values
from envs.gridworldv2 import GridworldV2Env

def q_learning(env: GridworldV2Env, max_episode=1000):
    
    # initialize q table 
    q = np.zeros((len(env.distinct_states), env.action_space.n))
    q[env.bomb_state] = -1
    q[env.goal_state] = 1 
    gamma = 0.99
    alpha = 0.01
    epsilon = 1 
    
    for episode in range(max_episode):
        done = False
        state = env.reset()
        
        while not done: 
            epsilon = decay(epsilon)
            action = greedy_policy(q[state], epsilon=epsilon)
            next_state, reward, done = env.step(action)

            # Q-update 
            q[state][action] += alpha * (
                reward + gamma * max(q[next_state]) - q[state][action]
            )

            state = next_state 
    
    visualize_grid_action_values(q)


def greedy_policy(q_values: np.ndarray, epsilon=0.1):
    """epsilon greedy policy"""
    if random.random() >= epsilon:
        return np.argmax(q_values)
    else: 
        return random.randint(0, env.action_space.n-1)

def decay(epsilon):
    max_epsilon = 1 
    min_epsilon = 0.01 
    df = 0.99
    if epsilon > max_epsilon:
        epsilon = max_epsilon
        return epsilon
    elif epsilon <= min_epsilon:
        return min_epsilon
    else:
        return df * epsilon

if __name__ == "__main__": 
    env = GridworldV2Env()
    max_episode = 4000
    q_learning(env, max_episode=max_episode)

