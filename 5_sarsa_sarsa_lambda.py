
import numpy as np
import random
from value_function_utils import visualize_grid_action_values
from envs.gridworldv2 import GridworldV2Env


def sarsa(env: GridworldV2Env, max_episodes):
    q = np.zeros((len(env.distinct_states), env.action_space.n))
    q[env.goal_state] = 1
    q[env.bomb_state] = -1
    gamma = 0.99
    alpha = 0.01
    epsilon = 1

    for episode in range(max_episodes):
        done = False 
        state = env.reset() 
        action = env.action_space.sample() # initialize by random
        
        while not done:
            next_state, reward, done = env.step(action)
            epsilon = decay(epsilon)
            next_action = greedy_policy(q[state], epsilon=epsilon)

            # q-update
            q[state][action] += alpha * (
                reward + gamma * q[next_state][next_action] - q[state][action]
            )
            state = next_state
            action = next_action
    
    visualize_grid_action_values(q)


def greedy_policy(q_vals, epsilon=0.1):
    if random.random() >= epsilon:
        return np.argmax(q_vals)
    else:
        return random.randint(0, env.action_space.n-1)

def decay(epsilon):
    max_epsilon = 1
    min_epsilon = 0.01
    decay_factor = 0.99
    if epsilon > max_epsilon:
        epsilon = decay_factor * max_epsilon
        return max_epsilon 
    elif (epsilon <= max_epsilon) & (epsilon > min_epsilon):
        epsilon *= decay_factor
        return epsilon 
    else:
        return min_epsilon





if __name__ == "__main__":
    max_episodes = 4000
    env = GridworldV2Env(step_cost=-0.1, max_ep_length=30)
    sarsa(env, max_episodes)

