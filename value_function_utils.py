# Utility functions to visualize value functions 

import matplotlib.pyplot as plt
import numpy as np

def visualize_maze_values(q_table, env, isMaze=True, arrow=True):
    """Plot the Tabular Q-value function
    
    Args: 
        q_table (np.array): Tabular Q-value function
        env (gym.Env): Gym environment with discrete space. e.g., MazeEnv
        isMaze (bool, optional): True for MazeEnv. Defaults to True.
        arrow (bool, optional): Set to True for drawing directional arrows. Defaults to True.
    """
    # (x, y) coordinates
    direction = {
        0: (0, -0.4),
        1: (0, 0.4),
        2: (-0.4, 0),
        3: (0.4, 0)
    }
    v = np.max(q_table, axis=1)
    best_action = np.argmax(q_table, axis=1)
    if isMaze:
        idx2cell = env.index_to_coordinate_map
        for i in range(8):
            _, ax = plt.subplots()
            ax.set_axis_off()
            y_mat = np.zeros(env.dim)
            for j in range(len(idx2cell)):
                pos = idx2cell[j]
                y_mat[pos[0], pos[1]] = v[8 * j + i]
                if arrow:
                    a = best_action[8*j+i]
                    ax.arrow(
                        pos[1],
                        pos[0],
                        direction[a][0],
                        direction[a][1],
                        head_width=0.05,
                        head_length=0.1,
                        fc="g",
                        ec="g",
                    )
            y_mat[env.goal_pos] = max(v) + 0.1
            ax.imshow(y_mat, cmap="hot")
            plt.savefig(f"results/value_iter_{i}.png", bbox_inches="tight")
    
    else:
        n = int(np.sqrt(len(v)))
        state_value_func = np.zeros((n, n))
        for r in range(n):
            for c in range(n):
                if not (r == (n-1) and c == (n-1)):
                    state_value_func[r, c] = v[n * c + r]
                    if arrow:
                        d = direction[best_action[n * c + r]]
                        plt.arrow(
                            c,
                            r,
                            d[0],
                            d[1],
                            head_width=0.05,
                            head_length=0.1,
                            fc="r",
                            ec="r"
                        )
        state_value_func[env.goal_pos] = max(v[:-1]) + 0.1
        plt.imshow(state_value_func, cmap="hot")
    plt.show()


def visualize_grid_state_values(grid_state_values):
    """Visualizes the state value function for the grid"""
    state_value = grid_state_values.reshape((3, 4))
    state_value_positions = [
        (0.15, 0.15),
        (1.15, 0.15),
        (2.15, 0.15),
        (3.15, 0.15),
        (0.15, 1.15),
        (1.15, 1.15),
        (2.15, 1.15),
        (3.15, 1.15),
        (0.15, 2.15),
        (1.15, 2.15),
        (2.15, 2.15),
        (3.15, 2.15)
    ]
    plt.figure(figsize=(12, 5))
    plt.imshow(
        grid_state_values,
        cmap="Greens",
        interpolation="nearest"
    )
    plt.colorbar()
    for i, (xi, yi) in enumerate(state_value_positions):
        plt.text(xi, yi, round(state_value.flatten()[i], 2), size=11, color="r")
    plt.show()



def visualize_grid_action_values(grid_action_values):
    top = grid_action_values[:, 0].reshape((3, 4))
    top_value_positions = [
        (0.38, 0.25),
        (1.38, 0.25),
        (2.38, 0.25),
        (3.38, 0.25),
        (0.38, 1.25),
        (1.38, 1.25),
        (2.38, 1.25),
        (3.38, 1.25),
        (0.38, 2.25),
        (1.38, 2.25),
        (2.38, 2.25),
        (3.38, 2.25)
    ]
    right = grid_action_values[:, 1].reshape((3, 4))
    right_value_positions = [
        (0.65, 0.5),
        (1.65, 0.5),
        (2.65, 0.5),
        (3.65, 0.5),
        (0.65, 1.5),
        (1.65, 1.5),
        (2.65, 1.5),
        (3.65, 1.5),
        (0.65, 2.5),
        (1.65, 2.5),
        (2.65, 2.5),
        (3.65, 2.5)
    ]
    bottom = grid_action_values[:, 2].reshape((3, 4))
    bottom_value_positions = [
        (0.38, 0.8),
        (1.38, 0.8),
        (2.38, 0.8),
        (3.38, 0.8),
        (0.38, 1.8),
        (1.38, 1.8),
        (2.38, 1.8),
        (3.38, 1.8),
        (0.38, 2.8),
        (1.38, 2.8),
        (2.38, 2.8),
        (3.38, 2.8)
    ]
    left = grid_action_values[:, 3].reshape((3, 4))
    left_value_positions = [
        (0.05, 0.5),
        (1.05, 0.5),
        (2.05, 0.5),
        (3.05, 0.5),
        (0.05, 1.5),
        (1.05, 1.5),
        (2.05, 1.5),
        (3.05, 1.5),
        (0.05, 2.5),
        (1.05, 2.5),
        (2.05, 2.5),
        (3.05, 2.5)
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_ylim(3, 0)
    tripcolor = plot_triangular(
        right=right,
        left=left,
        top=top,
        bottom=bottom,
        ax=ax,
        triplot={"color": "k", "lw": 1},
        tripcolorkw={"cmap": "rainbow_r"}
    )

    ax.margin(0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.colorbar(tripcolor)

    for i, (xi, yi) in enumerate(top_value_positions):
        plt.text(xi, yi, round(top.flatten()[i], 2), size=11, color="w")
    for i, (xi, yi) in enumerate(right_value_positions):
        plt.text(xi, yi, round(right.flatten()[i], 2), size=11, color="w")
    for i, (xi, yi) in enumerate(left_value_positions):
        plt.text(xi, yi, round(left.flatten()[i], 2), size=11, color="w")
    for i, (xi, yi) in enumerate(bottom_value_positions):
        plt.text(xi, yi, round(bottom.flatten()[i], 2), size=11, color="w")
    
    plt.show()


def plot_triangular(left,  bottom, right, top, ax=None, triplotkw={}, tripcolorkw={}):

    if not ax:
        ax = plt.gca()
    n = left.shape[0]
    m = left.shape[1]

    a = np.array([[0, 0],
                  [0, 1],
                  [0.5, 0.5],
                  [1, 0],
                  [1, 1]])
    tr = np.array([[0, 1, 2], [0, 2, 3], [2, 3, 4], [1, 2, 4]])

    A = np.zeros((n * m * 5, 2))
    Tr = np.zeros((n * m * 4, 3))
    
    for i in range(n):
        for j in range(m):
            k = i * m + j
            A[k * 5 : (k+1)*5, :] = np.c_[a[:, 0] + j, a[:, 1] + i]
            Tr[k * 4 : (k + 1) * 5, :] = tr + k * 5

    C = np.c_[
        left.flatten(), bottom.flatten(), right.flatten(), top.flatten()
    ].flatten()

    _ = ax.triplot(A[:, 0], A[:, 1], Tr, **triplotkw)
    tripcolor = ax.tripcolor(A[: 0], A[:, 1], Tr, facecolors=C, **tripcolorkw)
    return tripcolor