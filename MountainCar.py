import gym
import numpy as np
import pickle
import os

#! Initialize the MountainCar environment in human-rendering mode
env = gym.make("MountainCar-v0", render_mode="human")

#! Define the number of bins for discretizing position and velocity
num_bins = (50, 50)  #! Higher number of bins improves precision
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))

#! Create equally spaced bins for discretization
state_bins = [np.linspace(b[0], b[1], num_bins[i] - 1) for i, b in enumerate(state_bounds)]

#! Define the path to the trained Q-table file
q_table_path = os.path.join("MountainCar", "q_table.pkl")

#! Load the trained Q-table from the file
try:
    with open(q_table_path, "rb") as f:
        q_table = pickle.load(f)
    print("Q-table loaded successfully!")
except FileNotFoundError:
    #! Handle the case where the Q-table file is not found
    print(f"Error: Q-table file not found at '{q_table_path}'. Please ensure the file exists.")
    exit()

def discretize_state(state):
    """
    Discretize a continuous state into discrete indices for the Q-table.

    Args:
        state (numpy.ndarray): Continuous state from the environment.

    Returns:
        tuple: Discrete indices corresponding to the state.
    """
    #! Map each component of the state to its respective bin
    return tuple(np.digitize(state[i], state_bins[i]) for i in range(len(state)))

def test_agent(env, q_table, episodes=5):
    """
    Test the trained agent using the Q-table.

    Args:
        env (gym.Env): The MountainCar environment.
        q_table (numpy.ndarray): The trained Q-table.
        episodes (int): Number of episodes to test the agent.
    """
    for episode in range(episodes):
        #! Reset the environment at the start of each episode
        state, _ = env.reset()
        state = discretize_state(state)
        terminated = False
        total_reward = 0

        print(f"Starting Episode {episode + 1}...")
        while not terminated:
            #! Select the best action based on the Q-table
            action = np.argmax(q_table[state])
            #! Perform the selected action and observe the outcome
            next_state, reward, terminated, truncated, _ = env.step(action)
            env.render()  #! Render the environment for visualization
            total_reward += reward
            #! Update the current state to the discretized next state
            state = discretize_state(next_state)

        #! Display the total reward for the completed episode
        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

#! Run the agent testing process
test_agent(env, q_table, episodes=5)

#! Close the environment to release resources
env.close()