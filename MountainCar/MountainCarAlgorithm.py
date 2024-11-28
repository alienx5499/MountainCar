import gym
import numpy as np
import pickle
import os

class CustomMountainCarEnv(gym.Wrapper):
    """
    Custom wrapper for the MountainCar environment to add a modified reward system.
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        """
        Custom step function to modify the reward system.

        Args:
            action (int): Action taken by the agent.

        Returns:
            tuple: Updated state, modified reward, termination flag, truncation flag, and additional info.
        """
        state, reward, terminated, truncated, info = self.env.step(action)

        #! Decompose state into position and velocity
        position, velocity = state
        goal_position = 0.5

        #! Apply custom reward system
        reward = -1  #! Base penalty for each step to encourage faster goal completion
        if position >= goal_position:
            reward = 100  #! Large reward for reaching the goal
        else:
            reward += max(0, velocity * 10)  #! Reward for gaining positive velocity
            if (velocity > 0 and action == 2) or (velocity < 0 and action == 0):
                reward += 1  #! Reward for actions aligned with velocity direction
            else:
                reward -= 5  #! Penalize counterproductive actions

        return state, reward, terminated, truncated, info


#! Initialize environment with the custom wrapper
env = CustomMountainCarEnv(gym.make("MountainCar-v0"))

#! Hyperparameters for Q-learning
num_episodes = 200000  #! Total number of episodes for training
learning_rate = 0.1  #! Initial learning rate
discount_factor = 0.99  #! Discount factor for future rewards
epsilon = 1.0  #! Initial exploration rate
epsilon_decay = 0.995  #! Decay factor for exploration rate
min_epsilon = 0.01  #! Minimum exploration rate

#! Discretize the continuous state space into discrete bins
num_bins = (50, 50)  #! Number of bins for position and velocity
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bins = [np.linspace(b[0], b[1], num_bins[i] - 1) for i, b in enumerate(state_bounds)]

#! Initialize Q-table with zeros for all states and actions
q_table = np.zeros(num_bins + (env.action_space.n,))

#! Track metrics for analysis
rewards_per_episode = []  #! Store total reward per episode
epsilon_values = []  #! Track epsilon values over episodes

def discretize_state(state):
    """
    Convert a continuous state to discrete indices for the Q-table.

    Args:
        state (numpy.ndarray): Continuous state.

    Returns:
        tuple: Discrete indices corresponding to the state.
    """
    return tuple(np.digitize(state[i], state_bins[i]) for i in range(len(state)))

#! Training loop
for episode in range(num_episodes):
    #! Reset environment and initialize state
    state, _ = env.reset()
    state = discretize_state(state)
    total_reward = 0

    for _ in range(200):  #! Limit each episode to 200 steps
        #! Select action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  #! Explore random action
        else:
            action = np.argmax(q_table[state])  #! Exploit best known action

        #! Execute action in environment and observe results
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_state(next_state)

        #! Update Q-value using the Q-learning update rule
        best_next_action = np.argmax(q_table[next_state])  #! Best action for the next state
        q_table[state][action] += learning_rate * (
            reward + discount_factor * q_table[next_state][best_next_action] - q_table[state][action]
        )

        state = next_state  #! Transition to the next state
        total_reward += reward
        if terminated or truncated:  #! End episode if termination conditions are met
            break

    #! Track metrics for monitoring training progress
    rewards_per_episode.append(total_reward)
    epsilon_values.append(epsilon)

    #! Decay exploration rate (epsilon) after each episode
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    #! Optionally decay learning rate for stability
    learning_rate = max(0.01, learning_rate * 0.995)

    #! Print progress every 10,000 episodes
    if (episode + 1) % 10000 == 0:
        avg_reward = np.mean(rewards_per_episode[-10000:])  #! Average reward over last 10,000 episodes
        print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

env.close()  #! Close the environment after training is complete

#! Save the Q-table to a file
q_table_path = os.path.join("ModelData", "q_table.pkl")
os.makedirs("MountainCar", exist_ok=True)  #! Ensure the save directory exists
with open(q_table_path, "wb") as f:
    pickle.dump(q_table, f)

#! Save training metrics for analysis
metrics_path = os.path.join("ModelData", "training_metrics.pkl")
with open(metrics_path, "wb") as f:
    pickle.dump((rewards_per_episode, epsilon_values), f)

print("Training complete. Metrics and Q-table saved.")