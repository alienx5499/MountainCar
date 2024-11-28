import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def load_metrics(file_path):
    """
    Load training metrics from a file.

    Args:
        file_path (str): Path to the metrics file.

    Returns:
        tuple: A tuple containing rewards and epsilon values.
    """
    #! Load metrics like rewards and epsilon values from the specified file
    with open(file_path, "rb") as f:
        return pickle.load(f)

def visualize_rewards(rewards, window_size=100):
    """
    Plot the total rewards per episode and their moving average.

    Args:
        rewards (list): Total rewards for each episode.
        window_size (int): Window size for the moving average.
    """
    #! Calculate the moving average of rewards
    moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")

    #! Plot total rewards and the moving average
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label="Total Reward", alpha=0.5)
    plt.plot(range(len(moving_avg)), moving_avg, label=f"Moving Average (Window={window_size})", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.legend()
    plt.grid()
    plt.show()

def visualize_epsilon(epsilon_values):
    """
    Plot the epsilon decay over episodes.

    Args:
        epsilon_values (list): Epsilon values per episode.
    """
    #! Plot the epsilon values to visualize exploration decay
    plt.figure(figsize=(12, 6))
    plt.plot(epsilon_values, label="Epsilon", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay Over Episodes")
    plt.legend()
    plt.grid()
    plt.show()

def analyze_q_table(q_table):
    """
    Analyze the Q-table by visualizing its maximum Q-values.

    Args:
        q_table (numpy.ndarray): The Q-table from training.
    """
    #! Visualize the distribution of maximum Q-values for all states
    max_q_values = np.max(q_table, axis=-1)  #! Find the maximum Q-value for each state
    plt.figure(figsize=(12, 6))
    plt.hist(max_q_values.flatten(), bins=50, color="blue", alpha=0.7)
    plt.title("Distribution of Maximum Q-Values")
    plt.xlabel("Max Q-Value")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    #! Define file paths for training metrics and Q-table
    metrics_file = os.path.join("MountainCar", "training_metrics.pkl")
    q_table_file = os.path.join("MountainCar", "q_table.pkl")

    #! Load training metrics
    try:
        rewards, epsilon_values = load_metrics(metrics_file)
        print("Loaded training metrics successfully.")
    except FileNotFoundError:
        print(f"Error: Could not find the file '{metrics_file}'. Please ensure it exists.")
        rewards, epsilon_values = None, None

    #! Load Q-table
    try:
        with open(q_table_file, "rb") as f:
            q_table = pickle.load(f)
            print("Loaded Q-table successfully.")
    except FileNotFoundError:
        print(f"Error: Could not find the file '{q_table_file}'. Please ensure it exists.")
        q_table = None

    #! Visualize training metrics
    if rewards is not None:
        visualize_rewards(rewards, window_size=100)  #! Visualize rewards with moving average
    if epsilon_values is not None:
        visualize_epsilon(epsilon_values)  #! Visualize epsilon decay

    #! Analyze the Q-table
    if q_table is not None:
        analyze_q_table(q_table)  #! Analyze and visualize Q-table properties