from rl_networks import REINFORCE, REINFORCEValueNetwork
from agent import REINFORCEAgent

import numpy as np

from collections import deque

import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.optim as optim

# Gym
import gymnasium as gym
import random
import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
seeds = [5, 10, 15, 20, 25]
max_timesteps = 1000

# Initialize list to store rewards for each seed
all_rewards = []
env_id = "CartPole-v1"

# Create the env
env = gym.make(env_id)

# Get the state space and action space
s_size = env.observation_space.shape[0]
a_size = env.action_space.n

cartpole_hyperparameters = {
    "h_size": 16,
    "n_training_episodes": 1000,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 1e-2,
    "vfn_lr": 1e-1,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}

# Loop over multiple seeds
for seed in seeds:
    print(f"-------------- Starting the learning with seed: {seed} --------------")
    cartpole_policy = REINFORCE(
        cartpole_hyperparameters["state_space"],
        cartpole_hyperparameters["action_space"],
        cartpole_hyperparameters["h_size"],
        seed,
        device
    )
    cartpole_value_fn = REINFORCEValueNetwork(
        cartpole_hyperparameters["state_space"],
        seed,
        device
    )
    cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])
    cartpole_value_fn_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["vfn_lr"])

    agent = REINFORCEAgent(device, cartpole_policy, cartpole_optimizer, cartpole_value_fn, cartpole_value_fn_optimizer)

    begin_time = datetime.datetime.now()
    n_episodes=5000
    max_t=500

    # Initialize list to store rewards for the current seed
    episode_rewards = []

    scores_window = deque(maxlen=100)
    scores = []

    for i_episode in range(1, n_episodes+1):
        saved_log_probs = []
        rewards = []
        states = []
        state = env.reset(seed=seed)[0]
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = agent.act(state)
            saved_log_probs.append(log_prob)
            states.append(state)
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            if done:
                break

        scores_window.append(sum(rewards))
        episode_rewards.append(sum(rewards))
        scores.append(sum(rewards))

        ''' decrease epsilon '''
        agent.learn(saved_log_probs, rewards, states)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 250 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    ''' Trial run to check if algorithm runs and saves the data '''
    time_taken = datetime.datetime.now() - begin_time
    print(f"Time to finish with seed: {seed} = {time_taken}")
    all_rewards.append(episode_rewards)

# Convert rewards to numpy array for easier manipulation
all_rewards = np.array(all_rewards)

# Calculate mean and standard deviation across seeds
mean_rewards = np.mean(all_rewards, axis=0)
std_rewards = np.std(all_rewards, axis=0)

# Write mean and standard deviation to a file
with open('cartpole_reinforce_type_2_reward_statistics.txt', 'w') as file:
    file.write("Episode\tMean\tStd Dev\n")
    for episode, mean_reward, std_reward in zip(range(1, n_episodes + 1), mean_rewards, std_rewards):
        file.write(f"{episode}\t{mean_reward}\t{std_reward}\n")

# Plot mean reward with standard deviation error bars
plt.plot(np.arange(1, n_episodes + 1), mean_rewards, color='red', linewidth=2, label='Mean')
plt.fill_between(np.arange(1, n_episodes + 1), mean_rewards - std_rewards, mean_rewards + std_rewards,
                 color='lightblue', alpha=0.5, label='Variance')

plt.xlabel('Episode Number')
plt.ylabel('Episodic Return')
plt.title('Mean Episodic Return with Variance (REINFORCE)')
plt.legend()
plt.grid(True)
# plt.show()

# Save the plot as an image file
plt.savefig('cartpole_reinforce_type_2.png')
plt.show()