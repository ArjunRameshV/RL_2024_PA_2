from rl_networks import DuelingDQN
from agent import DDQNAgent
from replay_buffer import ReplayBuffer

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
env_id = "Acrobot-v1"

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
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}

# Loop over multiple seeds
for seed in seeds:
    print(f"-------------- Starting the learning with seed: {seed} --------------")
    local_network = DuelingDQN(a_size, s_size, seed, device)
    target_nework = DuelingDQN(a_size, s_size, seed, device)
    replay_buffer = ReplayBuffer(int(1e5), a_size, 64, device, seed)
    cartpole_optimizer = optim.Adam(local_network.parameters(), lr=cartpole_hyperparameters["lr"])
    agent = DDQNAgent(device, replay_buffer, local_network, target_nework, cartpole_optimizer, algo_type="type_2")


    begin_time = datetime.datetime.now()
    n_episodes=1000
    max_t=250

    # Initialize list to store rewards for the current seed
    episode_rewards = []

    scores_window = deque(maxlen=100)
    ''' last 100 scores for checking if the avg is more than 195 '''

    for i_episode in range(1, n_episodes+1):
        state = env.reset(seed=seed)[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, trunc, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            agent.learn()
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        episode_rewards.append(score)

        ''' decrease epsilon '''
        agent.decay_epsilon()

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 250 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        if np.mean(scores_window)>=450.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            break

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
with open('acrobot_dueling_dqn_type_2_reward_statistics.txt', 'w') as file:
    file.write("Episode\tMean\tStd Dev\n")
    for episode, mean_reward, std_reward in zip(range(1, n_episodes + 1), mean_rewards, std_rewards):
        file.write(f"{episode}\t{mean_reward}\t{std_reward}\n")

# Plot mean reward with standard deviation error bars
plt.plot(np.arange(1, n_episodes + 1), mean_rewards, color='red', linewidth=2, label='Mean')
plt.fill_between(np.arange(1, n_episodes + 1), mean_rewards - std_rewards, mean_rewards + std_rewards,
                 color='lightblue', alpha=0.5, label='Variance')

plt.xlabel('Episode Number')
plt.ylabel('Episodic Return')
plt.title('Mean Episodic Return with Variance (Dueling DQN)')
plt.legend()
plt.grid(True)
# plt.show()

# Save the plot as an image file
plt.savefig('acrobot_dueling_dqn_type_2.png')
plt.show()