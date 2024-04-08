import random
import torch as T
import numpy as np

from collections import deque, namedtuple

class ReplayBuffer():

    def __init__(self, buffer_size, n_actions, batch_size, device, seed):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.device = device
        self.seed = random.seed(seed)

        self.memory = deque(maxlen = buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __len__(self):
        """Returns the current size of internal memory"""
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """Randomly sample a batch of experience from memory"""
        sampled_experiences = random.sample(self.memory, k=self.batch_size)

        states = T.from_numpy(np.vstack([experience.state for experience in sampled_experiences if experience is not None])).float().to(self.device)
        actions = T.from_numpy(np.vstack([experience.action for experience in sampled_experiences if experience is not None])).long().to(self.device)
        rewards = T.from_numpy(np.vstack([experience.reward for experience in sampled_experiences if experience is not None])).float().to(self.device)
        next_states = T.from_numpy(np.vstack([experience.next_state for experience in sampled_experiences if experience is not None])).float().to(self.device)
        dones = T.from_numpy(np.vstack([experience.done for experience in sampled_experiences if experience is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    
    