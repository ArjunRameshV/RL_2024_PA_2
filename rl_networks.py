import numpy as np
import random
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.distributions import Categorical

    
class DuelingDQN(nn.Module):
    def __init__(self, n_actions, n_states, seed, device, fc1_units = 128, fc2_units = 64):
        super().__init__()
        self.n_actions = n_actions
        self.n_states = n_states

        self.fc = nn.Linear(n_states, fc1_units)

        self.fc_a = nn.Linear(fc1_units, fc2_units)
        self.a = nn.Linear(fc2_units, n_actions)

        self.fc_v = nn.Linear(fc1_units, fc2_units)
        self.v = nn.Linear(fc2_units, 1)

        # Network attributes
        self.seed = T.manual_seed(seed)
        self.device = device
        self.to(self.device)

    def forward_type_1(self, state):
        # aggregating layer 
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        x = F.relu(self.fc(state))

        a = F.relu(self.fc_a(x))
        a = self.a(a)

        v = F.relu(self.fc_v(x))
        v = self.v(v)

        return v + (a - a.mean(dim=1, keepdim=True))
    
    def forward_type_2(self, state):
        # aggregating layer 
        # Q(s,a) = V(s) + (A(s,a) - max[A(s,a')])
        x = F.relu(self.fc(state))

        a = F.relu(self.fc_a(x))
        a = self.a(a)

        v = F.relu(self.fc_v(x))
        v = self.v(v)

        return v + (a - T.max(a))

    def forward(self, state):
        # keeping type 1 as default
        return self.forward_type_1(state)

class REINFORCE(nn.Module):
    def __init__(self, s_size, a_size, h_size, seed, device):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

        self.seed = T.manual_seed(seed)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x + 1e-8  # Add a small epsilon for learning
        return F.softmax(x, dim=1)

# Define the value function network
class REINFORCEValueNetwork(nn.Module):
    def __init__(self, s_size, seed, device, h_size=128):
        super().__init__()
        self.fc1 = nn.Linear(s_size, 128)
        self.fc2 = nn.Linear(128, 1)

        self.seed = T.manual_seed(seed)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x