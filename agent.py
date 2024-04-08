import torch as T
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import deque

import random

class BaseAgent:
    '''
    This agent acts like a mixin, storing common constant values and helping define some utility functions
    '''
    def __init__(
            self, 
            batch_size = 64,
            gamma = 0.99, 
            epsilon = 1.0, 
            epsilon_min = 0.01, 
            epsilon_decay = 0.995, 
            target_network_update_interval = 1000
    ) -> None:
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_network_update_interval = target_network_update_interval

    def decay_epsilon(self):
        # decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class DDQNAgent(BaseAgent):
    def __init__(self, device, replay_buffer, online_network, target_network, optimizer, algo_type="type_1", **kwargs) -> None:
        super().__init__(**kwargs)

        self.device = device
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.online_network = online_network
        self.target_network = target_network

        self.learning_step_couter = 0

        if hasattr(self.online_network, f"forward_{algo_type}"):
            self.forward_attribute = getattr(self.online_network, f"forward_{algo_type}")
        else:
            print("Using the default forward method of inputNN")
            self.forward_attribute = getattr(self.online_network, "forward")
        

    def act(self, state):
        if random.random() > self.epsilon:
            state = T.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.online_network.eval()
            with T.no_grad():
                action_values = self.forward_attribute(state)
                
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.online_network.n_actions))
        
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            # only start learning when we have enough samples in memory
            return
        
        # print("actual learning")
        self.optimizer.zero_grad()

        # try updating the target network
        if self.learning_step_couter % self.target_network_update_interval == 0:
            self.update_target_network()

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # computing the expected Q value
        # with T.no_grad():
        Q_expected = self.online_network(states).gather(1, actions)

        # computing the target Q value
        Q_target_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)

        # Q_target_next = self.compute_q_value(V_target, A_target).detach().max(1)[0].unsqueeze(1)
        Q_target = rewards + (self.gamma * Q_target_next * (1 - dones))

        loss = F.mse_loss(Q_target, Q_expected).to(self.device)

        # print(f"The loss: {loss} because {T.mean(rewards)} for {T.mean(Q_target)} against {T.mean(Q_eval)}")
        loss.backward()

        ''' Gradiant Clipping '''
        for param in self.online_network.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        self.learning_step_couter += 1


class REINFORCEAgent(BaseAgent):
    def __init__(self, device, policy_network, optimizer, value_network=None, value_optimizer=None, max_t=1000, **kwargs) -> None:
        super().__init__(**kwargs)

        self.device = device
        self.max_t = max_t
        self.optimizer = optimizer
        self.policy_network = policy_network
        self.value_network = value_network
        self.value_optimizer = value_optimizer

    def act(self, state):
        state = T.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy_network.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def learn(self, saved_log_probs, rewards, states = []):
        returns = deque(maxlen=self.max_t)
        n_steps = len(rewards)

        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(self.gamma * disc_return_t + rewards[t])

        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = T.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = T.cat(policy_loss).sum()

        value_loss = []
        for disc_return, state in zip(returns, states):
            state = T.from_numpy(state).float().unsqueeze(0)
            value = self.value_network(state).squeeze()
            value_loss.append(F.mse_loss(value, disc_return.clone().detach()))

        # Line 8: PyTorch prefers gradient descent
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        if self.value_optimizer:
            self.value_optimizer.zero_grad()
            value_loss = T.stack(value_loss).sum()
            value_loss.backward()
            self.value_optimizer.step()