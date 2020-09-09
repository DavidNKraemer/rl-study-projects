import torch
import utils
import gym

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
from torch import optim
from torch.autograd import Variable


def qfun(env):
    states = env.observation_space.shape[0]
    actions = env.action_space.n

    return nn.Sequential(
        nn.Linear(states, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, actions)
    )


class EpsilonGreedyPolicy:

    def __init__(self, epsilon, qfun, decay=0.9):
        self.epsilon = epsilon
        self.qfun = qfun
        self.decay = decay

    def decay(self):
        self.epsilon *= self.decay

    def __call__(self, state):
        state = torch.from_numpy(np.asarray(state, dtype=np.float32))
        if np.random.rand() <= self.epsilon:
            return np.random.choice(range(self.qfun[-1].out_features))
        else:
            return np.argmax(self.qfun(state).detach().numpy())
        

class OnPolicyReplay:

    def __init__(self):
        self.keys = ['state', 'action', 'next_state', 'reward', 'done']

    def reset(self):
        self.data = defaultdict(list)

    def add(self, state, action, next_state, reward, done, info):
        collected = (state, action, next_state, reward, done)
        for key, val in zip(self.keys, collected):
            self.data[key].append(val)

    def sample(self, size):
        print(range(len(self)))
        indices = np.random.choice(range(len(self)), size=size)

        return {
            key: np.array(self.data[key], dtype=np.float32)[indices].astype(np.float32) \
            for key in self.keys
        }

    def __len__(self):
        return len(self.data[self.keys[0]])


class Sarsa:

    def __init__(self, env, policy_epsilon=1e-1, sim_steps=100, sample_size=50):
        self.env = env
        self.memory = OnPolicyReplay()
        self.policy_epsilon = policy_epsilon
        self.sim_steps = sim_steps
        self.sample_size = sample_size
        self.device = torch.device('cpu')

    def reset(self):
        self.env.reset()
        self.memory.reset()
        self.qfun = qfun(self.env)
        self.qloss = nn.MSELoss(reduction='sum')

        self.policy = EpsilonGreedyPolicy(self.policy_epsilon, self.qfun)

    def simulate_env(self):
        for step in range(self.sim_steps):
            state = np.asarray(self.env.state)
            action = self.policy(state)
            self.memory.add(state, action, *self.env.step(action))

    def update_q(self, data):
        q_preds = self.qfun(data['state'])
        next_q_preds = self.qfun(data['next_state'])

    def sample(self):
        batch = self.memory.sample(self.sample_size)

        for key in batch.keys():
            batch[key] = torch.from_numpy(batch[key]).to(self.device)

        return batch

    def train(self):
        # run the environment for a while
        self.simulate_env()

        # sample from memory, update Q-function
        self.update_q(self.sample())

        # decay epsilon
        self.policy.decay()


agent = Sarsa(gym.make('CartPole-v0'))
agent.reset()
agent.simulate_env()
