import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym

env = gym.make('CartPole-v0')
env.reset()

states = env.observation_space.shape[0]
actions = env.action_space.n
# print(states)
# print(actions)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.net1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.net2 = nn.Linear(128, 2)

        self.rewards = []

    def forward(self, x):
        x = self.net1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.net2(x)


        return F.softmax(action_scores, dim=1)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
# print(policy)
# print(policy.parameters())

# 根据环境来采取动作
def select_action(state):

    print()

# 采集一条轨迹
def run_episode():
    print('')