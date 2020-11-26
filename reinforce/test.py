import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.utils as utils
import gym
import os

from my_reinforce import REINFORCE

env = gym.make('CartPole-v1')
env.reset()

hidden_size = 128
states = env.observation_space.shape[0]
actions = env.action_space.n

agent = REINFORCE(hidden_size, states, actions)

for i_episode in range(1000):
    state = torch.Tensor([env.reset()])
    entropies = []
    log_probs = []
    rewards = []
    for t in range(1000):
        action, log_prob, entropy = agent.select_action(state)
        action = action.cpu()

        next_state, reward ,done = env.step(action.numpy()[0])

        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)
        state = torch.Tensor([next_state])

        if done:
            break

    agent.update_policy_params(rewards, log_probs, entropies, gamma=0.001)

    if i_episode % 100 == 0:
        torch.save(agent.model.state_dict(), os.path.join(dir, 'reinforce-'+str(i_episode)+'pkl'))
    print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))
env.close()
