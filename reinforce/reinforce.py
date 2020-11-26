import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

class REINFORCE():
    agent_name = 'REINFORCE'
    # 初始化
    def __init__(self, config):
        self.policy = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.hyperparameters['learning_rate'])
        self.episode_rewards = []
        self.episode_log_probabilities = []

    # 重置游戏
    def reset_game(self):
        self.state = self.environmet.reset_enironment() # undo
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = None
        self.total_episode_score_so_far = 0
        self.episode_rewards = []
        self.episode_log_probabilities = []
        self.episode_step_number = 0

    def step(self):
        while not self.done:
            self.pick_and_conduct_action_and_save_log_probabilities()
            self.update_next_state_reward_done_and_score() # undo
            self.store_reward()
            if self.time_to_learn():
                self.actor_learn()
            self.state = self.next_state
            self.episode_step_number += 1
        self.episode_step_number += 1

    # 选择并指导动作，并保存对数几率
    def pick_and_conduct_action_and_save_log_probabilities(self):
        action, log_probabilities = self.pick_action_and_get_log_probabilities()
        self.store_log_probabilities()
        self.store_action()
        self.conduct_action()

    # 选择动作并得到对数几率
    def pick_action_and_get_log_probabilities(self):
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        action_probabilities = self.policy.forward(state).cpu()
        action_distribution = Categorical(action_probabilities)
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action)

    # 保存对数几率
    def store_log_probabilities(self, log_probabilities):
        self.episode_log_probabilities.append(log_probabilities)

    # 保存动作
    def store_action(self, action):
        self.action = action

    # 保存奖励
    def store_reward(self):
        self.episode_rewards.append(self.reward)

    # actor学习
    def actor_learn(self):
        total_discounted_reward = self.calculate_episode_discounted_reward()
        policy_loss = self.calculate_policy_loss_on_episode(total_discounted_reward)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    # 计算折扣奖励
    def calculate_episode_discounted_reward(self):
        discounts = self.hyperparameters['discount_rate'] ** np.arange(len(self.episode_rewards))
        total_discounted_reward = np.dot(discounts, self.episode_rewards)
        return total_discounted_reward

    # 计算每一回合的策略损失
    def calculate_policy_loss_on_episode(self, total_discounted_reward):
        policy_loss = []
        for log_prob in self.episode_log_probabilities:
            policy_loss.append(-log_prob * total_discounted_reward)
        policy_loss = torch.cat(policy_loss).sum()
        return policy_loss

    def time_to_learn(self):
        return self.done
