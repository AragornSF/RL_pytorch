import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import time

'''
    AC:actor-critic算法。
    actor为策略网络，负责根据策略选择动作，根据critic网络生成的advantage函数对本网络的策略参数进行梯度上升更新，
    进一步提高所获得的总奖励。
    critic为评价网络，负责生成值函数，计算advantage函数，使用这个函数对策略网络进行更新，同时对自己网络中的φ参数
    进行梯度下降更新，更进一步接近真实的值函数。
'''

# hyperparameters
GAMMA = 0.95
LR = 0.01
EPSILON = 0.01
REPLACY_SIZE = 10000
BATCH_SIZE = 32
REPLACE_TARGET_FREQ = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False

# actor网络
class ActorNetwork(nn.Module):
    def __init__(self, hidden_size, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = self.layer2(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)

# critic网络
class CriticNetwork(nn.Module):
    def __init__(self, hidden_size, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = self.layer2(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)

class Actor(object):
    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # 创建网络
        self.network = ActorNetwork(hidden_size=128, state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)

        self.time_step = 0

    # 选择动作
    def choose_action(self, state):
        state = torch.FloatTensor(state).to(device)
        network_output = self.network.forward(state)
        with torch.no_grad():
            prob_weights = F.softmax(network_output, dim=0).cuda().data.cpu().numpy()
        action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights)
        return action

    # 更新参数
    def update_params(self, state, action, advantage):
        self.time_step += 1
        softmax_input = self.network.forward(torch.FloatTensor(state).to(device)).unsqueeze(0)
        action = torch.LongTensor([action]).to(device)
        neg_log_prob = F.cross_entropy(input=softmax_input, target=action, reduction='none')

        loss_action = -neg_log_prob * advantage
        self.optimizer.zero_grad()
        loss_action.backward()
        self.optimizer.step()

class Critic(object):
    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.network = CriticNetwork(hidden_size=128, state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)
        self.loss_function = nn.MSELoss()

        self.time_step = 0
        self.epsilon = EPSILON

    def train_Ctitic_network(self, state, reward, next_state):
        s, s_ = torch.FloatTensor(state).to(device), torch.FloatTensor(next_state).to(device)
        v = self.network.forward(s)
        v_ = self.network.forward(s_)

        loss_value = self.loss_function(reward + GAMMA * v_, v)
        self.optimizer.zero_grad()
        loss_value.backward()
        self.optimizer.step()

        with torch.no_grad():
            advantage = reward + GAMMA * v_ - v
        return advantage

ENV_NAME = 'FrozenLake-v0'
EPISODE = 30000
STEP = 30000
TEST = 100

def main():

    env = gym.make(ENV_NAME)
    actor = Actor(env)
    critic = Critic(env)

    for episode in range(EPISODE):
        state = env.reset()
        for step in range(STEP):
            action = actor.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            advantage = critic.train_Ctitic_network(state, reward, next_state)
            actor.update_params(state, action, advantage)
            state = next_state
            if done:
                break

        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = actor.choose_action(state)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode + 100, 'Evaluation Average Reward: ', ave_reward, 'Total_reward: ', total_reward)

if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('Total Time is: ', time_end - time_start)
