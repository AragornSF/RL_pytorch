import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# Hyper Parameters for Actor
GAMMA = 0.95  # discount factor
LR = 0.01  # learning rate

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False  # 非确定性算法


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


class Actor(object):
    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # 创建网络
        self.network = ActorNetwork(hidden_size=20, state_dim=self.state_dim, action_dim=self.action_dim)
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


# Hyper Parameters for Critic
EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch
REPLACE_TARGET_FREQ = 10  # frequency to update target Q network


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, 1)   # 这个地方和之前略有区别，输出不是动作维度，而是一维

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)


class Critic(object):
    def __init__(self, env):
        # 状态空间和动作空间的维度
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # init network parameters
        self.network = QNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

        # init some parameters
        self.time_step = 0
        self.epsilon = EPSILON  # epsilon值是随机不断变小的

    def train_Q_network(self, state, reward, next_state):
        s, s_ = torch.FloatTensor(state).to(device), torch.FloatTensor(next_state).to(device)
        # 前向传播
        v = self.network.forward(s)     # v(s)
        v_ = self.network.forward(s_)   # v(s')

        # 反向传播
        loss_q = self.loss_func(reward + GAMMA * v_, v)
        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()

        with torch.no_grad():
            td_error = reward + GAMMA * v_ - v

        return td_error

'''
class Critic(object):
    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.network = CriticNetwork(hidden_size=20, state_dim=self.state_dim, action_dim=self.action_dim)
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
'''

# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 3000  # Episode limitation
STEP = 3000  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    actor = Actor(env)
    critic = Critic(env)

    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        # Train
        for step in range(STEP):
            action = actor.choose_action(state)  # SoftMax概率选择action
            next_state, reward, done, _ = env.step(action)
            td_error = critic.train_Q_network(state, reward, next_state)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(state, action, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            state = next_state
            if done:
                break

        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = actor.choose_action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('Total time is ', time_end - time_start, 's')