import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.autograd import Variable

'''
    策略类 undo
    是一个神经网络，负责根据环境的状态选取动作，目标是对网络参数进行梯度上升更新，使得目标函数获得最大值
    目标函数：
'''
class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        p = F.softmax(self.linear2(x), -1)
        return p

'''
    reinforce算法 undo
    蒙特卡罗算法，采样一整条轨迹，然后计算在这条轨迹上获得的总奖励，使用这个奖励对目标函数进行更新。
    采集多条轨迹，取平均值。
'''
class REINFORCE():
    def __init__(self, hidden_size, num_inputs, num_outpus):
        self.model = Policy(hidden_size, num_inputs, num_outpus)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()

    # 选择动作
    def select_action(self, state):
        probs = self.model(Variable(state).cuda())
        action = probs.multinomial().data
        prob = probs[:, action[0, 0].view(1, -1)]
        log_prob = prob.log()
        entropy = -(probs*probs.log()).sum()

        return action[0], log_prob, entropy

    # 更新策略网络的网络参数
    def update_policy_params(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            loss = loss - (log_probs[i] * (Variable(R).expand_as(log_probs[i])).cuda()).sum() - (0.0001 * entropies[i].cuda()).sum()
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()
