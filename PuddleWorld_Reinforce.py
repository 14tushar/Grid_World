import numpy as np
import gym
#from collections import deque
import grid_world

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from IPython.display import clear_output
import matplotlib.pyplot as plt

epsilon = 0.1
gamma = 0.9

env = gym.make('PuddleWorld-v1')
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

class PolicyNN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PolicyNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape,32),
            nn.Dropout(p=0.6),
            nn.Linear(32,n_actions)
        )
    def forward(self, x):
        score = self.layers(x)
        return F.softmax(score, dim=1)

model = PolicyNN(obs_size, n_actions)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

def plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()

s = env.reset()
re = 0
losses = []
all_rewards = []
eps = np.finfo(np.float32).eps.item()

for epi in range(1000):
    policy_loss = []
    returns = []
    rewards = []
    log_probs = []
    done = False
    step=0
    re=0
    while (not done) and (step < 100000):
        state = torch.FloatTensor(s).unsqueeze(0)
        probs = model(state)
        m = Categorical(probs)
        a = m.sample()
        n_s, r, done = env.step(a.item())
        log_prob = m.log_prob(a)
        log_probs.append(log_prob)
        rewards.append(r)
        s = n_s
        re+=r
        step+=1
    print(re)
    R=0
    for i in rewards[::-1]:
        R = i + gamma*R
        returns.insert(0,R)
    returns = torch.tensor(returns)
    returns = (returns-returns.mean())/(returns.std() + eps)
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob*R)
    loss = torch.cat(policy_loss).sum()
    losses.append(loss)
    all_rewards.append(re)
    print(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    s = env.reset()
    if not epi%100:
        plot(epi, all_rewards, losses)
