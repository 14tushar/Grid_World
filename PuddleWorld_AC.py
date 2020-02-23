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

class ACNN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ACNN, self).__init__()
        self.layer1 = nn.Linear(input_shape,32)
        self.actor_head = nn.Linear(32, n_actions) #For_actor
        self.value_head = nn.Linear(32, 1) #For_critic
    def forward(self, x):
        x = F.relu(self.layer1(x))
        action_prob = F.softmax(self.actor_head(x))
        V_state = self.value_head(x)
        return action_prob, V_state

model = ACNN(obs_size, n_actions)
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
    policy_losses = []
    value_losses = []
    returns = []
    rewards = []
    log_probs = []
    done = False
    step=0
    re=0
    while (not done) and (step < 1000):
        state = torch.FloatTensor(s).unsqueeze(0)
        probs,V_s = model(state)
        m = Categorical(probs)
        a = m.sample()
        n_s, r, done = env.step(a.item())
        log_prob = m.log_prob(a)
        log_probs.append((log_prob, V_s))
        rewards.append(r)
        s = n_s
        re+=r
        step+=1
    R=0
    for i in rewards[::-1]:
        R = i + gamma*R
        returns.insert(0,R)
    returns = torch.tensor(returns)
    returns = (returns-returns.mean())/(returns.std() + eps)
    for (log_prob,V_s), R in zip(log_probs, returns):
        advantage = R - V_s.item()
        policy_losses.append(-log_prob*advantage)
        value_losses.append(F.smooth_l1_loss(V_s, torch.tensor([R])))
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    losses.append(loss)
    all_rewards.append(re)
    print(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    s = env.reset()
    if not epi%100:
        plot(epi, all_rewards, losses)
