import numpy as np
import gym
from collections import deque
import grid_world

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim

from IPython.display import clear_output
import matplotlib.pyplot as plt

epsilon = 0.1
gamma = 0.9

env = gym.make('PuddleWorld-v1')
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen = max_size)
    def add(self,exp):
        self.buffer.append(exp)
    def sample(self, batch_size):
        arr = np.random.choice(range(len(self.buffer)),size=batch_size,replace=False)
        #Replace=False ensures that the values are unique
        return [self.buffer[i] for i in arr]
    def length(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,n_actions)
        )
    def forward(self, x):
        return self.layers(x)
    def act(self, state):
        if np.random.random() < epsilon:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
            q_value = self.forward(state)
            a = q_value.max(1)[1].item()
            print(state, a)
        return a

model = DQN(obs_size, n_actions)
optimizer = optim.Adam(model.parameters())

def compute_loss(batch_size):
    batch = memory.sample(batch_size)
    s=[]
    a=[]
    r=[]
    n_s=[]
    done = []
    for j in batch:
        s.append(j[0])
        a.append(j[1])
        r.append(j[2])
        n_s.append(j[3])
        done.append(j[4])
    s = torch.FloatTensor(np.float32(s))
    with torch.no_grad():
        n_s = torch.FloatTensor(np.float32(n_s))
    a = torch.LongTensor(a)
    #LongTensor because gather() takes LongTensor as argument
    r = torch.FloatTensor(r)
    done = torch.FloatTensor(done)
    q_value = model(s).gather(1,a.unsqueeze(1)).squeeze(1)
    next_q_max = model(n_s).max(1)[0]
    #[0]:Values and [1] gives indices
    target = r + gamma*next_q_max*(1-done)
    loss = (q_value - target).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

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
memory = Memory()
batch_size = 32
re = 0
losses = []
all_rewards = []
for epi in range(100000):
    a = model.act(s)
    n_s, r, done = env.step(a)
    memory.add((s, a, r, n_s, done))
    re+=r
    if memory.length()>batch_size:
        loss = compute_loss(batch_size)
        print(loss)
        losses.append(loss.item())
    if done:
        s = env.reset()
        all_rewards.append(re)
        re=0
    else:
        s = n_s
    if not epi%500:
        plot(epi, all_rewards, losses)
