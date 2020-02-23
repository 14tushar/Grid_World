import gym
import grid_world
import numpy as np

from collections import defaultdict as dd
q = dd(float)
gamma = 0.9
alpha = 0.4
epsilon = 0.1

from IPython.display import clear_output
import matplotlib.pyplot as plt

x = input("Variant Number: ")
env = gym.make('PuddleWorld-v'+str(x))
actions = range(env.action_space.n)

def Q_update(s, r, a, s_next, done):
    if done:
        max_q_next = 0
    else:
        max_q_next = max([q[s_next,a] for a in actions])
    q[s,a] += alpha*(r + gamma*max_q_next - q[s,a])

def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('Episodes %s. Mean reward (last 10): %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.show()

st = env.reset()
rewards = []
reward = 0.0
act_per = []
epi=0
while(epi < 10000):
    epsilon /= 5
    if np.random.random() < epsilon:
        a = env.action_space.sample()
    else:
        q_st = [q[st,a] for a in actions]
        q_max = max(q_st)
        a = np.random.choice([i for i,j in enumerate(q_st) if j==q_max])
    act_per.append(a)
    st_next, r, done = env.step(a)
    Q_update(st, r, a, st_next, done)
    reward+=r
    if done:
        act_per = []
        rewards.append(reward)
        reward = 0.0
        st = env.reset()
        epi+=1
    else:
        st = st_next
plot(epi,rewards)
