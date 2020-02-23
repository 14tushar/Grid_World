import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

'''
    Observation:
        x: 0 to 11
        y: 0 to 11

    Action:
        0: North
        1: East
        2: West
        3: South

    Reward:
    +10, -3, -2, -1
'''

class PuddleWorldv2(gym.Env):
    metadata = {'render.modes':['human']}
    def __init__(self):
        low = np.array([0,0])
        high = np.array([11,11])
        self.observation_space = spaces.Box(low,high,dtype=np.int)
        self.action_space = spaces.Discrete(4)
    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"
        state = self.state
        if np.random.random() >= 0.9:
            action_spc = [0,1,2,3]
            action_spc.remove(action)
            action = np.random.choice(action_spc)
        if action==0:#North
            state = state[0],min(11,state[1]+1)
        elif action==1:#East
            state = min(11,state[0]+1),state[1]
        elif action==2:#South
            state = state[0],max(0,state[1]-1)
        elif action==3:#West
            state = max(0,state[0]-1),state[1]
        else:
            print("Invalid Action")
        if np.random.random() < 0.5:
            state = min(11,state[0]+1),state[1]
        self.state = state
        if state[0]==3:
            if 2<state[1]<10:
                reward = -1
            else:
                reward = 0
        elif state[0]==4:
            if state[1]==3 or state[1]==9:
                reward = -1
            elif 3<state[1]<9:
                reward = -2
            else:
                reward = 0
        elif state[0]==5:
            if state[1]==3 or state[1]==9:
                reward = -1
            elif state[1]==4 or state[1]==8:
                reward = -2
            elif 4<state[1]<8:
                reward = -3
            else:
                reward = 0
        elif state[0]==6:
            if state[1]==3 or state[1]==9:
                reward = -1
            elif state[1]==7:
                reward = -3
            elif 3<state[1]<9:
                reward = -2
            else:
                reward = 0
        elif state[0]==7:
            if 5<state[1]<9:
                reward = -2
            elif 2<state[1]<10:
                reward = -1
            else:
                reward = 0
        elif state[0]==8:
            if 4<state[1]<10:
                reward = -1
            else:
                reward = 0
        else:
            reward = 0
        done=False
        if state==(9,9):
            reward=10
            done=True
        return self.state, reward, done

    def reset(self):
        self.state = (0,np.random.choice([0,1,5,6]))
        #np.random.uniform(low, high, size=(n,)) Sample n values randomly in range low to high.
        return self.state
