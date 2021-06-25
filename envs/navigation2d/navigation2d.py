import gym
import sys
import os
import time
import copy
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt


class Navigation2d(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=[2], dtype=np.float32)

        self.world_min = 0
        self.world_max = 1
        self.observation_space = spaces.Box(low=self.world_min, high=self.world_max, shape=[2],
                                            dtype=np.float32)
        self.agent_start_state = [0.5, 0.5]
        self.agent_state = copy.deepcopy(self.agent_start_state)
        self.random_start_state = False


    def step(self, action):
        ''' return next observation, reward, finished, success '''
        # action = int(action)
        info = {}
        info['success'] = False
        nxt_agent_state = (self.agent_state[0] + action[0],
                           self.agent_state[1] + action[1])

        if nxt_agent_state[0] < self.world_min or nxt_agent_state[0] >= self.world_max:
            info['success'] = False
            return (self.agent_state, 0, False, info)
        if nxt_agent_state[1] < self.world_min or nxt_agent_state[1] >= self.world_max:
            info['success'] = False
            return (self.agent_state, 0, False, info)

        self.agent_state = nxt_agent_state
        info['success'] = True
        return (self.agent_state, 0, False, info)


    def reset(self):
        if self.random_start_state:
            self.agent_state = self._get_random_state()
        else:
            self.agent_state = copy.deepcopy(self.agent_start_state)
        return self.agent_state

    def get_agent_state(self):
        ''' get current agent state '''
        return self.agent_state

    def get_start_state(self):
        ''' get current start state '''
        return self.agent_start_state

    def set_start_state(self, start_state):
        self.agent_start_state = start_state

    def set_random_start_state(self, random):
        self.random_start_state = random

    def _get_random_state(self):
        return self.observation_space.sample()

    def _close_env(self):
        plt.close(1)
        return
