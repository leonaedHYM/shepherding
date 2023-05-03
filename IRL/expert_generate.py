import readchar
import numpy as np
from collections import deque
import random
#from environment import Environment
from multi_agent_environment import multi_Environment
from parameters import *
import torch as T
from torchsummary import summary
import time
import argparse
import pygame
import json

# actions
right = 0
up = 3
left = 2
down = 1

#agent_num = 3

# key mapping
arrow_keys1 = {
    'w': up,
    's': down,
    'd': right,
    'a': left   }
arrow_keys2 = {
     'i': up,
     'k': down,
     'l': right,
     'j': left   }

env = multi_Environment(RENDER)    

trajectories = list()
episode_step = 0

for episode in range(10):
    trajectory = list()
    step=0
    env.reset()
    num_agents = env.num_agents
    print("episode_step", episode_step)
    while True:
        env.render()
        print("step", step)
        # if multiple keyboard input?
        key1 = readchar.readkey()

        if key1 not in arrow_keys1.keys():
            break
        action1 = arrow_keys1[key1]

        key2 = readchar.readkey()
        if key2 not in arrow_keys2.keys():
            break

        action2= arrow_keys2[key2]
        action = [action1, action2]
        #action = [action1]
        reward, game_over, max_agent_dist, average_agent_dist = env.step(action)
        state = env.get_state()
        max_dist = max([env.dist(r, env.target) for r in env.agents])
        if max_dist <= TARGET_RADIUS:
            print(len(trajectory))
            break
        trajectory.append((state.tolist(),action))
        step +=1
    # trajectory_numpy = np.array(trajectory, float)
    # print("trajectory_numpy.shape", trajectory_numpy.shape)
    episode_step += 1
    trajectories.append(trajectory)

#np_trajectories = np.array(trajectories, float)
print("np_trajectories.shape", len(trajectories))

#np.save("expert_trajectories", arr=np_trajectories)
#with open('temp.txt', 'w') as f:
#    f.write(json.dumps(trajectories))

