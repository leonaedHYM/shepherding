
#import pylab
import numpy as np
from maxent import maxent_irl
from environment import Environment
from parameters import *
import pygame
import time
import argparse
import random
import json
import matplotlib.pyplot as plt

expert_dim = FIELD_LENGTH #100
n_actions = 4
q_table = np.zeros((expert_dim,expert_dim,expert_dim,expert_dim,n_actions))
gamma = 0.99
q_learning_rate = 0.03
theta_learning_rate = 0.05
n_episode = 20


def expert_feature_expectations(demonstrations):
    expert = np.zeros((expert_dim,expert_dim,expert_dim,expert_dim))
    for demonstration in demonstrations: 
        for pair in demonstration: #pair[0] is states.
            temp = np.zeros((expert_dim,expert_dim,expert_dim,expert_dim))
            temp[pair[0][0]][pair[0][1]][pair[0][2]][pair[0][3]] = 1  #100*100*100*100*100*100
            expert += temp
    expert /= len(demonstrations)
    return expert

def update_q_table(state, action, reward, next_state):
    q_1 = q_table[state[0]][state[1]][state[2]][state[3]][action]
    q_2 = reward + gamma * max(q_table[next_state[0]][next_state[1]][next_state[2]][next_state[3]]) # q[state][action]'s target value
    q_table[state[0]][state[1]][state[2]][state[3]][action] += q_learning_rate * (q_2 - q_1) # update the q value

def main():
    env = Environment(RENDER)
    timer = time.time()  
    
    #demonstrations = np.load(file="expert_demo/expert_demo.npy")
    with open('test.txt', 'r') as f:  # demonstration 目前的局限性
        demonstrations = json.loads(f.read())
    #print(demonstrations)
    expert = expert_feature_expectations(demonstrations)
    
    learner_feature_expectations = np.zeros((expert_dim,expert_dim,expert_dim,expert_dim))

    theta = -(np.random.uniform(size=(FIELD_LENGTH,FIELD_LENGTH,FIELD_LENGTH,FIELD_LENGTH))) # a uniformly random distribution between 0~1 with length=400

    stepss, scores = [], []
    grades = 0
    for episode in range(n_episode):
        env.reset() #随即处于一个state
        state = env.get_state()
        score = 0
        steps = 0

        #if (episode != 0 and episode == 1) or (episode > 100 and episode % 50 == 0):
        #if (episode != 0 and episode == 100) or (episode > 100 and episode % 50 == 0):
        if episode != 0:
            learner = learner_feature_expectations / episode #对episode 求平均值，算出每个state出现的概率
            maxent_irl(expert, learner, theta, theta_learning_rate) #更新theta
                
        while True:
            #state_idx = idx_state(env, state)

            action = np.argmax(q_table[state[0]][state[1]][state[2]][state[3]])  
            reward, game_over, max_agent_dist, average_agent_dist = env.step(action)
            
            irl_reward = theta
            next_state = env.get_state()
            update_q_table(state, action, irl_reward[state[0]][state[1]][state[2]][state[3]], next_state)
            # 按照IRL来计算q-table里的值
            learner_feature_expectations[state[0]][state[1]][state[2]][state[3]] += 1 #长度400的向量, 统计一共有几个这种state
        
            score += reward # 每一步的 reward, 一个负数。
            state = next_state
            steps +=1
            
            if game_over:  # 完成一次是一个episode
                grades += 1
                scores.append(score) # score是每次完成时，每一步reward之和，越靠近0说明效果越好。
                stepss.append(steps) # 每次需要多少个episode才能成功。
                print('succeed')
                break

    np.save('q_table.npy', q_table)

    print(scores)
    print(stepss)

    filename = open('results/scores20.txt', 'w')
    for score in scores:
        filename.write(str(score))
    filename.close()
        
    filename = open('results/stepss20.txt', 'w')
    for steps in stepss:
        filename.write(str(steps))
    filename.close()
'''
    fig = 
    x1 = len(scores)
    x2 = len(episodes)
    plt.plot(x1, scores)
    plt.plot(x2, episodes)
    plt.show()
'''
'''
        if episode % 1 == 0:
        #if episode % 10 == 0:
            print('successful times', grades)
            score_avg = np.mean(scores) # 
            print('{} episode score is {:.2f}'.format(episode, score_avg))
            #pylab.plot(episodes, scores, 'b')
            #pylab.savefig("./learning_curves/maxent_3000.png")
            #np.save("./results/maxent_q_table", arr=q_table)
'''

if __name__ == '__main__':
    main()