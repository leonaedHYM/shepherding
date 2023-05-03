import numpy as np
from collections import deque
import random
#from multi_agent_environment import multi_Environment
from multi_agent_Env2 import Multi_Environment2
from parameters import *
import torch as T
from torchsummary import summary
import time
import argparse
import pygame
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

if CNN_NETWORK:
    from dqn_cnn import DQN
else:
    from dqn_model import DQN

class DQNAgent:
    def __init__(self, gamma=.99, lr=.03, batch_size=BATCH_SIZE, max_mem_size=100000, 
                eps_start=0.9, eps_end=0.01, eps_decay=200):
        self.episode_num = 0
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.mem_size = max_mem_size
        self.memory = deque(maxlen=self.mem_size)
        if CNN_NETWORK:
            self.input_size = (FIELD_LENGTH, FIELD_LENGTH)
        else:
            self.input_size = 2*(MAX_NUM_AGENTS+NUM_SHEPHERDS)+2
            # self.input_size = 2*MAX_NUM_AGENTS+4  # why input size is this？  each agent's, Shepherd's and target's XY coordinate      
        #self.output_size = 4                      # 4 actions in total
        self.action_size = 4
        self.output_size = 4**NUM_SHEPHERDS
        self.batch_size = batch_size
        self.shepherd_num = NUM_SHEPHERDS
        self.dqn = DQN(lr, self.input_size, self.output_size)                             
        self.target_dqn = DQN(lr, self.input_size, self.output_size)
        self.target_dqn.eval()

    def remember(self, state_old, action, reward, state_new, game_over):
        self.memory.append((state_old, action, reward, state_new, game_over))

    def get_action(self, state):  # output is a batch_size*4*4的矩阵
        # exploration vs exploitation
        eps_threshold = self.eps_end+(self.eps_start-self.eps_end)*\
            np.exp(-1.*self.episode_num/self.eps_decay)
        if random.random() < eps_threshold:
            action = np.array([random.randint(0, self.action_size-1), random.randint(0, self.action_size-1)])  
            #action = T.tensor(action, dtype=T.long).to(self.dqn.device) #
        else:
            state_tensor = T.tensor(np.array(state), dtype=T.float).to(self.dqn.device)
            # add batch dimension
            state_tensor = T.unsqueeze(state_tensor, 0)
            prediction = self.dqn.forward(state_tensor)   # prediction is a [1, 4, 4] 的矩阵
            # output the index pair of chosen action
            #action_index = (prediction[0]==T.max(prediction[0])).nonzero()[0] 
            temp_id = T.argmax(prediction)
            action_index = [temp_id.item()//4, temp_id.item()%4]# action: [dog1's action index, dog2's action index]
            action1 = action_index[0]
            action2 = action_index[1]
            action = np.array([action1, action2])
        return action  # 这是一个tensor

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_old, action, reward, state_new, game_over = zip(*batch)
        state_old = T.tensor(np.array(state_old), dtype=T.float).to(self.dqn.device)
        action = T.tensor(np.array(action), dtype=T.long).to(self.dqn.device)
        reward = T.tensor(np.array(reward), dtype=T.float).to(self.dqn.device)
        state_new = T.tensor(np.array(state_new), dtype=T.float).to(self.dqn.device)
        game_over = T.tensor(np.array(game_over), dtype=T.bool).to(self.dqn.device)
        
        
        outputFq = self.dqn.forward(state_old)           # output is a batch_size*4*4的矩阵
        q = outputFq[range(self.batch_size), action[:,0], action[:,1]] # action is batch_size*2 matrix
        # expected values of actions computed based on the "older" target_dqn
        q_next = self.target_dqn.forward(state_new)      # output is a batch_size*4*4的矩阵
        q_next[game_over] = 0.0   #  game_over 是一个dim=batch_size的vector, 里面每个元素都是True or False. When game_over is True, q_next will become all 0. When game_over is False, q_next will keep the same.

        q_target = reward + self.gamma*T.max(q_next)
        #q_target = reward + self.gamma*T.max(q_next, dim=1)[0] # 取第一个batch里面的max
        loss = self.dqn.loss(q, q_target).to(self.dqn.device)
        
        # gradient descent
        self.dqn.optimizer.zero_grad()
        loss.backward()
        self.dqn.optimizer.step()

    def save(self,num_episode):
        file_name = str(num_episode)+MODEL_PATH
        self.dqn.save(self.episode_num, file_name)

    def load(self, mode, print_model):
        self.episode_num = self.dqn.load(MODEL_PATH)
        self.update_target_dqn()
        if print_model:
            if type(self.input_size) is tuple:
                summary(self.dqn, (1,) + self.input_size)
            else:
                summary(self.dqn, (1, self.input_size))
        if mode == 'train':
            print("Training Mode")
            self.dqn.train()
        elif mode == 'eval':
            print("Evaluation Mode")
            self.dqn.eval()

    def update_target_dqn(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

def train(dqn_agent):
    #env = multi_Environment()
    env = Multi_Environment2(True)  
    timer = time.time()
    reward_memory = []
    num_wins = 0
    all_reward = list()
    mean_reward = list()
    wins_number = list()

    while True:
        score = 0   # := number of frames
        episode_reward = 0
        game_over = False
        state_old = env.get_state()
        while not game_over: #and (not RENDER or env.pygame_running()):
            if RENDER:
                env.render()
            if USER_INPUT:
                action = env.get_key_input()
                env.tick(60)
            else: 
                action = dqn_agent.get_action(state_old)
                
            
            reward, game_over = env.step(action)
            state_new = env.get_state()
            #action = T.tensor(np.array(action), dtype=T.long).to(self.dqn.device)
            dqn_agent.remember(state_old, action, reward, state_new, game_over)
            dqn_agent.learn()
            state_old = state_new
            score += 1
            episode_reward += reward
            # tensorboard session
            # writer.add_scalar("reward/train", episode_reward, dqn_agent.episode_num)
            

            if score >= FRAME_RESET:
                game_over = True
        # drawing plots for episode reward
        all_reward.append(episode_reward)
        reward_memory.append(episode_reward)
        indicator = "L "
        if score < FRAME_RESET:
            indicator = "W "
            num_wins += 1
        if OUTPUT:
            print(indicator + f"Episode {dqn_agent.episode_num}: time={time.time()-timer:.2f}s, " \
                    + f"score={score}, reward={episode_reward: .2f}")
        timer = time.time()
        
        env.reset()
        dqn_agent.episode_num += 1

        # update target network and save every x games
        if dqn_agent.episode_num % SAVE_TARGET == 0:
            dqn_agent.update_target_dqn()
            dqn_agent.save(dqn_agent.episode_num)
            print(f"Network saved on episode {dqn_agent.episode_num}, " \
                + f"avg reward={np.average(reward_memory):.2f}, " \
                + f"wins={num_wins}")
            #mean_reward.append(np.mean(reward_memory))
            mean_reward.append(np.mean(all_reward))
            reward_memory = []
            # plot the num_wins:
            wins_number.append(num_wins)
            #plt.plot(wins_number)
            #plt.savefig('./plot/NumWhenWins/new_Env_old_NUMwins_all'+str(NUM_SHEPHERDS)+'dog'+str(MAX_NUM_AGENTS)+'sheep_when_episode='+str(dqn_agent.episode_num)+'.jpg')

            #if num_wins >= SAVE_TARGET*.8:
            if num_wins >= SAVE_TARGET * SUCCESS_METRICS:    
                plt.plot(all_reward)
                #plt.legend('episode reward', loc = 'upper left')
                plt.savefig('./plot/RewardWhenWins/new_network_new_Env_old_reward2/new_Env_old_reward_all'+str(NUM_SHEPHERDS)+'dog'+str(MAX_NUM_AGENTS)+'sheep_when_episode='+str(dqn_agent.episode_num)+'.jpg')
                print('reward plot painted')
                break
            num_wins = 0
        if dqn_agent.episode_num % 200 == 0 and dqn_agent.episode_num > 0:
            plt.plot(all_reward)
            #plt.legend('episode reward', loc = 'upper left')
            plt.savefig('./plot/allReward/new_network_new_Env_old_reward2/new_Env_GCM_reward_all.jpg')
            print('reward plot painted')
        
        if dqn_agent.episode_num % 300 == 5 and dqn_agent.episode_num > 0:
            plt.plot(mean_reward)
            #plt.legend('episode reward', loc = 'upper left')
            plt.savefig('./plot/meanReward/new_network_new_Env_old_reward2/new_Env_GCM_reward_mean.jpg')
            print('reward plot painted')
if __name__ == '__main__':
    # Construct an argument parser
    all_args = argparse.ArgumentParser()
    all_args.add_argument('-t', action='store_true')    # train (load from existing)
    all_args.add_argument('-r', action='store_true')    # reset (train from scratch)
    all_args.add_argument('-p', action='store_true')    # print model
    args = vars(all_args.parse_args())
    
    # initialize and load model
    dqn_agent = DQNAgent()
    if args['t']:
        if not args['r']:  
            dqn_agent.load(mode='train', print_model=args['p'])
        train(dqn_agent)
        #writer.flush()
    else:
        dqn_agent.load(mode='eval', print_model=args['p'])
        Multi_Environment2(True).run(dqn_agent)
        # exp_n = 100
        # success_count = 0
        # max_count = 0
        # average_count = 0
        # t_sum = 0
         
        # dqn_agent.load(mode='eval', print_model=args['p'])
        # for i in range(exp_n):
        #     success, max_agent_dist, average_agent_dist = multi_Environment(True).run(dqn_agent)
        #     t_per = pygame.time.get_ticks()
        #     success_count = success_count + success
        #     t_sum += t_per
            
          
        #     max_count += max_agent_dist
        #     average_count += average_agent_dist
            
        # print('success times in '+ str(exp_n)+ 'experiments: '+ str(success_count) )
        # print('average time used: '+str(t_sum/exp_n)+'ms')
        # print('average max distance of all agents among '+str(exp_n)+ 'experiments: ' + str(max_count/success_count))
        # print('average mean distance of all agents among '+str(exp_n)+ 'experiments: ' + str(average_count/success_count))
    