import pygame
import numpy as np
from parameters import *
from functools import reduce
import threading

FPS = 80
fpsClock = pygame.time.Clock()

class Multi_Environment2:
    def __init__(self, render):
        self.padding = np.array([30, 30])   # padding for GUI
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((FIELD_LENGTH+2*self.padding[0],
                                                    FIELD_LENGTH+2*self.padding[1]))
        self.action = 0
        self.reset_enabled = True
        # initalized in reset function
        self.shepherd = None
        self.target = None
        self.agents = None
        self.num_agents = 0
        self.num_nearest = 0
        self.reset()
        self.max_agent_dist = 0
        self.average_agent_dist = 0

    def reset(self):
        self.num_agents = MAX_NUM_AGENTS # for 1 sheep, 2 shepherd situation
        self.num_shepherd = NUM_SHEPHERDS
        #self.num_agents = np.random.randint(1, MAX_NUM_AGENTS+1) # 1-5 agents
        self.num_nearest = self.num_agents-1
        # shepherd and target start at same random location
        # self.shepherd = np.random.randint(FIELD_LENGTH, size=2)
        # self.target = np.copy(self.shepherd)
        #self.shepherd = np.array([[0, 0]]) # shpherd 初始位置在（0，0）
        self.shepherd = np.array([[0, 0],[10,0]]) # shpherd 初始位置在（0，0）
        #self.shepherd = np.random.randint(FIELD_LENGTH-1, size=2)  # random initial position of shepherd
        self.target = np.array([FIELD_LENGTH-10, FIELD_LENGTH-10])
        self.agents = R_S//2 + np.random.randint(FIELD_LENGTH*.2, size=(self.num_agents, 2))  # RS//2=20//2=10, randint(0,50) with size (num agents,2)
        # agents = (num_agents, 2) = [[x1,y1],[x2,y2]......[xn,yn]]
    def step(self, action):
        
        
        # map action [0, 3] to direction vector, outputsize = 4, how to define these 4 actions.
        
        #shep_direction = np.array([[(1-action[i]%2)*(-action[i]+1), (action[i]%2)*(-action[i]+2)] for i in range(self.num_shepherd)])
        shep_direction = np.array([[(1-action[0]%2)*(-action[0]+1), (action[0]%2)*(-action[0]+2)], [(1-action[1]%2)*(-action[1]+1), (action[1]%2)*(-action[1]+2)]])  #two shepherds velocity
        # update shepherd
        self.shepherd += shep_direction # shepherd的坐标
        self.shepherd = self.shepherd.clip(0, FIELD_LENGTH-1) # 超出边框时，卡住不动，不可循环。

        # update agents
        # lookup table distances[i][j] = dist(self.agent[i], self.agent[j])
        # if i >= j, distances[i][j] = 0
        distances = [[self.dist(self.agents[i], self.agents[j]) if i < j else 0 for j in range(self.num_agents)] 
                    for i in range(self.num_agents)]
        for i in range(self.num_agents):
            # local repulsion
            v_a_ = [self.unit_vect(self.agents[i], self.agents[j]) for j in range(self.num_agents) 
                            if i != j and distances[min(i,j)][max(i,j)] < R_R]
            # if two agents in same location, unit_vect returns zero; need to map to random unit vector
            for v_index in range(len(v_a_)):
                if (v_a_[v_index]==0).all():
                    v_a_[v_index] = self.rand_unit()
            v_a = 0 if len(v_a_) == 0 else self.unit_vect(reduce(np.add, v_a_))

            # attracted to local center of mass of nearest agents (ignore self at index 0)
            v_c = 0
            if self.num_nearest > 0:
                sorted_agents = sorted(self.agents, key=lambda x: self.dist(self.agents[i], x))
                nearest_agents = sorted_agents[1:self.num_nearest+1]     # ignore self at index 0
                lcm = reduce(np.add, nearest_agents)/self.num_nearest
                v_c = self.unit_vect(lcm, self.agents[i])

            minmum_dist = self.min_dist(self.shepherd, self.agents[i])
            if (minmum_dist > R_S):
                # agent outside of shepherd detection radius, only consider local repulsion
                self.agents[i] = np.rint(self.agents[i] + self.unit_vect(P_A*v_a + P_C*v_c))
                # elif:
                #     self.agents[i] = np.rint(self.agents[i] + self.unit_vect(P_A*v_a + P_C*v_c))
                # elif:
                #     self.agents[i] = np.rint(self.agents[i] + self.unit_vect(P_A*v_a + P_C*v_c))
            else:
                # repelled from shepherd (if in same location, run towards center of board)
                # 此处开始对shepherd for循环
                v_s = 0
                for j in range(self.num_shepherd):
                    if (self.dist(self.shepherd[j], self.agents[i]) <= R_S):
                        v_s = v_s + self.unit_vect(self.agents[i], self.shepherd[j])
                        if (v_s==0).all():
                            v_s = v_s + self.unit_vect(self.target, self.agents[i])

                self.agents[i] = np.rint(self.agents[i] + self.unit_vect(P_A*v_a + P_C*v_c + P_S*v_s))
        
        self.agents = self.agents.clip(0, FIELD_LENGTH-1)
        

        # give a reward based on GCM distance to target
        gcm = reduce(np.add, self.agents)/self.num_agents #Is it right?
        gcm_to_target = self.dist(self.target, gcm)
        #reward = -(gcm_to_target-TARGET_RADIUS)/R_S  # encourage shepherd to approach target. The reward should increase when the training process goes on.
        reward = -(gcm_to_target-TARGET_RADIUS)/FIELD_LENGTH
        
        # encourgae shepherd and sheep move towards target
        # encourage shepherd to approach a: 2  agent situation
        shep_to_agent = 0
        #shep_to_target = 0
        for shepherd in self.shepherd: 
            shep_to_agent += self.dist(gcm, shepherd)
            #shep_to_target += self.dist(shepherd, self.target)
        reward -= shep_to_agent/R_S - 1
        #reward -= shep_to_target/R_S
        #reward -= shep_to_target/FIELD_LENGTH
        

        # return game_won = True if furthest agent within target_radius
        game_over = False
        max_agent_dist = max([self.dist(a, self.target) for a in self.agents])
        #max_list.append(max_agent_dist)
        average_agent_dist = np.mean([self.dist(a, self.target) for a in self.agents])
        #average_list.append(average_agent_dist)
        if max_agent_dist < TARGET_RADIUS:
            reward = 1000
            game_over = True # means mission completed
            max_agent_dist = max_agent_dist
            average_agent_dist = average_agent_dist
            
            #self.max_agent_dist = max_agent_dist
            #self.average_agent_dist = average_agent_dist
            
        return reward, game_over

    # distance from [a1, a2] to [b1, b2]
    def dist(self, a, b=np.array([0, 0])):
        return np.linalg.norm(a-b)

    def min_dist(self, a, b): # a 是 shepherd list，b是一只sheep的位置,批量比较大小
        dist_ls = []
        for i in range(np.shape(a)[0]):
            DS_dist = self.dist(a[i], b)
            dist_ls.append(DS_dist)
        return np.min(dist_ls)
        # 

    # unit vect from [b1, b2] to [a1, a2]
    def unit_vect(self, a, b=np.array([0, 0])):
        if self.dist(a, b) == 0:
            return np.array([0, 0])
        return (a-b)/self.dist(a, b)

    # return random unit vector
    def rand_unit(self):
        return self.unit_vect([np.random.rand()-.5, np.random.rand()-.5])

    def get_key_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            self.action = 0
        elif keys[pygame.K_DOWN]:
            self.action = 1
        elif keys[pygame.K_LEFT]:
            self.action = 2
        elif keys[pygame.K_UP]:
            self.action = 3
        return self.action      

    def render(self):
        self.screen.fill((0, 0, 0))
        [pygame.draw.circle(self.screen, (255, 0, 0), tuple(self.padding + b), 1, 0) for b in self.shepherd]
        pygame.draw.circle(self.screen, (0, 255, 255), tuple(self.padding + self.target), 1, 0)
        [pygame.draw.circle(self.screen, (0, 255, 0), tuple(self.padding + a), 1, 0)
            for a in self.agents]
        pygame.display.update()

    def get_state(self):
        if CNN_NETWORK:
            state = np.zeros((FIELD_LENGTH, FIELD_LENGTH), dtype=np.int32)
            # 100 := shepherd
            # 200 := target
            # x % 100 := number of sheep at given coordinates
            state[self.shepherd[0], self.shepherd[1]] += 100
            state[self.target[0], self.target[1]] += 200
            for i, j in self.agents:
                state[i, j] += 1
            # add channel dimension
            return np.expand_dims(state, axis=0)
        else:
            # pad self.agents with coordinates (-100, 100) to maintain constant dimensions
            padded_agents = self.agents.flatten() # [x1,y1,x2,y2......]
            padded_agents = np.pad(padded_agents, (0, 2*MAX_NUM_AGENTS-len(padded_agents)), 
                                'constant', constant_values=-100)  # 补齐操作

            # [x1,y1,...,x5,y5], [xs,ys], [xt,yt] concatenate: [x1,y1,...,x5,y5, xs,ys, xt,yt] in shape of (1, 7*2)
            return np.concatenate((padded_agents, self.shepherd.flatten(), self.target.flatten()), axis=0)
            #return np.concatenate((padded_agents, self.shepherd.flatten()), axis=0)

    def pygame_running(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def tick(self, fps=FPS):
        fpsClock.tick(fps)

    def enable_reset(self):
        self.reset_enabled = True

    def run(self, dqn_agent=None):        
        # game loop
        average_list = 0
        max_list = 0
        success = 0
        n = 0
        
        while self.pygame_running():
            self.render()

            if dqn_agent is None:
                self.get_key_input()
            else:
                self.action = dqn_agent.get_action(self.get_state())
            _, game_over= self.step(self.action)
            
            #max_list.append(max_agent_dist)
        
            #average_list.append(average_agent_dist)
            
            keys = pygame.key.get_pressed()  
            
            n +=1 
           
            if game_over or (keys[pygame.K_r] and self.reset_enabled):
                #max_list += max_agent_dist
        
                #average_list += average_agent_dist
                
                success += 1
                self.reset()
                self.reset_enabled = False
                threading.Timer(.5, self.enable_reset).start()
                
            
              
            if keys[pygame.K_ESCAPE] or n > 10000: # to avoid the agent stuck in one position
                break
            
            
            self.tick()
        return success, max_list, average_list

        
            
#if __name__ == "__main__":

    #exp_n = 10
    #success_count = 0
    #for i in range(exp_n):
    #success = Environment(True).run()
    #success_count = success_count + success
        
    #print(success)
    
        
    