import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from threading import Thread
from parameters import *

class DQN(nn.Module):
    def __init__(self, lr, input_size, output_size):
        super(DQN, self).__init__()
        # two hidden layers
        hidden_size1 = 128
        hidden_size2 = 256
        self.input = nn.Linear(input_size, hidden_size1)
        self.hidden1 = nn.Linear(hidden_size1, hidden_size2)
        self.hidden2 = nn.Linear(hidden_size2, hidden_size1)
        self.output = nn.Linear(hidden_size1, output_size) # output size should be = action_size**NUM_SHEPHERD
        self.shepherd_num = 2
        self.action_size = 4
        

        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x =  F.relu(self.input(x))
        x =  F.relu(self.hidden1(x))
        x =  F.relu(self.hidden2(x))
        # to add the softmax layer
        output = self.output(x) # 2 shepherd situation
        return output.view(-1, self.action_size, self.action_size)   

    def save(self, episode_num, file_name):
        # create thread to prevent keyboard interrupt
        a = Thread(target=self.save_thread, args=(episode_num, file_name))
        a.start()
        a.join()

    def save_thread(self, episode_num, file_name):
        # save model
        folder_path = FOLDER_PATH
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = os.path.join(folder_path, file_name)
        T.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_num': episode_num
        }, file_name)

    # load model and return episode number
    def load(self, file_name):
        checkpoint = T.load('./model/' + file_name, map_location=self.device)
        #checkpoint = T.load(file_name, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['episode_num']
