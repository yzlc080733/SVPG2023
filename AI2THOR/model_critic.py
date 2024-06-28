# -*- coding: utf-8 -*-
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F




# Model MLP




def print_mlp_info(input_string=''):
    MLP_MODEL_INFO_MESSAGE = '\033[94mMODEL_MLP_INFO|\033[0m'
    print(MLP_MODEL_INFO_MESSAGE, input_string)



class Critic(nn.Module):
    def __init__(self,
                input_size,
                output_size,
                small=False,
            ):
        super(Critic, self).__init__()
        self.small = small
        # -----Hidden layer------
        if self.small:
            self.fc1 = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(),)
            self.fc2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),)
            self.fc3 = nn.Sequential(nn.Linear(128, output_size),)
        else:
            self.fc1 = nn.Sequential(nn.Linear(input_size, 2048), nn.ReLU(),)
            self.fc2 = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(),)
            self.fc3 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(),)
            self.fc4 = nn.Sequential(nn.Linear(1024, output_size),)
        
        if self.small:
            # self.optimizer = torch.optim.SGD(params=self.parameters(),
            #     lr=0.01, weight_decay=0, momentum=0)
            # self.optimizer = torch.optim.SGD(params=self.parameters(),
            #     lr=0.001, weight_decay=0, momentum=0)
            # self.optimizer = torch.optim.SGD(params=self.parameters(), lr=0.01)
            self.optimizer = torch.optim.RMSprop(params=self.parameters(), lr=0.01)
        else:
            self.optimizer = torch.optim.Adam(params=self.parameters(), lr=0.001)
        
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        if self.small:
            output = self.fc3(self.fc2(self.fc1(x)))
        else:
            output = self.fc4(self.fc3(self.fc2(self.fc1(x))))
        return output

    def learn(self, a_predict, a_target):
        loss = self.loss_function(a_predict, a_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()

    def save_model(self, name=''):
        file_name = './log_model/' + name + '_1' + '.pt'
        torch.save(self.state_dict(), file_name)
        file_name = './log_model/' + name + '_2' + '.pt'
        torch.save(self.state_dict(), file_name)

    def load_model(self, name=''):
        try:
            file_name = './log_model/' + name + '_1' + '.pt'
            self.load_state_dict(torch.load(file_name))
        except:
            print_mlp_info('Error: current1 model currupted.')
            file_name = './log_model/' + name + '_2' + '.pt'
            self.load_state_dict(torch.load(file_name))
        print_mlp_info('load: %s' % (file_name))
