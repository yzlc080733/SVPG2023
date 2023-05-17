# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


# Model critic


def print_info(input_string=''):
    print('\033[94mMODEL_CRITIC_INFO|\033[0m', input_string)


class Critic(nn.Module):
    def __init__(self, input_size, output_size, small=False, dev=torch.device('cpu')):
        super(Critic, self).__init__()
        self.dev = dev
        self.small = small
        if self.small:
            self.fc1 = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU(),)
            self.fc2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),)
            self.fc3 = nn.Sequential(nn.Linear(64, output_size),)
        else:
            self.fc1 = nn.Sequential(nn.Linear(input_size, 1024), nn.ReLU(),)
            self.fc2 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(),)
            self.fc3 = nn.Sequential(nn.Linear(1024, output_size),)
        if self.small:
            # self.optimizer = torch.optim.RMSprop(params=self.parameters(), lr=0.01)
            # self.optimizer = torch.optim.SGD(params=self.parameters(), lr=0.01)
            self.optimizer = torch.optim.Adam(params=self.parameters(), lr=0.001)
        else:
            # self.optimizer = torch.optim.RMSprop(params=self.parameters(), lr=0.01)
            # self.optimizer = torch.optim.SGD(params=self.parameters(), lr=0.01)
            self.optimizer = torch.optim.Adam(params=self.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()
        self.to(self.dev)

    def forward(self, x):
        output = self.fc3(self.fc2(self.fc1(x)))
        return output

    def learn(self, value_predict, value_target):
        difference = value_target - value_predict
        loss = (difference * difference).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
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
            print_info('Error: current1 model currupted.')
            file_name = './log_model/' + name + '_2' + '.pt'
            self.load_state_dict(torch.load(file_name))
        print_info('load: %s' % file_name)
