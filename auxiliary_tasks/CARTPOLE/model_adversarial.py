# -*- coding: utf-8 -*-
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F




# Model Adversarial



def print_info(input_string=''):
    print('\033[94mMODEL_Adversarial_INFO|\033[0m', input_string)



class Adversarial(nn.Module):
    def __init__(self,
                input_size,
                output_size
            ):
        super(Adversarial, self).__init__()
        # -----Hidden layer------
        self.fc1 = nn.Sequential(nn.Linear(input_size, 2048), nn.ReLU(),)
        self.fc2 = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(),)
        self.fc3 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(),)
        self.fc4 = nn.Sequential(nn.Linear(1024, output_size), nn.Softmax(dim=1))
        
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        output = self.fc4(self.fc3(self.fc2(self.fc1(x))))
        return output

    def learn(self, s1, a_predict, a_target):
        loss = self.loss_function(a_predict, a_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()
