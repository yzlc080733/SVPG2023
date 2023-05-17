# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import numpy as np

# Model MLP


def print_info(input_string=''):
    print('\033[94mMODEL_MLP_INFO|\033[0m', input_string)


class MLP_3(nn.Module):
    def __init__(self, layer_sizes=[784, 1000, 10], hid_activate='relu',
                 hid_group_size=10,
                 out_activate='softmax',
                 optimizer_name='sgd',
                 optimizer_learning_rate=0.001,
                 entropy_ratio=0.0,
                 dev=torch.device('cpu')
                 ):
        super(MLP_3, self).__init__()
        # -----Get params--------
        if len(layer_sizes) != 3:
            print_info('Error in layer_sizes')
        self.layer_sizes = layer_sizes
        self.hid_activate = hid_group_size
        self.hid_group_size = hid_group_size
        self.out_activate = out_activate
        self.optimizer_name = optimizer_name
        self.optimizer_learning_rate = optimizer_learning_rate
        self.entropy_ratio = entropy_ratio
        self.dev = dev
        # -----Hidden layer------
        if hid_activate == 'relu':
            self.hid = nn.Sequential(
                    nn.Linear(layer_sizes[0], layer_sizes[1]),
                    nn.ReLU(),
                )
        elif hid_activate == 'softmax':
            if layer_sizes[1] % hid_group_size != 0:
                hid_group_num = None
                print_info('Error in hid_group_size')
            else:
                hid_group_num = int(layer_sizes[1] / hid_group_size)
            self.hid = nn.Sequential(
                    nn.Linear(layer_sizes[0], layer_sizes[1]),
                    nn.Unflatten(1, (hid_group_num, hid_group_size)),
                    nn.Softmax(dim=2),
                    nn.Flatten(), )
        else:
            print_info('Error in hid_activate string')
        # -----Output layer------
        if out_activate == 'relu':
            self.out = nn.Sequential(
                    nn.Linear(layer_sizes[1], layer_sizes[2]),
                    nn.ReLU(),
                )
        elif out_activate == 'softmax':
            self.out = nn.Sequential(
                    nn.Linear(layer_sizes[1], layer_sizes[2]),
                    nn.Softmax(dim=1),
                )
        elif out_activate == 'none':
            self.out = nn.Sequential(
                    nn.Linear(layer_sizes[1], layer_sizes[2]),
                )
        else:
            print_info('Error in out_activate string')
        # -----optimizer---------
        if optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(params=self.parameters(),
                                             lr=optimizer_learning_rate,
                                             weight_decay=0, momentum=0)
        elif optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(params=self.parameters(),
                                              lr=optimizer_learning_rate)
        elif optimizer_name == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params=self.parameters(),
                                                 lr=optimizer_learning_rate)
        else:
            print_info('Error in optimizer_name')
        # Device
        self.to(self.dev)

    def forward(self, x):
        hid_x = self.hid(x)
        out_x = self.out(hid_x)
        return [out_x, None]

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
        # print_info('load: %s' % file_name)

    def learn_ppo(self, a_logprob, old_logprob, advantage, epsilon_clip, a_entropy, **kwargs):
        ratio = torch.exp(a_logprob - old_logprob.detach())
        advantage_squeeze = advantage
        target_1 = ratio * advantage_squeeze
        target_2 = torch.clamp(ratio, 1-epsilon_clip, 1+epsilon_clip) * advantage_squeeze
        loss = -torch.min(target_1, target_2).mean() - self.entropy_ratio * a_entropy.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn_reinforce(self, a_logprob, advantage, a_entropy, **kwargs):
        loss = -(a_logprob * advantage).mean() - self.entropy_ratio * a_entropy.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def add_noise_abs(self, noise_type, noise_param):
        if noise_type not in ['gaussian', 'uniform']:
            print_info('Error network noise type')
        with torch.no_grad():
            for param in self.parameters():
                if noise_type == 'gaussian':
                    param.add_(torch.randn(param.size()).to(self.dev) * noise_param)
                if noise_type == 'uniform':
                    param.add_((torch.rand(param.size()).to(self.dev) - 0.5) * 2 * noise_param)

    def add_noise_relative(self, noise_type, noise_param):
        with torch.no_grad():
            for param in self.parameters():
                mean_value = np.mean(np.abs(param.cpu().numpy()))
                if noise_type == 'gaussian':
                    param.add_(torch.randn(param.size()).to(self.dev) * noise_param * mean_value)
                if noise_type == 'uniform':
                    param.add_((torch.rand(param.size()).to(self.dev) - 0.5) * 2 * noise_param * mean_value)


if __name__ == "__main__":
    print_info('MLP model start')
    if not os.path.exists('./log_model'):
        os.mkdir('./log_model')
    device = torch.device('cuda:0')
    model = MLP_3(layer_sizes=[784, 1000, 10], hid_activate='softmax',
                  hid_group_size=10, out_activate='softmax',
                  dev=device)
    input_array = torch.randn([1000, 784]).to(device)
    output_array = model(input_array)
    model.save_model('model_mlp_test')



print('\033[91mFINISH: model_mlp\033[0m')


