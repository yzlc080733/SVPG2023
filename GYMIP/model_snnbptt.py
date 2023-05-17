# -*- coding: utf-8 -*-
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch
from snntorch import surrogate
import numpy as np



# Model SNN LIF BPTT



def print_info(input_string=''):
    print('\033[94mMODEL_SNNBPTT_INFO|\033[0m', input_string)




class SNNBPTT3(nn.Module):
    def __init__(self,
                layer_sizes=[784, 1000, 10],
                snn_num_steps = 100,
                optimizer_name='sgd',
                optimizer_learning_rate=0.0001,
                entropy_ratio=0.0,
                dev=torch.device('cpu'),
            ):
        super(SNNBPTT3, self).__init__()
        # -----Get params--------
        if len(layer_sizes) != 3:
            print_info('Error in layer_sizes')
        self.layer_sizes = layer_sizes
        self.snn_num_steps = snn_num_steps
        self.snn_res_step_num = snn_num_steps           # average the full sequence for firing rate
        self.optimizer_name = optimizer_name
        self.optimizer_learning_rate = optimizer_learning_rate
        self.entropy_ratio = entropy_ratio
        # -----Layers------------
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = torch.nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc2 = torch.nn.Linear(layer_sizes[1], layer_sizes[2])
        self.lif1 = snntorch.Leaky(beta=0.9, spike_grad=spike_grad, learn_beta=True)
        self.lif2 = snntorch.Leaky(beta=0.9, spike_grad=spike_grad, learn_beta=True)
        # -----optimizer---------
        if optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(params=self.parameters(),
                lr=optimizer_learning_rate, weight_decay=0, momentum=0)
        elif optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(params=self.parameters(),
                lr=optimizer_learning_rate)
        elif optimizer_name == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params=self.parameters(),
                lr=optimizer_learning_rate)
        else:
            print_info('Error in optimizer_name')
        # Device
        self.dev = dev
        self.to(self.dev)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        device = next(self.parameters()).device
        spk2_rec = torch.zeros([self.snn_num_steps, x.shape[0], self.layer_sizes[2]]).to(device)
        mem2_rec = []
        for spk_step_i in range(self.snn_num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec[spk_step_i, :, :] = spk2
            mem2_rec.append(mem2)
        stk_mean = torch.mean(spk2_rec[-self.snn_res_step_num:, :, :], dim=0)
        action_probability = F.softmax(stk_mean, dim=1)
        return [action_probability, None]

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
        # print_info('load: %s' % (file_name))

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
        if not noise_type in ['gaussian', 'uniform']:
            print_info('Error network noise type')
        device = next(self.parameters()).device
        with torch.no_grad():
            param_list = [_ for _ in self.parameters()]
            for param in param_list[0:4]:
                if noise_type == 'gaussian':
                    param.add_(torch.randn(param.size()).to(device) * noise_param)
                if noise_type == 'uniform':
                    param.add_((torch.rand(param.size()).to(device) - 0.5) * 2 * noise_param)

    def add_noise_relative(self, noise_type, noise_param):
        device = next(self.parameters()).device
        with torch.no_grad():
            param_list = [_ for _ in self.parameters()]
            for param in param_list[0:4]:
                mean_value = np.mean(np.abs(param.cpu().numpy()))
                if noise_type == 'gaussian':
                    param.add_(torch.randn(param.size()).to(device) * noise_param * mean_value)
                if noise_type == 'uniform':
                    param.add_((torch.rand(param.size()).to(device) - 0.5) * 2 * noise_param * mean_value)




if __name__ == "__main__":
    print_info('SNNBPTT3 model start')
    model = SNNBPTT3(
            layer_sizes=[784, 1000, 10],
            snn_num_steps = 100,
            optimizer_name='sgd',
            optimizer_learning_rate=0.0001,
        )
    device = torch.device('cuda:0')
    model = model.to(device)
    input_array = torch.randn([1000, 784]).to(device)
    output_array = model(input_array)
    model.save_model('model_SNNBPTT3_test')



print('\033[91mFINISH: model_snnbptt\033[0m')


