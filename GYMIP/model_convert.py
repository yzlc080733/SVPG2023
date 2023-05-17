# -*- coding: utf-8 -*-
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import spikingjelly.clock_driven.ann2snn.examples.utils as utils
from spikingjelly.clock_driven.ann2snn import parser, classify_simulator
from spikingjelly.clock_driven import functional


# Model ANN2SNN


def print_info(input_string=''):
    print('\033[94mMODEL_MLP_INFO|\033[0m', input_string)





class NetBP(nn.Module):
    def __init__(self,
                layer_sizes=[784, 1000, 10],
                hid_activate='relu',
                hid_group_size=10,
            ):
        super(NetBP, self).__init__()
        # -----Get params--------
        if len(layer_sizes) != 3:
            print_info('Error in layer_sizes')
        self.layer_sizes = layer_sizes
        self.hid_activate = hid_group_size
        self.hid_group_size = hid_group_size
        # -----Hidden layer------
        if hid_activate == 'relu':
            self.hid = nn.Sequential(
                    nn.Linear(layer_sizes[0], layer_sizes[1]),
                    nn.ReLU(),
                )
        elif hid_activate == 'softmax':
            if layer_sizes[1] % hid_group_size != 0:
                print_info('Error in hid_group_size')
            else:
                hid_group_num = int(layer_sizes[1] / hid_group_size)
            self.hid = nn.Sequential(
                    nn.Linear(layer_sizes[0], layer_sizes[1]),
                    nn.Unflatten(1, (hid_group_num, hid_group_size)),
                    nn.Softmax(dim=2),
                    nn.Flatten(),
                )
        else:
            print_info('Error in hid_activate string')
        # -----Output layer------
        self.out = nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]),)
        self.optional_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        hid_x = self.hid(x)
        out_x = self.out(hid_x)
        return out_x

    def get_prediction(self, x):
        out_x = self.forward(x)
        out_x = self.optional_softmax(out_x)
        return out_x

    def load_model(self, name=''):
        try:
            file_name = './log_model/' + name + '_1' + '.pt'
            self.load_state_dict(torch.load(file_name))
        except:
            print_info('Error: current1 model currupted.')
            file_name = './log_model/' + name + '_2' + '.pt'
            self.load_state_dict(torch.load(file_name))
        print_info('load: %s' % (file_name))



class MLP_3:
    def __init__(self,
                layer_sizes=[784, 1000, 10],
                hid_activate='relu',
                hid_group_size=10,
                out_activate='softmax',
                optimizer_name='sgd',
                optimizer_learning_rate=0.001,
                snn_num_steps=25,
                entropy_ratio=0.0,
                device=torch.device('cpu')):
        self.ANN_model = NetBP(layer_sizes=layer_sizes,
                               hid_activate=hid_activate,
                               hid_group_size=hid_group_size)
        self.device = device
        self.ANN_model.to(self.device)
        self.model_ann_name = None
        self.temp_file_path = None

        self.model_convert_collect_list = torch.zeros([20000, layer_sizes[0]], device=self.device)
        self.model_convert_collect_index = 0
        self.model_collect_full = False

        self.snn, self.sim = None, None
        self.snn_num_steps = snn_num_steps

    def to(self, torch_device=torch.device('cpu')):       # replacement for torch model.to()
        return self

    def load_model_ann(self, name=''):
        update_name = name.replace('ann2snn', 'mlp3relu')
        print_info('ANN2SNN load: %s' % (update_name))
        self.ANN_model.load_model(update_name)
        self.model_ann_name = update_name
        self.temp_file_path = './ann2snn/temp_files_' + name + '/'

    def convert_model(self):
        onnxparser = parser(name=self.model_ann_name, kernel='onnx', log_dir=self.temp_file_path)
        snn = onnxparser.parse(self.ANN_model, self.model_convert_collect_list.to(self.device))
        torch.save(snn, os.path.join(self.temp_file_path, 'snn_' + self.model_ann_name + '.pkl'))

    def add_s_list(self, state_input):
        start_i = self.model_convert_collect_index
        end_i = min(self.model_convert_collect_list.shape[0], start_i + state_input.shape[0])
        self.model_convert_collect_list[start_i:end_i, :] = state_input[0:(end_i - start_i), :]
        self.model_convert_collect_index = end_i
        if end_i >= self.model_convert_collect_list.shape[0]:
            self.model_collect_full = True

    def load_model(self, name):
        self.snn = torch.load(os.path.join(self.temp_file_path, 'snn_' + self.model_ann_name + '.pkl')).to(self.device)
        self.sim = classify_simulator(self.snn, log_dir=self.temp_file_path + 'simulator', device=self.device)


    def __call__(self, x, *args, **kwargs):
        with torch.no_grad():
            functional.reset_net(self.snn)
            for snn_t in range(self.snn_num_steps):
                enc = self.sim.encoder(x).float()
                out = self.snn(enc)
                if snn_t == 0:
                    counter = out
                else:
                    counter = counter + out
            torch.cuda.empty_cache()
        return counter, None

    def add_noise_abs(self, noise_type, noise_param):
        if not noise_type in ['gaussian', 'uniform']:
            print_info('Error network noise type')
        with torch.no_grad():
            for param in self.snn.parameters():
                if noise_type == 'gaussian':
                    param.add_(torch.randn(param.size()).to(self.device) * noise_param)
                if noise_type == 'uniform':
                    param.add_((torch.rand(param.size()).to(self.device) - 0.5) * 2 * noise_param)

    def add_noise_relative(self, noise_type, noise_param):
        if not noise_type in ['gaussian', 'uniform']:
            print_info('Error network noise type')
        with torch.no_grad():
            for param in self.snn.parameters():
                mean_value = np.mean(np.abs(param.cpu().numpy()))
                if noise_type == 'gaussian':
                    param.add_(torch.randn(param.size()).to(self.device) * noise_param * mean_value)
                if noise_type == 'uniform':
                    param.add_((torch.rand(param.size()).to(self.device) - 0.5) * 2 * noise_param * mean_value)



if __name__ == "__main__":
    print_info('MLP model start')
    model = MLP_3(layer_sizes=[784, 1000, 10], hid_activate='softmax', hid_group_size=10, out_activate='softmax')
    device = torch.device('cuda:0')
    model = model.to(device)
    input_array = torch.randn([1000, 784]).to(device)
    output_array = model(input_array)
    model.save_model('model_mlp_test')



print('\033[91mFINISH: model_mlp\033[0m')


